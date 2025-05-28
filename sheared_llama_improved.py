import os
import json
import time
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import torch
import torch.nn as nn
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from composer.models import ComposerModel
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from peft import LoraConfig, get_peft_model
from torch.nn.utils import prune
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    logger.warning("flash-attn not available. Attention optimization will be disabled.")
    FLASH_AVAILABLE = False
    flash_attn_func = None
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    logger.warning("bitsandbytes not available. Quantization will be disabled.")
    BNB_AVAILABLE = False
    bnb = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImprovementConfig:
    model_name: str = "princeton-nlp/Sheared-LLaMA-1.3B"
    output_dir: str = "./improved_sheared_llama"
    quantization_bits: int = 4
    batch_size: int = 2  # Reduced for typical Colab T4
    mixed_precision: str = "fp16"
    max_length: int = 512
    num_epochs: int = 1  # Reduced for demo
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    eval_batch_size: int = 2
    use_gradient_checkpointing: bool = True
    run_quantization: bool = BNB_AVAILABLE  # Only enable if bitsandbytes is available
    run_attention_optimization: bool = FLASH_AVAILABLE  # Only enable if flash-attn is available
    run_pruning: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    pruning_sparsity: float = 0.5

    def to_json(self, path: str):
        directory = os.path.dirname(path)
        logger.info(f"Attempting to save config to: {path}")
        if directory:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Config saved successfully to {path}")

    @classmethod
    def from_json(cls, path: str):
        logger.info(f"Attempting to load config from: {path}")
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"No such file: {path}")
        with open(path, 'r') as f:
            return cls(**json.load(f))

class SystemMonitor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def measure_memory_usage(self) -> Dict[str, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            return {
                "model_memory_gb": torch.cuda.memory_allocated() / 1e9,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9
            }
        return {"model_memory_gb": 0.0, "peak_memory_gb": 0.0}

class ModelLoader:
    def __init__(self, config: ImprovementConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            logger.warning("Running on CPU. This will be very slow. Consider using a GPU if available.")
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def load_model(self, quantized: bool = False):
        model_kwargs = {
            "torch_dtype": torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16,
            "device_map": "auto" if self.device.type == "cuda" else None
        }
        
        if quantized and BNB_AVAILABLE:
            try:
                model_kwargs["quantization_config"] = bnb.BitsAndBytesConfig(
                    load_in_4bit=self.config.quantization_bits == 4,
                    load_in_8bit=self.config.quantization_bits == 8,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            except Exception as e:
                logger.warning(f"Failed to set up quantization: {e}. Falling back to non-quantized model.")
                quantized = False
        
        try:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs)
            if self.config.use_gradient_checkpointing:
                model.gradient_checkpointing_enable()
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

class ModelAnalyzer:
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def benchmark_inference_speed(self, num_samples: int = 50, seq_length: int = 128) -> Dict[str, float]:
        self.model.eval()
        input_ids = torch.randint(0, len(self.tokenizer), (num_samples, seq_length)).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            _ = self.model(input_ids)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        tokens_per_second = (num_samples * seq_length) / elapsed
        return {"tokens_per_second": tokens_per_second}
    
    def measure_memory_usage(self) -> Dict[str, float]:
        return SystemMonitor().measure_memory_usage()
    
    def compute_perplexity(self, dataset) -> float:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in dataset:
            inputs = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs, labels=labels)
                total_loss += outputs.loss.item() * inputs.size(0)
                total_tokens += inputs.size(0)
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()

class QuantizationOptimizer:
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def apply_dynamic_quantization(self):
        if not BNB_AVAILABLE:
            logger.warning("bitsandbytes not available, skipping quantization")
            return self.model
        return torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)

class AttentionOptimizer:
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def apply_attention_optimization(self):
        if not FLASH_AVAILABLE:
            logger.warning("flash-attn not available, skipping attention optimization")
            return self.model
        
        try:
            for layer in self.model.model.layers:
                if hasattr(layer, 'self_attn'):
                    original_forward = layer.self_attn.forward
                    def new_forward(*args, **kwargs):
                        try:
                            return flash_attn_func(*args, **kwargs)
                        except Exception as e:
                            logger.warning(f"Flash attention failed, falling back to original: {e}")
                            return original_forward(*args, **kwargs)
                    layer.self_attn.forward = new_forward
            return self.model
        except Exception as e:
            logger.warning(f"Failed to apply attention optimization: {e}")
            return self.model

class PruningOptimizer:
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, config: ImprovementConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def apply_structured_pruning(self):
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(self.model, lora_config)
        for i, layer in enumerate(model.base_model.model.layers):
            sparsity = self.config.pruning_sparsity * (1 - i / len(model.base_model.model.layers))
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=sparsity)
        return model

class ModelTrainer:
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, config: ImprovementConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def prepare_dataset(self, dataset_name: str = "togethercomputer/RedPajama-Data-1T-Sample", split: str = "train"):
        logger.info(f"Loading dataset: {dataset_name} (split: {split})")
        dataset = load_dataset(dataset_name, split=split)
        
        def curriculum_filter(example, idx):
            quality_domains = ["wiki", "arxiv"]
            if idx < len(dataset) // 2:
                return any(domain in example.get("meta", {}).get("red_pajama_subset", "") for domain in quality_domains)
            return True
        
        dataset = dataset.filter(curriculum_filter)
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"✓ Dataset prepared - {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def create_trainer(self, train_dataset, eval_dataset=None):
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            fp16=self.config.mixed_precision == "fp16",
            bf16=self.config.mixed_precision == "bf16",
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

class ResultsVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_performance_comparison(self, baseline_metrics: Dict, improved_metrics: Dict, output_path: str):
        labels = ["Tokens/Second", "Memory (GB)", "Perplexity"]
        baseline_values = [
            baseline_metrics["inference_speed"]["tokens_per_second"],
            baseline_metrics["memory_usage"]["model_memory_gb"],
            baseline_metrics.get("perplexity", 0)
        ]
        improved_values = [
            improved_metrics["inference_speed"]["tokens_per_second"],
            improved_metrics["memory_usage"]["model_memory_gb"],
            improved_metrics.get("perplexity", 0)
        ]
        
        x = range(len(labels))
        plt.figure(figsize=(10, 6))
        plt.bar([i - 0.2 for i in x], baseline_values, 0.4, label="Baseline", color="#36A2EB")
        plt.bar([i + 0.2 for i in x], improved_values, 0.4, label="Improved", color="#FF6384")
        plt.xticks(x, labels)
        plt.ylabel("Value")
        plt.title("Baseline vs Improved Performance (RedPajama)")
        plt.legend()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

class ExperimentRunner:
    def __init__(self, config: ImprovementConfig):
        self.config = config
        self.monitor = SystemMonitor()
        self.results = {}
    
    def run_complete_experiment(self, dataset_name: str = "togethercomputer/RedPajama-Data-1T-Sample"):
        logger.info("Starting enhanced Sheared LLaMA experiment with RedPajama")
        
        # Load model and tokenizer
        model_loader = ModelLoader(self.config)
        tokenizer = model_loader.load_tokenizer()
        baseline_model = model_loader.load_model(quantized=False)
        
        # Analyze baseline performance
        logger.info("Analyzing baseline performance")
        analyzer = ModelAnalyzer(baseline_model, tokenizer)
        trainer = ModelTrainer(baseline_model, tokenizer, self.config)
        eval_dataset = trainer.prepare_dataset(dataset_name, split="train[:1%]")
        baseline_metrics = {
            "inference_speed": analyzer.benchmark_inference_speed(num_samples=50),
            "memory_usage": analyzer.measure_memory_usage(),
            "perplexity": analyzer.compute_perplexity(eval_dataset)
        }
        
        print(f"Baseline Performance (Sheared LLaMA 1.3B, RedPajama):")
        print(f"  Tokens/second: {baseline_metrics['inference_speed']['tokens_per_second']:.1f}")
        print(f"  Memory: {baseline_metrics['memory_usage']['model_memory_gb']:.2f} GB")
        print(f"  Perplexity: {baseline_metrics['perplexity']:.2f}")
        
        # Apply enhancements
        logger.info("Applying enhancements")
        improved_metrics = {}
        
        # Quantization
        if self.config.run_quantization:
            logger.info("Applying quantization...")
            quant_model = model_loader.load_model(quantized=True)
            quant_analyzer = ModelAnalyzer(quant_model, tokenizer)
            improved_metrics['quantization'] = {
                "inference_speed": quant_analyzer.benchmark_inference_speed(num_samples=50),
                "memory_usage": quant_analyzer.measure_memory_usage(),
                "perplexity": quant_analyzer.compute_perplexity(eval_dataset)
            }
        
        # Attention optimization
        if self.config.run_attention_optimization:
            logger.info("Applying attention optimization...")
            attn_optimizer = AttentionOptimizer(baseline_model, tokenizer)
            attn_model = attn_optimizer.apply_attention_optimization()
            attn_analyzer = ModelAnalyzer(attn_model, tokenizer)
            improved_metrics['attention_optimization'] = {
                "inference_speed": attn_analyzer.benchmark_inference_speed(num_samples=50),
                "memory_usage": attn_analyzer.measure_memory_usage(),
                "perplexity": attn_analyzer.compute_perplexity(eval_dataset)
            }
        
        # Pruning with LoRA
        if self.config.run_pruning:
            logger.info("Applying LoRA-based pruning...")
            prune_optimizer = PruningOptimizer(baseline_model, tokenizer, self.config)
            pruned_model = prune_optimizer.apply_structured_pruning()
            prune_analyzer = ModelAnalyzer(pruned_model, tokenizer)
            improved_metrics['lora_pruning'] = {
                "inference_speed": prune_analyzer.benchmark_inference_speed(num_samples=50),
                "memory_usage": prune_analyzer.measure_memory_usage(),
                "perplexity": prune_analyzer.compute_perplexity(eval_dataset)
            }
        
        # Save results
        logger.info("Saving results")
        self.save_results(baseline_metrics, improved_metrics)
        
        # Generate visualizations
        logger.info("Generating visualizations")
        visualizer = ResultsVisualizer(self.config.output_dir)
        for method, metrics in improved_metrics.items():
            visualizer.plot_performance_comparison(
                baseline_metrics, 
                metrics,
                os.path.join(self.config.output_dir, f"performance_comparison_{method}.png")
            )
        
        return baseline_metrics, improved_metrics
    
    def save_results(self, baseline_metrics: Dict, improved_metrics: Dict):
        results = {
            "config": asdict(self.config),
            "baseline_metrics": baseline_metrics,
            "improved_metrics": improved_metrics,
            "experiment_timestamp": time.time(),
            "dataset": "RedPajama-Data-1T-Sample"
        }
        
        results_path = os.path.join(self.config.output_dir, "experiment_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        summary_data = [{
            "Method": "Baseline (Sheared LLaMA 1.3B)",
            "Tokens/Second": baseline_metrics['inference_speed']['tokens_per_second'],
            "Memory_GB": baseline_metrics['memory_usage']['model_memory_gb'],
            "Perplexity": baseline_metrics['perplexity']
        }]
        
        for method, metrics in improved_metrics.items():
            summary_data.append({
                "Method": method.replace('_', ' ').title(),
                "Tokens/Second": metrics['inference_speed']['tokens_per_second'],
                "Memory_GB": metrics['memory_usage']['model_memory_gb'],
                "Perplexity": metrics['perplexity']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.config.output_dir, "results_summary_redpajama.csv")
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Results saved to {self.config.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Sheared LLaMA Model Improvement")
    parser.add_argument("--config", type=str, default="./configs/llama2/1.3b.yaml", help="Path to YAML configuration file")
    parser.add_argument("--mode", choices=["optimize", "train", "evaluate"], default="optimize", help="Experiment mode")
    parser.add_argument("--model-name", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B", help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="./improved_sheared_llama", help="Output directory")
    parser.add_argument("--dataset-name", type=str, default="togethercomputer/RedPajama-Data-1T-Sample", help="Dataset name or path")
    
    args = parser.parse_args()

    # Load YAML configuration if provided
    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            yaml_config = yaml.safe_load(file)
        logger.info(f"Loaded YAML configuration from {args.config}")
    else:
        yaml_config = {}
        logger.warning(f"YAML config file not found at {args.config}, using default values")

    # Initialize config with YAML values or defaults
    config = ImprovementConfig(
        model_name=yaml_config.get("model_name", args.model_name),
        output_dir=yaml_config.get("output_dir", args.output_dir),
        quantization_bits=yaml_config.get("quantization_bits", 4),
        batch_size=yaml_config.get("batch_size", 2),
        mixed_precision=yaml_config.get("mixed_precision", "fp16"),
        max_length=yaml_config.get("max_length", 512),
        num_epochs=yaml_config.get("num_epochs", 1),
        learning_rate=yaml_config.get("learning_rate", 2e-5),
        warmup_steps=yaml_config.get("warmup_steps", 100),
        logging_steps=yaml_config.get("logging_steps", 100),
        save_steps=yaml_config.get("save_steps", 1000),
        eval_steps=yaml_config.get("eval_steps", 1000),
        eval_batch_size=yaml_config.get("eval_batch_size", 2),
        use_gradient_checkpointing=yaml_config.get("use_gradient_checkpointing", True),
        run_quantization=yaml_config.get("run_quantization", BNB_AVAILABLE),
        run_attention_optimization=yaml_config.get("run_attention_optimization", FLASH_AVAILABLE),
        run_pruning=yaml_config.get("run_pruning", True),
        lora_r=yaml_config.get("lora_r", 16),
        lora_alpha=yaml_config.get("lora_alpha", 32),
        lora_dropout=yaml_config.get("lora_dropout", 0.05),
        pruning_sparsity=yaml_config.get("pruning_sparsity", 0.5)
    )

    # Save the configuration
    config_path = os.path.join(config.output_dir, "experiment_config.json")
    config.to_json(config_path)
    
    # Initialize experiment runner
    experiment_runner = ExperimentRunner(config)
    
    # Run experiment based on mode
    if args.mode == "optimize":
        baseline_metrics, improved_metrics = experiment_runner.run_complete_experiment(dataset_name=args.dataset_name)
        logger.info("Optimization experiment completed")
    elif args.mode == "train":
        model_loader = ModelLoader(config)
        tokenizer = model_loader.load_tokenizer()
        model = model_loader.load_model(quantized=config.run_quantization)
        trainer = ModelTrainer(model, tokenizer, config)
        train_dataset = trainer.prepare_dataset(dataset_name=args.dataset_name)
        eval_dataset = trainer.prepare_dataset(dataset_name=args.dataset_name, split="train[:1%]")
        trainer.create_trainer(train_dataset, eval_dataset).train()
        logger.info("Training completed")
    elif args.mode == "evaluate":
        model_loader = ModelLoader(config)
        tokenizer = model_loader.load_tokenizer()
        model = model_loader.load_model(quantized=config.run_quantization)
        analyzer = ModelAnalyzer(model, tokenizer)
        eval_dataset = ModelTrainer(model, tokenizer, config).prepare_dataset(dataset_name=args.dataset_name, split="train[:1%]")
        metrics = {
            "inference_speed": analyzer.benchmark_inference_speed(),
            "memory_usage": analyzer.measure_memory_usage(),
            "perplexity": analyzer.compute_perplexity(eval_dataset)
        }
        experiment_runner.save_results(metrics, {})
        logger.info("Evaluation completed")

if __name__ == "__main__":
    main()