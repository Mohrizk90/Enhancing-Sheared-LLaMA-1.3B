import matplotlib.pyplot as plt
import os

def plot_performance_comparison(baseline_metrics, improved_metrics, method, output_path):
    labels = ["Tokens/Second", "Memory (GB)", "Perplexity"]
    baseline_values = [
        baseline_metrics["inference_speed"]["tokens_per_second"],
        baseline_metrics["memory_usage"]["model_memory_gb"],
        baseline_metrics["perplexity"]
    ]
    improved_values = [
        improved_metrics["inference_speed"]["tokens_per_second"],
        improved_metrics["memory_usage"]["model_memory_gb"],
        improved_metrics["perplexity"]
    ]
    
    x = range(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], baseline_values, 0.4, label="Baseline", color="#36A2EB")
    plt.bar([i + 0.2 for i in x], improved_values, 0.4, label=f"Improved ({method.replace('_', ' ').title()})", color="#FF6384")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title(f"Baseline vs {method.replace('_', ' ').title()} Performance (RedPajama)")
    plt.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

# Simulated metrics
baseline_metrics = {
    "inference_speed": {"tokens_per_second": 500.0},
    "memory_usage": {"model_memory_gb": 2.8},
    "perplexity": 15.2
}
improved_metrics = {
    "quantization": {
        "inference_speed": {"tokens_per_second": 550.0},
        "memory_usage": {"model_memory_gb": 1.4},
        "perplexity": 16.0
    },
    "attention_optimization": {
        "inference_speed": {"tokens_per_second": 700.0},
        "memory_usage": {"model_memory_gb": 2.7},
        "perplexity": 15.2
    },
    "lora_pruning": {
        "inference_speed": {"tokens_per_second": 600.0},
        "memory_usage": {"model_memory_gb": 2.0},
        "perplexity": 15.8
    }
}

# Generate plots for each optimization method
output_dir = "./improved_sheared_llama"
for method, metrics in improved_metrics.items():
    plot_performance_comparison(
        baseline_metrics,
        metrics,
        method,
        os.path.join(output_dir, f"performance_comparison_{method}.png")
    )