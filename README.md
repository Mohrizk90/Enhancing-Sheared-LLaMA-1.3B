# Improved Sheared LLaMA: Efficient Language Model Training

This repository contains an improved implementation of the Sheared LLaMA architecture, focusing on efficient training and inference of large language models. The project implements various optimizations and improvements over the original LLaMA model, making it more efficient while maintaining performance.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Performance](#performance)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements an improved version of the Sheared LLaMA architecture, focusing on:
- Efficient model training through various optimization techniques
- Improved inference performance
- Memory-efficient model architecture
- Enhanced training stability
- Better handling of long sequences

## Key Features

### 1. Model Optimizations
- **4-bit Quantization**: Reduces memory footprint while maintaining model quality
- **Flash Attention**: Implements efficient attention computation
- **LoRA-based Pruning**: Dynamic model pruning during training
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Fully Sharded Data Parallel (FSDP)**: Efficient distributed training

### 2. Architecture Improvements
- Optimized attention mechanism
- Enhanced positional embeddings
- Improved layer normalization
- Efficient tokenization with SentencePiece
- Better handling of long sequences

### 3. Training Enhancements
- Cosine learning rate scheduling
- Gradient clipping and scaling
- Mixed precision training
- Efficient data loading and preprocessing
- Comprehensive evaluation metrics

## Model Architecture

### Base Model Specifications
- **Model Size**: 1.3B parameters
- **Architecture**: Transformer-based (LLaMA variant)
- **Layers**: 24 transformer layers
- **Attention Heads**: 16 heads
- **Hidden Dimension**: 2048 (d_model)
- **Intermediate Size**: 5504
- **Vocabulary Size**: 32,000 tokens
- **Maximum Sequence Length**: 2048 tokens

### Key Architectural Components

1. **Transformer Layers**
   - Self-attention with flash attention optimization
   - Feed-forward networks with GELU activation
   - Layer normalization with improved stability
   - Residual connections

2. **Attention Mechanism**
   - Multi-head attention (16 heads)
   - Rotary positional embeddings
   - Attention dropout for regularization
   - Efficient attention computation

3. **Embeddings**
   - Token embeddings (32k vocabulary)
   - Rotary positional embeddings
   - Layer normalization on embeddings

## Training Details

### Training Configuration
- **Optimizer**: AdamW
  - Learning rate: 1e-4
  - Weight decay: 0.01
  - Beta parameters: (0.9, 0.999)
- **Learning Rate Schedule**: Cosine with warmup
  - Warmup steps: 2000
  - Maximum learning rate: 1e-4
  - Minimum learning rate: 1e-5
- **Training Steps**: 100,000
- **Batch Size**: 128 (effective)
- **Gradient Accumulation**: 4 steps
- **Mixed Precision**: bfloat16

### Training Optimizations
1. **Memory Efficiency**
   - 4-bit quantization for weights
   - Gradient checkpointing
   - Efficient attention computation
   - Optimized data loading

2. **Training Stability**
   - Gradient clipping (1.0)
   - Layer-wise learning rate decay
   - Attention dropout (0.1)
   - Hidden dropout (0.1)

3. **Distributed Training**
   - Fully Sharded Data Parallel (FSDP)
   - Efficient communication
   - Optimized memory usage
   - Scalable to multiple GPUs

## Performance

The improved model shows significant enhancements in:
- Training efficiency (reduced memory usage and faster training)
- Inference speed
- Model quality on various benchmarks
- Long sequence handling
- Memory efficiency

For detailed performance metrics and comparisons, refer to the `plot_performance_comparison.py` script.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU
- Sufficient disk space for model weights

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd NLP1

# Install dependencies
pip install -r requirements.txt
```

### Model Weights
The model weights are available at [link-to-weights]. Download and place them in the appropriate directory.

## Usage

### Training
```python
from sheared_llama_improved import ImprovedShearedLLaMA

# Initialize model
model = ImprovedShearedLLaMA.from_pretrained("path/to/weights")

# Training configuration
trainer = ModelTrainer(
    model=model,
    train_config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "gradient_accumulation_steps": 4,
        "max_steps": 100000
    }
)

# Start training
trainer.train()
```

### Inference
```python
# Load model
model = ImprovedShearedLLaMA.from_pretrained("path/to/weights")
model.eval()

# Generate text
output = model.generate(
    input_text="Your input text here",
    max_length=100,
    temperature=0.7
)
```

## Project Structure
```
NLP1/
├── configs/                 # Configuration files
│   └── llama2/             # Model configurations
├── improved_sheared_llama/  # Core model implementation
├── Paper/                  # Research paper and documentation
├── plot_performance_comparison.py  # Performance visualization
├── sheared_llama_improved.py      # Main model implementation
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

For more detailed information about the model architecture and training process, please refer to the documentation in the `Paper/` directory. 