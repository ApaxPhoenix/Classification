# Image Classification Training Framework

A PyTorch framework for training image classification models. This repo gives you a single interface to work with eleven different classification architectures, handling the optimization and training specifics for each one.

## Supported Models

- **AlexNet** - The CNN that kicked off the deep learning era
- **ResNet18/50** - Residual networks with skip connections
- **VGG11/16** - Deep networks with small convolutional filters
- **MobileNetV3 Large/Small** - Efficient architectures built for mobile
- **EfficientNet B0/B3** - Compound scaled networks balancing depth, width, and resolution
- **DenseNet121/169** - Dense connectivity for better feature reuse

## What You Get

- Single pipeline for all eleven architectures
- Transfer learning with pre-trained ImageNet weights
- Separate log files for each component (trust me, this helps)
- Configure everything from the command line
- Auto-saves checkpoints and runs validation
- Handles any image size and custom datasets

## Setup

Install the dependencies:

```bash
pip install torch torchvision numpy
```

## How to Use

### Quick Start

```bash
python main.py \
  --module resnet50 \
  --classes 10 \
  --training-path ./data/train \
  --validation-path ./data/val \
  --output ./models/trained_model.pt
```

### Command Line Arguments

#### Model Setup
- `--module` - Which architecture to use (required)
- `--classes` - Number of classes (required)
- `--channels` - Input channels, typically 3 for RGB
- `--weights` - Use pre-trained weights (enabled by default)

#### Data Paths
- `--training-path` - Where your training data lives (required)
- `--validation-path` - Where your validation data lives (required)
- `--testing-path` - Where your test data lives
- `--weights-path` - Path to checkpoint if resuming training

#### Training Parameters
- `--epochs` - How long to train (default: 25)
- `--batch-size` - Samples per batch (default: 64)
- `--learning-rate` - Initial learning rate (default: 0.0001)
- `--dimensions` - Image size as width height (default: 64 64)
- `--workers` - Number of data loading threads (default: 4)
- `--seed` - Random seed for reproducibility

#### Advanced Options
- `--weight-decay` - L2 regularization strength
- `--gamma` - Learning rate decay factor
- `--momentum` - Momentum for SGD

## Logging

Logs get written to four separate files:

- `main.log` - High-level program flow
- `loader.log` - Data loading operations
- `modules.log` - Model-specific operations
- `trainer.log` - Training progress and metrics

## Structure

```
.
├── main.py           # Entry point
├── trainer.py        # Training logic
├── modules.py        # All model implementations
└── logs/            # Where logs end up
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- NumPy

## Notes

Pre-trained weights load automatically when you have them. They're especially useful when you're working with smaller datasets.

The defaults work well enough to get started, but you'll want to tune them based on your specific dataset and hardware.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
