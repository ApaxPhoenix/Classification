# Image Classification Training Framework

A PyTorch framework for training image classification models. This repo gives you a single interface to work with eleven different classification architectures, handling the optimization and training specifics for each one.

## Supported Models

- **AlexNet** - Classic CNN architecture that started the deep learning revolution
- **ResNet18/50** - Residual networks with skip connections
- **VGG11/16** - Deep networks with small convolutional filters
- **MobileNetV3 Large/Small** - Efficient mobile-optimized architectures
- **EfficientNet B0/B3** - Compound scaled networks balancing depth, width, and resolution
- **DenseNet121/169** - Dense connectivity patterns for feature reuse

## What's Included

- One training pipeline that works across all architectures
- Pre-trained ImageNet weights support for transfer learning
- Logging split across different components (makes debugging way easier)
- Command-line config for everything
- Automatic checkpointing and evaluation
- Works with custom datasets at any resolution
- Built-in validation and testing

## Setup

You'll need PyTorch and the usual suspects:

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

### All the Options

#### Model Setup
- `--module` - Pick your architecture (required)
- `--classes` - How many classes you're classifying (required)
- `--channels` - Image channels, usually 3 for RGB
- `--weights` - Load pre-trained weights (on by default)

#### Where Your Data Lives
- `--training-path` - Training data folder (required)
- `--validation-path` - Validation data folder (required)
- `--testing-path` - Test data if you have it
- `--weights-path` - Existing checkpoint to continue from

#### Training Settings
- `--epochs` - Training epochs (default: 25)
- `--batch-size` - Batch size (default: 64)
- `--learning-rate` - Starting LR (default: 0.0001)
- `--dimensions` - Image size as width height (default: 64 64)
- `--workers` - Data loading threads (default: 4)
- `--seed` - Set this for reproducible runs

#### If You Want to Get Fancy
- `--weight-decay` - L2 regularization
- `--gamma` - LR decay rate
- `--momentum` - For SGD

### Real Example

```bash
python main.py \
  --module efficientnet-b0 \
  --classes 100 \
  --training-path ./datasets/train \
  --validation-path ./datasets/val \
  --testing-path ./datasets/test \
  --dimensions 224 224 \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 0.0005 \
  --momentum 0.9 \
  --workers 8 \
  --seed 42 \
  --output ./model.pt
```

## What Happens During Training

Pretty straightforward flow:

1. Model gets initialized with your settings
2. Data loaders spin up for training and validation
3. Training loop runs with automatic gradient updates
4. Validation happens after each epoch
5. Optional test evaluation at the end
6. Best weights get saved based on validation performance

## Logging

Logs are split into four files so you're not hunting through one massive log:

- `main.log` - Overall flow and status
- `loader.log` - Data loading stuff
- `modules.log` - Model operations
- `trainer.log` - Training metrics and performance

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

## Things Worth Knowing

Pre-trained weights load automatically when available. This usually helps a lot, especially if you don't have tons of training data.

Each model comes with sensible defaults, but you'll probably want to tweak things based on your dataset and how much compute you have available.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for detai
