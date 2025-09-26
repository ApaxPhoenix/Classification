# Image Classification Training Framework

A PyTorch framework for training image classification models with support for popular CNN architectures and automatic optimization configuration.

## Features

- **11 classification models** including AlexNet, ResNet variants, VGG, MobileNet, EfficientNet, and DenseNet
- **Automatic optimization** with model-specific learning rates and schedulers based on research best practices
- **Progress tracking** with real-time loss monitoring and training metrics
- **Multi-GPU support** for faster training on multi-card systems
- **Detailed logging** with separate files for different components
- **Overfitting detection** with automatic checkpoint creation
- **Deterministic training** with configurable random seeds for reproducibility

## Available Models

| Model | Parameters | Speed | Accuracy | Memory Usage | Best For |
|-------|------------|-------|----------|--------------|----------|
| `alexnet` | 61M | Fast | Medium | Low | Basic classification, learning |
| `resnet18` | 11M | Fast | Medium | Low | Limited resources, quick experiments |
| `resnet50` | 25M | Medium | High | Medium | General-purpose classification |
| `mobilenetv3large` | 5.4M | Very Fast | Medium | Very Low | Mobile deployment |
| `mobilenetv3small` | 2.9M | Very Fast | Medium | Very Low | Edge computing |
| `vgg11` | 132M | Medium | Medium | High | Educational purposes |
| `vgg16` | 138M | Slow | Medium | High | Feature extraction |
| `efficientnet-b0` | 5.3M | Fast | High | Low | Balanced efficiency |
| `efficientnet-b3` | 12M | Medium | Very High | Medium | High accuracy needs |
| `densenet121` | 8M | Medium | High | Medium | Parameter efficiency |
| `densenet169` | 14M | Slow | High | High | Maximum accuracy |

## Dataset Structure

Organize your images in directories where each subdirectory represents a class:

```
dataset/
├── train/
│   ├── cats/
│   │   ├── cat001.jpg
│   │   └── ...
│   ├── dogs/
│   │   ├── dog001.jpg
│   │   └── ...
│   └── cars/
│       └── ...
├── val/
│   ├── cats/
│   ├── dogs/
│   └── cars/
└── test/
    ├── cats/
    ├── dogs/
    └── cars/
```

## Installation

```bash
pip install torch torchvision numpy pillow pathlib asyncio
```

## Basic Usage

```bash
python main.py --module resnet50 --classes 10 --training-path ./dataset/train --validation-path ./dataset/val --testing-path ./dataset/test
```

## Command Line Options

### Required Parameters
- `-m, --module` - Model architecture to use
- `-c, --classes` - Number of classes in your dataset
- `-tp, --training-path` - Training dataset directory
- `-vp, --validation-path` - Validation dataset directory

### Optional Parameters
- `-tep, --testing-path` - Test dataset directory
- `-w, --weights` - Use pre-trained weights (default: True)
- `-ch, --channels` - Input channels: 3 for RGB, 1 for grayscale (default: 3)
- `-wp, --weights-path` - Path to existing model checkpoint
- `-d, --dimensions` - Image size as width height (default: 64 64)
- `-e, --epochs` - Number of training epochs (default: 25)
- `-b, --batch-size` - Training batch size (default: 64)
- `-lr, --learning-rate` - Override default learning rate (default: 0.0001)
- `-wk, --workers` - Data loading workers (default: 4)
- `-s, --seed` - Random seed for reproducible results
- `-wd, --weight-decay` - L2 regularization factor
- `-g, --gamma` - Learning rate scheduler decay
- `-mm, --momentum` - SGD momentum parameter
- `-o, --output` - Output path for trained model

## Model-Specific Optimizations

The framework automatically configures optimal training settings for each architecture:

**ResNet models** use SGD with cosine annealing or milestone scheduling
**MobileNet variants** use RMSprop with exponential decay  
**EfficientNet models** use Adam with cosine annealing
**VGG and DenseNet** use SGD with milestone scheduling
**AlexNet** uses Adam with step scheduling

Learning rates, weight decay, and scheduler parameters are set based on published research for each model type.

## Output Files

Training generates several log files:
- `main.log` - Overall application status and errors
- `loader.log` - Dataset loading operations and issues
- `modules.log` - Model initialization and configuration
- `trainer.log` - Training progress, loss values, and metrics

Model weights are saved to the path specified with `--output` (defaults to current directory).

## Performance Tips

**Memory issues?**
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Use smaller images: `--dimensions 32 32`
- Switch to lightweight models: `mobilenetv3small` or `resnet18`

**Training too slow?**
- Increase workers: `--workers 8`
- Use faster models: `mobilenetv3large` or `efficientnet-b0`
- Try smaller image sizes or fewer epochs

**Poor accuracy?**
- Use larger models: `resnet50`, `efficientnet-b3`, or `densenet121`
- Increase image dimensions: `--dimensions 224 224`
- Train for more epochs: `--epochs 100`
- Make sure you have enough training data per class

## Troubleshooting

**"Directory not found" errors**
- Verify your dataset paths exist and contain class subdirectories
- Check that image files are in supported formats (jpg, png, jpeg, tiff)
- Make sure each class directory contains at least some images

**"Model type not supported" errors**
- Double-check the model name spelling against the available models list
- Use `--module` followed by one of the exact names from the table above

**Out of memory errors**
- Start with `--batch-size 8` and work your way up
- Try smaller image dimensions like `--dimensions 64 64`
- Use a lighter model like `mobilenetv3small` or `resnet18`

**Training loss not decreasing**
- Check that your dataset is properly organized with clear class separation
- Try a different learning rate with `--learning-rate`
- Verify you have enough training data (at least 100 images per class recommended)
- Make sure images aren't corrupted (check the loader.log file)

**Validation loss increasing while training loss decreases**
- This indicates overfitting - the framework will automatically create checkpoints
- Try adding more training data or using data augmentation
- Consider using a smaller model or more regularization

## Model Recommendations

- **First experiments**: `resnet18` or `mobilenetv3large` - fast and reliable
- **Production systems**: `resnet50` or `efficientnet-b0` - good balance of speed and accuracy  
- **Mobile/edge deployment**: `mobilenetv3small` or `mobilenetv3large`
- **Maximum accuracy**: `efficientnet-b3` or `densenet169`
- **Limited memory**: `resnet18` or `mobilenetv3small`
- **Educational use**: `alexnet` or `vgg11` - simple architectures to understand

The framework handles the complexity of training configuration automatically, so you can focus on preparing good data and choosing the right model for your needs.
