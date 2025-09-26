import torch
import torch.nn as nn
from torchvision import models as modules
import logging

# Get our logger from the main app
logger = logging.getLogger("modules")

class AlexNet(nn.Module):
    """
    The classic AlexNet that started the deep learning revolution.

    This is the model that won ImageNet in 2012 and basically kicked off
    the whole deep learning craze. It's pretty old-school now but still
    works well for basic classification tasks. Uses big kernels and
    aggressive pooling to process images efficiently.

    Args:
        classes: How many different things you want to classify
        channels: Input channels (3 for RGB, 1 for grayscale)
        weights: Start with ImageNet pretrained weights? (Usually yes)
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up AlexNet for {classes} classes with {channels} input channels")

        # Load the base AlexNet model
        self.module = modules.alexnet(
            weights=modules.AlexNet_Weights.DEFAULT if weights else None
        )

        # If we're not using standard RGB images, change the first layer
        if channels != 3:
            self.module.features[0] = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2,
            )

        # Replace the final layer to match our number of classes
        features = self.module.classifier[6].in_features
        self.module.classifier[6] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model on some input images."""
        return self.module(x)


class ResNet50(nn.Module):
    """
    ResNet-50: The workhorse of computer vision.

    This 50-layer network uses residual connections (skip connections) to
    avoid the vanishing gradient problem. It's probably the most popular
    architecture for image classification and works great as a feature
    extractor for other tasks too.

    Args:
        classes: Number of classes to classify
        channels: Input image channels (usually 3)
        weights: Use ImageNet pretrained weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up ResNet-50 for {classes} classes with {channels} input channels")

        # Load ResNet-50 with optional pretrained weights
        self.module = modules.resnet50(
            weights=modules.ResNet50_Weights.DEFAULT if weights else None
        )

        # Modify the first layer if needed
        if channels != 3:
            self.module.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Update the final classification layer
        features = self.module.fc.in_features  # Should be 2048
        self.module.fc = nn.Linear(in_features=features, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ResNet-50 forward pass."""
        return self.module(x)


class ResNet18(nn.Module):
    """
    ResNet-18: The lightweight cousin of ResNet-50.

    Same idea as ResNet-50 but with only 18 layers, so it's faster to train
    and uses less memory. Perfect when you don't have huge datasets or
    powerful hardware. Still uses residual connections for stable training.

    Args:
        classes: Number of output classes
        channels: Input channels (3 for RGB)
        weights: Use pretrained weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up ResNet-18 for {classes} classes with {channels} input channels")

        # Load the lighter ResNet variant
        self.module = modules.resnet18(
            weights=modules.ResNet18_Weights.DEFAULT if weights else None
        )

        # Handle non-RGB inputs
        if channels != 3:
            self.module.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Fix the final layer for our classes
        features = self.module.fc.in_features  # Should be 512
        self.module.fc = nn.Linear(in_features=features, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ResNet-18 forward pass."""
        return self.module(x)


class MobileNetLarge(nn.Module):
    """
    MobileNet-V3 Large: Great for mobile and edge devices.

    Uses depthwise separable convolutions to be super efficient while
    still getting good accuracy. Has squeeze-and-excitation blocks and
    hard-swish activations. Perfect when you need to run on phones or
    embedded devices.

    Args:
        classes: Number of classes to predict
        channels: Input image channels
        weights: Use pretrained weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up MobileNet-V3 Large for {classes} classes with {channels} input channels")

        # Load the mobile-optimized model
        self.module = modules.mobilenet_v3_large(
            weights=modules.MobileNet_V3_Large_Weights.DEFAULT if weights else None
        )

        # Adjust first layer if needed
        if channels != 3:
            self.module.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        # Update the classifier
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run MobileNet-V3 Large forward pass."""
        return self.module(x)


class MobileNetSmall(nn.Module):
    """
    MobileNet-V3 Small: Even more compact for ultra-low power.

    This is the smallest MobileNet variant - designed for when you really
    need to squeeze every bit of efficiency out. Great for real-time
    applications on very limited hardware.

    Args:
        classes: Number of output classes
        channels: Input channels (usually 3)
        weights: Use pretrained weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up MobileNet-V3 Small for {classes} classes with {channels} input channels")

        # Load the most compact variant
        self.module = modules.mobilenet_v3_small(
            weights=modules.MobileNet_V3_Small_Weights.DEFAULT if weights else None
        )

        # Handle custom input channels
        if channels != 3:
            self.module.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        # Fix the final classifier
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run MobileNet-V3 Small forward pass."""
        return self.module(x)


class VGG11(nn.Module):
    """
    VGG-11: Simple and straightforward CNN architecture.

    The simplest VGG model with just 11 layers. Uses small 3x3 convolutions
    throughout and has a very straightforward design. Good for learning
    how CNNs work since it's so simple to understand.

    Args:
        classes: Number of classes to classify
        channels: Input image channels
        weights: Use pretrained ImageNet weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up VGG-11 for {classes} classes with {channels} input channels")

        # Load the simplest VGG variant
        self.module = modules.vgg11(
            weights=modules.VGG11_Weights.DEFAULT if weights else None
        )

        # Modify first layer for different input channels
        if channels != 3:
            self.module.features[0] = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=3,
                padding=1
            )

        # Update the final classifier
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run VGG-11 forward pass."""
        return self.module(x)


class VGG16(nn.Module):
    """
    VGG-16: The most popular VGG variant.

    This 16-layer version is the one most people think of when they say "VGG".
    It's deeper than VGG-11 so it can learn more complex features, but it's
    still simple enough to understand. Uses lots of 3x3 convolutions and
    max pooling.

    Args:
        classes: Number of output classes
        channels: Input channels (3 for RGB)
        weights: Use pretrained weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up VGG-16 for {classes} classes with {channels} input channels")

        # Load the classic VGG-16 model
        self.module = modules.vgg16(
            weights=modules.VGG16_Weights.DEFAULT if weights else None
        )

        # Handle different input channels
        if channels != 3:
            self.module.features[0] = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=3,
                padding=1
            )

        # Fix the final layer
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run VGG-16 forward pass."""
        return self.module(x)


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0: The baseline of the super-efficient EfficientNet family.

    This model figured out the perfect balance between network depth, width,
    and input resolution. It gets amazing accuracy while being really efficient.
    Uses compound scaling to optimize all dimensions at once.

    Args:
        classes: Number of classes to predict
        channels: Input image channels
        weights: Use pretrained weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up EfficientNet-B0 for {classes} classes with {channels} input channels")

        # Load the baseline EfficientNet model
        self.module = modules.efficientnet_b0(
            weights=modules.EfficientNet_B0_Weights.DEFAULT if weights else None
        )

        # Modify stem for custom channels
        if channels != 3:
            self.module.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        # Update classifier
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run EfficientNet-B0 forward pass."""
        return self.module(x)


class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3: Scaled up for higher accuracy.

    This is B0 but scaled up using compound scaling - it's deeper, wider,
    and takes higher resolution inputs. More accurate than B0 but still
    way more efficient than other models with similar performance.

    Args:
        classes: Number of output classes
        channels: Input channels (usually 3)
        weights: Use pretrained weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up EfficientNet-B3 for {classes} classes with {channels} input channels")

        # Load the scaled-up B3 model
        self.module = modules.efficientnet_b3(
            weights=modules.EfficientNet_B3_Weights.DEFAULT if weights else None
        )

        # Handle custom input channels
        if channels != 3:
            self.module.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=40,  # Wider than B0 due to scaling
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        # Update final classifier
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run EfficientNet-B3 forward pass."""
        return self.module(x)


class DenseNet121(nn.Module):
    """
    DenseNet-121: Every layer connects to every other layer.

    This architecture connects each layer to all the layers that come after it.
    Sounds crazy but it actually works really well! It reuses features
    efficiently and needs fewer parameters than you'd expect. The "121"
    refers to the total number of layers.

    Args:
        classes: Number of classes to classify
        channels: Input image channels
        weights: Use pretrained ImageNet weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up DenseNet-121 for {classes} classes with {channels} input channels")

        # Load DenseNet with dense connections
        self.module = modules.densenet121(
            weights=modules.DenseNet121_Weights.DEFAULT if weights else None
        )

        # Handle different input channels
        if channels != 3:
            self.module.features.conv0 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Update the classifier
        features = self.module.classifier.in_features
        self.module.classifier = nn.Linear(in_features=features, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run DenseNet-121 forward pass."""
        return self.module(x)


class DenseNet169(nn.Module):
    """
    DenseNet-169: Deeper version with even more dense connections.

    Same idea as DenseNet-121 but with 169 layers instead. The extra depth
    lets it learn more complex patterns, though it takes longer to train
    and uses more memory. Still way more parameter-efficient than other
    models of similar depth.

    Args:
        classes: Number of output classes
        channels: Input channels (3 for RGB)
        weights: Use pretrained weights?
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Setting up DenseNet-169 for {classes} classes with {channels} input channels")

        # Load the deeper DenseNet variant
        self.module = modules.densenet169(
            weights=modules.DenseNet169_Weights.DEFAULT if weights else None
        )

        # Modify first layer if needed
        if channels != 3:
            self.module.features.conv0 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Update classifier for our classes
        features = self.module.classifier.in_features
        self.module.classifier = nn.Linear(in_features=features, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run DenseNet-169 forward pass."""
        return self.module(x)