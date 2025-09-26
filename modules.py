import torch
import torch.nn as nn
from torchvision import models as modules
import logging

# Use logger configured in main application
logger = logging.getLogger("modules")

class AlexNet(nn.Module):
    """
    AlexNet architecture for image classification.

    Historic CNN architecture that demonstrated the effectiveness of deep learning
    on ImageNet in 2012. Features large convolutional kernels and aggressive
    max pooling for computational efficiency. Suitable for basic classification
    tasks with moderate accuracy requirements.

    Args:
        classes: Number of output classification classes
        channels: Input image channels (3 for RGB, 1 for grayscale)
        weights: Whether to initialize with ImageNet pre-trained weights
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing AlexNet: {classes} classes, {channels} input channels")

        # Load base AlexNet architecture
        self.module = modules.alexnet(
            weights=modules.AlexNet_Weights.DEFAULT if weights else None
        )

        # Modify input layer for non-RGB images
        if channels != 3:
            self.module.features[0] = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2,
            )

        # Replace final classification layer
        features = self.module.classifier[6].in_features
        self.module.classifier[6] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through AlexNet architecture."""
        return self.module(x)


class ResNet50(nn.Module):
    """
    ResNet-50 architecture with residual connections.

    50-layer convolutional network utilizing skip connections to address
    vanishing gradient problems in deep networks. Widely adopted architecture
    that serves as strong baseline for image classification and feature
    extraction tasks.

    Args:
        classes: Number of target classification classes
        channels: Input image channels (typically 3 for RGB)
        weights: Whether to use ImageNet pre-trained initialization
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing ResNet-50: {classes} classes, {channels} input channels")

        # Load ResNet-50 with optional pre-trained weights
        self.module = modules.resnet50(
            weights=modules.ResNet50_Weights.DEFAULT if weights else None
        )

        # Adjust input layer for non-standard channel counts
        if channels != 3:
            self.module.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Configure final classification layer
        features = self.module.fc.in_features  # Standard ResNet-50 has 2048 features
        self.module.fc = nn.Linear(in_features=features, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute ResNet-50 forward propagation."""
        return self.module(x)


class ResNet18(nn.Module):
    """
    ResNet-18 lightweight architecture.

    Compact 18-layer residual network offering faster training and lower
    memory requirements compared to deeper variants. Maintains residual
    connection benefits while providing practical efficiency for resource-
    constrained environments.

    Args:
        classes: Number of output classification classes
        channels: Input image channels (3 for RGB)
        weights: Whether to use pre-trained weight initialization
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing ResNet-18: {classes} classes, {channels} input channels")

        # Load lightweight ResNet variant
        self.module = modules.resnet18(
            weights=modules.ResNet18_Weights.DEFAULT if weights else None
        )

        # Handle non-RGB input configurations
        if channels != 3:
            self.module.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Configure classification head
        features = self.module.fc.in_features  # ResNet-18 has 512 features
        self.module.fc = nn.Linear(in_features=features, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute ResNet-18 forward pass."""
        return self.module(x)


class MobileNetLarge(nn.Module):
    """
    MobileNet-V3 Large for efficient mobile deployment.

    Optimized architecture using depthwise separable convolutions, squeeze-
    and-excitation blocks, and hard-swish activations. Designed for mobile
    and edge computing with balanced accuracy-efficiency trade-offs.

    Args:
        classes: Number of classification classes
        channels: Input image channels
        weights: Whether to use pre-trained weights
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing MobileNet-V3 Large: {classes} classes, {channels} input channels")

        # Load mobile-optimized architecture
        self.module = modules.mobilenet_v3_large(
            weights=modules.MobileNet_V3_Large_Weights.DEFAULT if weights else None
        )

        # Modify initial convolution for custom channel counts
        if channels != 3:
            self.module.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        # Update classification layer
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute MobileNet-V3 Large forward propagation."""
        return self.module(x)


class MobileNetSmall(nn.Module):
    """
    MobileNet-V3 Small for ultra-efficient deployment.

    Most compact MobileNet variant optimized for extremely resource-constrained
    environments. Prioritizes computational efficiency over accuracy for
    real-time applications on limited hardware.

    Args:
        classes: Number of output classification classes
        channels: Input image channels (typically 3)
        weights: Whether to use pre-trained initialization
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing MobileNet-V3 Small: {classes} classes, {channels} input channels")

        # Load most compact mobile variant
        self.module = modules.mobilenet_v3_small(
            weights=modules.MobileNet_V3_Small_Weights.DEFAULT if weights else None
        )

        # Configure input layer for custom channels
        if channels != 3:
            self.module.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=16,
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
        """Execute MobileNet-V3 Small forward pass."""
        return self.module(x)


class VGG11(nn.Module):
    """
    VGG-11 basic convolutional architecture.

    Straightforward CNN design using uniform 3x3 convolutions and max pooling.
    Simple architecture that serves as educational baseline for understanding
    convolutional neural network fundamentals.

    Args:
        classes: Number of classification classes
        channels: Input image channels
        weights: Whether to use ImageNet pre-trained weights
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing VGG-11: {classes} classes, {channels} input channels")

        # Load basic VGG architecture
        self.module = modules.vgg11(
            weights=modules.VGG11_Weights.DEFAULT if weights else None
        )

        # Modify input layer for different channel configurations
        if channels != 3:
            self.module.features[0] = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=3,
                padding=1
            )

        # Configure classification layer
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute VGG-11 forward propagation."""
        return self.module(x)


class VGG16(nn.Module):
    """
    VGG-16 deep convolutional architecture.

    Standard 16-layer VGG implementation using small receptive fields and
    deep structure. More complex feature learning capability compared to
    VGG-11 while maintaining architectural simplicity.

    Args:
        classes: Number of output classification classes
        channels: Input image channels (3 for RGB)
        weights: Whether to use pre-trained initialization
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing VGG-16: {classes} classes, {channels} input channels")

        # Load standard VGG-16 architecture
        self.module = modules.vgg16(
            weights=modules.VGG16_Weights.DEFAULT if weights else None
        )

        # Handle non-standard input channels
        if channels != 3:
            self.module.features[0] = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=3,
                padding=1
            )

        # Configure final classification layer
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute VGG-16 forward pass."""
        return self.module(x)


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 compound scaling baseline.

    Optimally balanced architecture using compound scaling methodology
    to jointly optimize network depth, width, and resolution. Provides
    strong accuracy-efficiency trade-offs through systematic scaling.

    Args:
        classes: Number of classification classes to predict
        channels: Input image channels
        weights: Whether to use pre-trained initialization
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing EfficientNet-B0: {classes} classes, {channels} input channels")

        # Load baseline EfficientNet architecture
        self.module = modules.efficientnet_b0(
            weights=modules.EfficientNet_B0_Weights.DEFAULT if weights else None
        )

        # Modify stem convolution for custom channels
        if channels != 3:
            self.module.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        # Configure classification head
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute EfficientNet-B0 forward propagation."""
        return self.module(x)


class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 scaled architecture for enhanced accuracy.

    Compound-scaled version of EfficientNet-B0 with increased depth, width,
    and resolution parameters. Provides improved classification performance
    while maintaining computational efficiency relative to conventional scaling.

    Args:
        classes: Number of output classification classes
        channels: Input image channels (typically 3)
        weights: Whether to use pre-trained weights
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing EfficientNet-B3: {classes} classes, {channels} input channels")

        # Load scaled EfficientNet variant
        self.module = modules.efficientnet_b3(
            weights=modules.EfficientNet_B3_Weights.DEFAULT if weights else None
        )

        # Handle custom input channel configurations
        if channels != 3:
            self.module.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=40,  # Wider stem due to compound scaling
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        # Update classification layer
        features = self.module.classifier[-1].in_features
        self.module.classifier[-1] = nn.Linear(
            in_features=features, out_features=classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute EfficientNet-B3 forward pass."""
        return self.module(x)


class DenseNet121(nn.Module):
    """
    DenseNet-121 with dense connectivity patterns.

    Architecture where each layer receives feature maps from all preceding
    layers, promoting feature reuse and gradient flow. The 121-layer
    configuration provides strong performance with parameter efficiency
    through dense connections.

    Args:
        classes: Number of classification classes
        channels: Input image channels
        weights: Whether to use ImageNet pre-trained weights
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing DenseNet-121: {classes} classes, {channels} input channels")

        # Load DenseNet with dense connectivity
        self.module = modules.densenet121(
            weights=modules.DenseNet121_Weights.DEFAULT if weights else None
        )

        # Configure input layer for different channel counts
        if channels != 3:
            self.module.features.conv0 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Update classification layer
        features = self.module.classifier.in_features
        self.module.classifier = nn.Linear(in_features=features, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute DenseNet-121 forward propagation."""
        return self.module(x)


class DenseNet169(nn.Module):
    """
    DenseNet-169 deep dense architecture.

    Extended dense connectivity network with 169 layers providing enhanced
    feature learning capability. Maintains parameter efficiency through
    feature reuse while offering increased model capacity compared to
    DenseNet-121.

    Args:
        classes: Number of output classification classes
        channels: Input image channels (3 for RGB)
        weights: Whether to use pre-trained initialization
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(f"Initializing DenseNet-169: {classes} classes, {channels} input channels")

        # Load deeper DenseNet variant
        self.module = modules.densenet169(
            weights=modules.DenseNet169_Weights.DEFAULT if weights else None
        )

        # Modify input layer for custom channel configurations
        if channels != 3:
            self.module.features.conv0 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Configure final classifier
        features = self.module.classifier.in_features
        self.module.classifier = nn.Linear(in_features=features, out_features=classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute DenseNet-169 forward pass."""
        return self.module(x)