import torch
import torch.nn as nn
from torchvision import models as modules
import logging
import warnings

# Handlers are configured in main file
logger = logging.getLogger("modules")


class FCN_ResNet50(nn.Module):
    """FCN with ResNet50 backbone for semantic segmentation.

    Fully Convolutional Network that replaces final FC layers with convolutions
    to preserve spatial information. Uses skip connections from intermediate
    layers for better boundary delineation and detail recovery.

    Args:
        classes (int): Number of segmentation classes including background
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        weights (bool): Whether to load pretrained COCO weights

    Input:
        Tensor of shape (batch, channels, height, width)

    Output:
        Tensor of shape (batch, classes, height, width) with per-pixel predictions
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing FCN ResNet50: {classes} classes, {channels} channels"
        )

        # Load base model with optional pretrained weights
        self.module = modules.segmentation.fcn_resnet50(
            weights=(
                modules.segmentation.FCN_ResNet50_Weights.DEFAULT if weights else None
            ),
            classes=classes,
        )

        # Modify input layer for non-RGB images (grayscale, multispectral, etc.)
        if channels != 3:
            self.module.backbone.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FCN network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions with shape (batch, classes, height, width)
        """
        # Extract main output from torchvision FCN (ignores auxiliary output)
        output = self.module(x)
        return output["out"]


class FCN_ResNet101(nn.Module):
    """FCN with ResNet101 backbone for semantic segmentation.

    Deeper variant of FCN_ResNet50 with better feature extraction capability.
    Higher computational cost but improved accuracy on complex scenes with
    fine-grained details and multiple object scales.

    Args:
        classes (int): Number of segmentation classes including background
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        weights (bool): Whether to load pretrained COCO weights

    Input:
        Tensor of shape (batch, channels, height, width)

    Output:
        Tensor of shape (batch, classes, height, width) with per-pixel predictions
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing FCN ResNet101: {classes} classes, {channels} channels"
        )

        # Load deeper ResNet101 backbone for enhanced feature extraction
        self.module = modules.segmentation.fcn_resnet101(
            weights=(
                modules.segmentation.FCN_ResNet101_Weights.DEFAULT if weights else None
            ),
            classes=classes,
        )

        # Modify input layer for non-RGB images
        if channels != 3:
            self.module.backbone.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FCN ResNet101 network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions with shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]


class DeepLabV3_ResNet50(nn.Module):
    """DeepLabV3 with ResNet50 backbone for semantic segmentation.

    Advanced segmentation model using Atrous Spatial Pyramid Pooling (ASPP)
    to capture multi-scale context. Excellent for scenes with objects at
    different scales and complex spatial relationships.

    Args:
        classes (int): Number of segmentation classes including background
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        weights (bool): Whether to load pretrained COCO weights

    Input:
        Tensor of shape (batch, channels, height, width)

    Output:
        Tensor of shape (batch, classes, height, width) with per-pixel predictions
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing DeepLabV3 ResNet50: {classes} classes, {channels} channels"
        )

        # Load model with ASPP for multi-scale feature extraction
        self.module = modules.segmentation.deeplabv3_resnet50(
            weights=(
                modules.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
                if weights
                else None
            ),
            classes=classes,
        )

        # Modify input layer for non-RGB images
        if channels != 3:
            self.module.backbone.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepLabV3 network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions with shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]


class DeepLabV3_ResNet101(nn.Module):
    """DeepLabV3 with ResNet101 backbone for semantic segmentation.

    Premium segmentation model combining DeepLabV3's ASPP architecture with
    deeper ResNet101 backbone. Best accuracy for complex segmentation tasks
    where computational cost is less critical than precision.

    Args:
        classes (int): Number of segmentation classes including background
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        weights (bool): Whether to load pretrained COCO weights

    Input:
        Tensor of shape (batch, channels, height, width)

    Output:
        Tensor of shape (batch, classes, height, width) with per-pixel predictions
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing DeepLabV3 ResNet101: {classes} classes, {channels} channels"
        )

        # Load deepest variant for maximum feature extraction capability
        self.module = modules.segmentation.deeplabv3_resnet101(
            weights=(
                modules.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
                if weights
                else None
            ),
            classes=classes,
        )

        # Modify input layer for non-RGB images
        if channels != 3:
            self.module.backbone.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepLabV3 ResNet101 network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions with shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]


class DeepLabV3_MobileNetV3Large(nn.Module):
    """DeepLabV3 with MobileNetV3-Large backbone for semantic segmentation.

    Mobile-optimized segmentation model using depthwise separable convolutions.
    Excellent balance between accuracy and efficiency for deployment on
    resource-constrained devices while maintaining ASPP benefits.

    Args:
        classes (int): Number of segmentation classes including background
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        weights (bool): Whether to load pretrained COCO weights

    Input:
        Tensor of shape (batch, channels, height, width)

    Output:
        Tensor of shape (batch, classes, height, width) with per-pixel predictions
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing DeepLabV3 MobileNetV3-Large: {classes} classes, {channels} channels"
        )

        # Load mobile-optimized model with depthwise separable convolutions
        self.module = modules.segmentation.deeplabv3_mobilenet_v3_large(
            weights=(
                modules.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                if weights
                else None
            ),
            classes=classes,
        )

        # Modify input layer for non-RGB images
        # MobileNet architecture has different first layer structure
        if channels != 3:
            self.module.backbone.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepLabV3 MobileNetV3 network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions with shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]


class LRASPP_MobileNetV3Large(nn.Module):
    """LRASPP with MobileNetV3-Large backbone for semantic segmentation.

    Lightweight Real-time Semantic Segmentation model optimized for mobile
    deployment. Simplified ASPP with only global pooling and 1x1 convolutions
    for fastest inference while maintaining reasonable accuracy.

    Args:
        classes (int): Number of segmentation classes including background
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        weights (bool): Whether to load pretrained COCO weights

    Input:
        Tensor of shape (batch, channels, height, width)

    Output:
        Tensor of shape (batch, classes, height, width) with per-pixel predictions
    """

    def __init__(self, classes: int, channels: int = 3, weights: bool = True) -> None:
        super().__init__()

        logger.info(
            f"Initializing LRASPP MobileNetV3-Large: {classes} classes, {channels} channels"
        )

        # Load fastest segmentation model for real-time applications
        self.module = modules.segmentation.lraspp_mobilenet_v3_large(
            weights=(
                modules.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
                if weights
                else None
            ),
            classes=classes,
        )

        # Modify input layer for non-RGB images
        if channels != 3:
            self.module.backbone.features[0][0] = nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LRASPP network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions with shape (batch, classes, height, width)
        """
        output = self.module(x)
        return output["out"]

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation.

    Classic encoder-decoder with symmetric skip connections for precise
    localization. Excellent for biomedical imaging and tasks requiring
    fine boundary delineation. Works well with limited training data.

    Args:
        classes (int): Number of segmentation classes including background
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        base (int): Base number of feature channels (doubles each level)
        weights (bool): Ignored - custom implementation without pretrained weights

    Input:
        Tensor of shape (batch, channels, height, width)
        Note: Height and width should be divisible by 16

    Output:
        Tensor of shape (batch, classes, height, width) with per-pixel predictions
    """

    def __init__(self, classes: int, channels: int = 3, base: int = 64, weights: bool = None) -> None:
        super().__init__()

        if weights is not None:
            warnings.warn(
                "This module is from barebones and does not include pretrained weights. "
                "The 'weights' parameter will be ignored."
            )
            logger.warning(
                "This module is from barebones and does not include pretrained weights. "
                "The 'weights' parameter will be ignored."
            )

        logger.info(
            f"Initializing U-Net: {classes} classes, {channels} channels, {base} base filters"
        )

        # Encoder blocks - progressive downsampling with feature extraction
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )

        # Bottleneck - deepest feature representation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 16, 3, padding=1),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 16, base * 16, 3, padding=1),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),
        )

        # Decoder upsampling layers
        self.upconv4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base * 16, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )

        self.upconv3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Final classification layer
        self.final = nn.Conv2d(base, classes, 1)

        # Pooling layer for encoder downsampling
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net with skip connections.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions with shape (batch, classes, height, width)
        """
        # Encoder path with skip connection storage
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder path with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


class SegNet(nn.Module):
    """SegNet architecture for semantic segmentation.

    Memory-efficient segmentation using max pooling indices for upsampling.
    No skip connections - relies on stored pooling indices for spatial
    reconstruction. Excellent for precise boundary delineation.

    Args:
        classes (int): Number of segmentation classes including background
        channels (int): Input image channels (3 for RGB, 1 for grayscale)
        weights: Ignored - custom implementation without pretrained weights

    Input:
        Tensor of shape (batch, channels, height, width)
        Note: Height and width should be divisible by 8

    Output:
        Tensor of shape (batch, classes, height, width) with per-pixel predictions
    """

    def __init__(self, classes: int, channels: int = 3, weights = None) -> None:
        super().__init__()

        if weights is not None:
            warnings.warn(
                "This module is from barebones and does not include pretrained weights. "
                "The 'weights' parameter will be ignored."
            )
            logger.warning(
                "This module is from barebones and does not include pretrained weights. "
                "The 'weights' parameter will be ignored."
            )

        logger.info(
            f"Initializing SegNet: {classes} classes, {channels} channels"
        )

        # Encoder blocks - VGG-style feature extraction
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks - mirror encoder for upsampling
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classes, 3, padding=1),
        )

        # Pooling with index storage for precise upsampling
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SegNet using pooling indices.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Segmentation predictions with shape (batch, classes, height, width)
        """
        # Encoder path with index storage
        x1 = self.enc_conv1(x)
        x1_pooled, idx1 = self.pool(x1)

        x2 = self.enc_conv2(x1_pooled)
        x2_pooled, idx2 = self.pool(x2)

        x3 = self.enc_conv3(x2_pooled)
        x3_pooled, idx3 = self.pool(x3)

        # Decoder path using stored indices
        x3_up = self.unpool(x3_pooled, idx3)
        x3_dec = self.dec_conv3(x3_up)

        x2_up = self.unpool(x3_dec, idx2)
        x2_dec = self.dec_conv2(x2_up)

        x1_up = self.unpool(x2_dec, idx1)
        output = self.dec_conv1(x1_up)

        return output