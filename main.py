import torch.nn as nn
import argparse
import asyncio
import logging.config
import warnings
from pathlib import Path
from typing import Dict, Type
from trainer import Trainer
import modules

# Global logging configuration for all application modules
configuration = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        # Main application handler
        "main": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "main.log",
            "mode": "w",
        },
        # Data loading handler
        "loader": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "loader.log",
            "mode": "w",
        },
        # Module operations handler
        "modules": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "modules.log",
            "mode": "w",
        },
        # Training process handler
        "trainer": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "trainer.log",
            "mode": "w",
        },
    },
    "loggers": {
        # Main application logger
        "main": {"handlers": ["main"], "level": "INFO", "propagate": False},
        # Data loader logger
        "loader": {"handlers": ["loader"], "level": "INFO", "propagate": False},
        # Modules logger
        "modules": {"handlers": ["modules"], "level": "INFO", "propagate": False},
        # Trainer logger
        "trainer": {"handlers": ["trainer"], "level": "INFO", "propagate": False},
    },
}

# Registry of available semantic segmentation architectures
modules: Dict[str, Type[nn.Module]] = {
    "unet": modules.UNet,
    "fcn_resnet50": modules.FCN_ResNet50,
    "fcn_resnet101": modules.FCN_ResNet101,
    "deeplabv3_resnet50": modules.DeepLabV3_ResNet50,
    "deeplabv3_resnet101": modules.DeepLabV3_ResNet101,
    "deeplabv3_mobilenet": modules.DeepLabV3_MobileNetV3Large,
    "lraspp_mobilenet": modules.LRASPP_MobileNetV3Large,
    "segnet": modules.SegNet,
}

if __name__ == "__main__":
    # Initialize logging system for all components
    logging.config.dictConfig(configuration)
    logger = logging.getLogger("main")
    logger.info("Starting semantic segmentation training application...")

    # Command-line interface setup
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train different types of segmentation modules using PyTorch framework..."
    )

    # Model architecture selection
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        required=True,
        metavar="...",
        help=f"Module architecture to use. Available modules: {', '.join(modules.keys())}...",
    )

    # Pre-trained weights configuration
    parser.add_argument(
        "-w",
        "--weights",
        type=bool,
        default=True,
        metavar="...",
        help="Whether to load pre-trained weights (Default: True)...",
    )

    # Class configuration
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=True,
        metavar="...",
        help="Number of segmentation classes to predict...",
    )

    # Input channels configuration
    parser.add_argument(
        "-ch",
        "--channels",
        type=int,
        default=3,
        metavar="...",
        help="Number of input channels (default: 3 for RGB images)...",
    )

    # Dataset path configurations
    parser.add_argument(
        "-tp",
        "--training-path",
        type=Path,
        required=True,
        metavar="...",
        help="Path to training dataset directory...",
    )

    parser.add_argument(
        "-vp",
        "--validation-path",
        type=Path,
        required=True,
        metavar="...",
        help="Path to validation dataset directory...",
    )

    parser.add_argument(
        "-tep",
        "--testing-path",
        type=Path,
        required=True,
        metavar="...",
        help="Path to test dataset directory...",
    )

    # Model weights configuration
    parser.add_argument(
        "-wp",
        "--weights-path",
        type=Path,
        default=None,
        metavar="...",
        help="Path to existing model weights file (optional)...",
    )

    # Image processing parameters
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar="...",
        help="Input image dimensions as height width...",
    )

    # Training hyperparameters
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=50,
        metavar="...",
        help="Number of training epochs...",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        metavar="...",
        help="Batch size for training...",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="...",
        help="Learning rate for optimization...",
    )

    parser.add_argument(
        "-wk",
        "--workers",
        type=int,
        default=4,
        metavar="...",
        help="Number of data loading workers...",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        metavar="...",
        help="Random seed for reproducibility...",
    )

    # Advanced optimization parameters
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=0.0005,
        metavar="...",
        help="Weight decay for the optimizer...",
    )

    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.1,
        metavar="...",
        help="Gamma for the learning rate scheduler...",
    )

    # Distributed training configuration
    parser.add_argument(
        "-p",
        "--parallelism",
        type=bool,
        default=False,
        metavar="...",
        help="Enable distributed training on multiple GPUs...",
    )

    # Output configuration
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model.pt"),
        metavar="...",
        help="Path to save the trained model (default: model.pt)...",
    )

    # Parse and validate command line arguments
    args: argparse.Namespace = parser.parse_args()
    logger.info("Command line arguments parsed successfully...")

    # Validate selected module architecture
    if args.module not in modules:
        available_modules: str = ", ".join(modules.keys())
        logger.warning(
            f"Module '{args.module}' is not implemented. Available modules: {available_modules}..."
        )
        warnings.warn(
            f"Module '{args.module}' is not implemented. Available modules: {available_modules}...",
            UserWarning,
        )
        raise NotImplementedError(
            f"Module '{args.module}' is not implemented. Available modules are: {available_modules}..."
        )
    else:
        logger.info(f"Selected module: {args.module}...")

    # Initialize the selected model architecture
    logger.info("Initializing selected segmentation model...")
    try:
        module: nn.Module = modules[args.module](
            classes=args.classes,
            channels=args.channels,
            weights=args.weights,
        )
        logger.info(
            f"Successfully initialized {args.module} with {args.classes} classes..."
        )
    except Exception as error:
        logger.error(f"Failed to initialize model: {str(error)}...")
        raise Exception(f"Model initialization failed: {str(error)}...", RuntimeWarning)

    # Initialize the training system
    logger.info("Initializing trainer...")
    try:
        trainer: Trainer = Trainer(
            module=module,
            training_path=args.training_path,
            validation_path=args.validation_path,
            testing_path=args.testing_path,
            weights_path=args.weights_path,
            dimensions=args.dimensions,
            epochs=args.epochs,
            batch=args.batch_size,
            lr=args.learning_rate,
            decay=args.weight_decay,
            gamma=args.gamma,
            workers=args.workers,
            seed=args.seed,
            parallelism=args.parallelism,
        )
        logger.info("Trainer initialized successfully...")
    except Exception as error:
        logger.error(f"Failed to initialize trainer: {str(error)}...")
        raise Exception(f"Trainer initialization failed: {str(error)}...", RuntimeWarning)

    # Execute the training process
    logger.info("Starting segmentation training process...")
    try:
        asyncio.run(trainer.train())
        logger.info("Training completed successfully...")
    except Exception as error:
        logger.error(f"Training failed: {str(error)}...")
        warnings.warn(f"Training process failed: {str(error)}...", RuntimeWarning)

    # Execute testing process
    logger.info("Starting testing process...")
    try:
        asyncio.run(trainer.test())
        logger.info("Testing completed successfully...")
    except Exception as error:
        logger.error(f"Testing failed: {str(error)}...")
        warnings.warn(f"Testing process failed: {str(error)}...", RuntimeWarning)

    # Save the trained model to specified output path
    logger.info(f"Saving trained model to: {args.output}...")
    try:
        trainer.save(filepath=args.output)
        logger.info("Model saved successfully...")
    except Exception as error:
        logger.error(f"Failed to save model: {str(error)}...")
        warnings.warn(f"Model saving failed: {str(error)}...", RuntimeWarning)

    logger.info("Segmentation training application completed successfully...")