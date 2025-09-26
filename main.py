import argparse
import asyncio
import logging.config
import warnings
import torch.nn as nn
from pathlib import Path
from typing import Dict, Type
from trainer import Trainer
import modules

# Configure logging system for tracking training progress
configuration = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        # Application-level logging
        "main": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "main.log",
            "mode": "w",
        },
        # Dataset loading operations
        "loader": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "loader.log",
            "mode": "w",
        },
        # Model architecture operations
        "modules": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "modules.log",
            "mode": "w",
        },
        # Training and validation logs
        "trainer": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "trainer.log",
            "mode": "w",
        },
    },
    "loggers": {
        "main": {"handlers": ["main"], "level": "INFO", "propagate": False},
        "loader": {"handlers": ["loader"], "level": "INFO", "propagate": False},
        "modules": {"handlers": ["modules"], "level": "INFO", "propagate": False},
        "trainer": {"handlers": ["trainer"], "level": "INFO", "propagate": False},
    },
}

# Available classification model architectures
modules: Dict[str, Type[nn.Module]] = {
    "alexnet": modules.AlexNet,
    "resnet50": modules.ResNet50,
    "resnet18": modules.ResNet18,
    "mobilenetv3large": modules.MobileNetLarge,
    "mobilenetv3small": modules.MobileNetSmall,
    "vgg11": modules.VGG11,
    "vgg16": modules.VGG16,
    "efficientnet-b0": modules.EfficientNetB0,
    "efficientnet-b3": modules.EfficientNetB3,
    "densenet121": modules.DenseNet121,
    "densenet169": modules.DenseNet169,
}

if __name__ == "__main__":
    # Initialize logging system
    logging.config.dictConfig(configuration)
    logger = logging.getLogger("main")
    logger.info("Initializing classification model training pipeline")

    # Configure command line argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train image classification models using PyTorch"
    )

    # Model architecture selection
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        required=True,
        metavar="...",
        help=f"Choose model architecture from: {', '.join(modules.keys())}",
    )

    # Pre-trained weight initialization
    parser.add_argument(
        "-w",
        "--weights",
        type=bool,
        default=True,
        metavar="...",
        help="Initialize with ImageNet pre-trained weights (recommended)",
    )

    # Dataset configuration
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=True,
        metavar="...",
        help="Number of classification classes in dataset",
    )

    parser.add_argument(
        "-ch",
        "--channels",
        type=int,
        default=3,
        metavar="...",
        help="Input image channels (3 for RGB, 1 for grayscale)",
    )

    # Dataset paths
    parser.add_argument(
        "-tp",
        "--training-path",
        type=Path,
        required=True,
        metavar="...",
        help="Directory containing training dataset",
    )

    parser.add_argument(
        "-vp",
        "--validation-path",
        type=Path,
        required=True,
        metavar="...",
        help="Directory containing validation dataset",
    )

    parser.add_argument(
        "-tep",
        "--testing-path",
        type=Path,
        default=None,
        metavar="...",
        help="Directory containing test dataset (optional)",
    )

    # Model checkpoint loading
    parser.add_argument(
        "-wp",
        "--weights-path",
        type=Path,
        default=None,
        metavar="...",
        help="Path to existing model checkpoint (optional)",
    )

    # Image preprocessing parameters
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        nargs=2,
        default=(64, 64),
        metavar="...",
        help="Input image dimensions as width height",
    )

    # Training hyperparameters
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=25,
        metavar="...",
        help="Number of training epochs",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="...",
        help="Training batch size",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0001,
        metavar="...",
        help="Optimizer learning rate",
    )

    parser.add_argument(
        "-wk",
        "--workers",
        type=int,
        default=4,
        metavar="...",
        help="Number of data loading worker processes",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        metavar="...",
        help="Random seed for reproducible training",
    )

    # Advanced optimization parameters
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=None,
        metavar="...",
        help="L2 regularization weight decay factor",
    )

    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=None,
        metavar="...",
        help="Learning rate scheduler decay factor",
    )

    parser.add_argument(
        "-mm",
        "--momentum",
        type=float,
        default=None,
        metavar="...",
        help="SGD optimizer momentum parameter",
    )

    # Output configuration
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="...",
        help="Output path for saving trained model weights",
    )

    # Parse command line arguments
    args: argparse.Namespace = parser.parse_args()
    logger.info("Command line arguments parsed successfully")

    # Validate model architecture selection
    if args.module not in modules:
        types: str = ", ".join(modules.keys())
        logger.warning(f"Invalid model '{args.module}' specified. Available: {types}")
        warnings.warn(f"Model '{args.module}' not available. Options: {types}", UserWarning)
    else:
        logger.info(f"Selected model architecture: {args.module}")

    # Initialize model instance
    logger.info("Creating model instance")
    try:
        model: nn.Module = modules[args.module](
            classes=args.classes,
            channels=args.channels,
            weights=args.weights,
        )
        logger.info(f"Model initialized: {args.module} with {args.classes} output classes")
    except Exception as error:
        logger.error(f"Model initialization failed: {str(error)}")
        raise Exception(f"Unable to create model: {str(error)}", RuntimeWarning)

    # Configure training pipeline
    logger.info("Setting up training controller")
    try:
        trainer: Trainer = Trainer(
            module=model,
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
            momentum=args.momentum,
            workers=args.workers,
            seed=args.seed,
        )
        logger.info("Training pipeline configured successfully")
    except Exception as error:
        logger.error(f"Trainer initialization error: {str(error)}")
        raise Exception(f"Training setup failed: {str(error)}", RuntimeWarning)

    # Execute training phase
    logger.info("Beginning model training")
    try:
        asyncio.run(trainer.train())
        logger.info("Training phase completed")
    except Exception as error:
        logger.error(f"Training process failed: {str(error)}")
        warnings.warn(f"Training interrupted: {str(error)}", RuntimeWarning)

    # Run evaluation on test set if available
    if args.testing_path is not None:
        logger.info("Evaluating model on test dataset")
        try:
            asyncio.run(trainer.test())
            logger.info("Model evaluation completed")
        except Exception as error:
            logger.error(f"Testing phase failed: {str(error)}")
            warnings.warn(f"Evaluation error: {str(error)}", RuntimeWarning)
    else:
        logger.info("Test dataset not provided - skipping evaluation")

    # Save trained model weights
    if args.output:
        logger.info(f"Saving model weights to {args.output}")
        try:
            trainer.save(filepath=args.output)
            logger.info("Model weights saved successfully")
        except Exception as error:
            logger.error(f"Model saving failed: {str(error)}")
            warnings.warn(f"Unable to save model: {str(error)}", RuntimeWarning)
    else:
        logger.warning("Output path not specified - model will not be saved")

    logger.info("Training pipeline execution completed")