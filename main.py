import argparse
import asyncio
import logging.config
import warnings
import torch.nn as nn
from pathlib import Path
from typing import Dict, Type
from trainer import Trainer
import modules

# Set up logging to track what's happening during training
configuration = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        # Main app logs
        "main": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "main.log",
            "mode": "w",
        },
        # Data loading logs
        "loader": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "loader.log",
            "mode": "w",
        },
        # Model operation logs
        "modules": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "modules.log",
            "mode": "w",
        },
        # Training process logs
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

# All the different classification models we can train
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
    # Get logging working
    logging.config.dictConfig(configuration)
    logger = logging.getLogger("main")
    logger.info("Starting up the classification trainer...")

    # Set up command line arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train classification models with PyTorch"
    )

    # Which model do you want to train?
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        required=True,
        metavar="...",
        help=f"Pick your model. Options: {', '.join(modules.keys())}",
    )

    # Should we start with pretrained weights?
    parser.add_argument(
        "-w",
        "--weights",
        type=bool,
        default=True,
        metavar="...",
        help="Use pretrained weights? (Default: True)",
    )

    # How many classes are we trying to classify?
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=True,
        metavar="...",
        help="Number of different classes to classify",
    )

    # Image channels (RGB = 3, grayscale = 1)
    parser.add_argument(
        "-ch",
        "--channels",
        type=int,
        default=3,
        metavar="...",
        help="Image channels (3 for RGB, 1 for grayscale)",
    )

    # Where's our data?
    parser.add_argument(
        "-tp",
        "--training-path",
        type=Path,
        required=True,
        metavar="...",
        help="Path to your training data folder",
    )

    parser.add_argument(
        "-vp",
        "--validation-path",
        type=Path,
        required=True,
        metavar="...",
        help="Path to your validation data folder",
    )

    parser.add_argument(
        "-tep",
        "--testing-path",
        type=Path,
        default=None,
        metavar="...",
        help="Path to test data (optional)",
    )

    # Got existing weights to load?
    parser.add_argument(
        "-wp",
        "--weights-path",
        type=Path,
        default=None,
        metavar="...",
        help="Path to saved model weights (optional)",
    )

    # Image size settings
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        nargs=2,
        default=(64, 64),
        metavar="...",
        help="Image size as width height",
    )

    # Training settings
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=25,
        metavar="...",
        help="How many epochs to train for",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="...",
        help="Batch size for training",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0001,
        metavar="...",
        help="Learning rate",
    )

    parser.add_argument(
        "-wk",
        "--workers",
        type=int,
        default=4,
        metavar="...",
        help="Number of data loading workers",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        metavar="...",
        help="Random seed for reproducible results",
    )

    # Advanced training options
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=None,
        metavar="...",
        help="Weight decay for regularization",
    )

    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=None,
        metavar="...",
        help="Learning rate scheduler gamma",
    )

    parser.add_argument(
        "-mm",
        "--momentum",
        type=float,
        default=None,
        metavar="...",
        help="SGD momentum",
    )

    # Where to save the trained model
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="...",
        help="Where to save your trained model",
    )

    # Parse all the arguments
    args: argparse.Namespace = parser.parse_args()
    logger.info("Got all the command line arguments")

    # Make sure they picked a valid model
    if args.module not in modules:
        types: str = ", ".join(modules.keys())
        logger.warning(f"'{args.module}' isn't a valid model. Try: {types}")
        warnings.warn(f"'{args.module}' isn't available. Options: {types}", UserWarning)
    else:
        logger.info(f"Using {args.module} model")

    # Create the model
    logger.info("Setting up the model...")
    try:
        model: nn.Module = modules[args.module](
            classes=args.classes,
            channels=args.channels,
            weights=args.weights,
        )
        logger.info(f"Model ready! {args.module} set up for {args.classes} classes")
    except Exception as error:
        logger.error(f"Couldn't create model: {str(error)}")
        raise Exception(f"Model setup failed: {str(error)}", RuntimeWarning)

    # Set up the trainer
    logger.info("Getting the trainer ready...")
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
        logger.info("Trainer is ready to go")
    except Exception as error:
        logger.error(f"Trainer setup failed: {str(error)}")
        raise Exception(f"Couldn't set up trainer: {str(error)}", RuntimeWarning)

    # Start training!
    logger.info("Starting training...")
    try:
        asyncio.run(trainer.train())
        logger.info("Training finished!")
    except Exception as error:
        logger.error(f"Training crashed: {str(error)}")
        warnings.warn(f"Training failed: {str(error)}", RuntimeWarning)

    # Test the model if we have test data
    if args.testing_path is not None:
        logger.info("Running final tests...")
        try:
            asyncio.run(trainer.test())
            logger.info("Testing complete!")
        except Exception as error:
            logger.error(f"Testing failed: {str(error)}")
            warnings.warn(f"Testing didn't work: {str(error)}", RuntimeWarning)
    else:
        logger.info("No test data provided, skipping tests")

    # Save the model if they want us to
    if args.output:
        logger.info(f"Saving model to {args.output}...")
        try:
            trainer.save(filepath=args.output)
            logger.info("Model saved!")
        except Exception as error:
            logger.error(f"Couldn't save model: {str(error)}")
            warnings.warn(f"Model saving failed: {str(error)}", RuntimeWarning)
    else:
        logger.warning("No save path provided - your model won't be saved")

    logger.info("All done!")