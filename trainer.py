import os
import random
import asyncio
import modules
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from loader import DatasetLoader
from pathlib import Path
from typing import Optional, Union, Tuple
import logging
import warnings
from bar import Bar

# Set up logging so we can keep track of what's happening during training
logger = logging.getLogger("trainer")


class Trainer:
    """A trainer class that handles everything you need for training image classification models.

    This class takes care of the entire training process for popular models like AlexNet,
    ResNet, VGG, MobileNet, EfficientNet, and DenseNet. It loads your data, sets up the
    model, runs training and validation loops, tests the final model, and saves everything
    properly.

    It also handles multi-GPU training if you have multiple graphics cards, keeps detailed
    logs of what's happening, watches for overfitting, and automatically saves checkpoints.
    Plus, it sets up the right optimizers and learning rates for each type of model based
    on what works best in practice.
    """

    def __init__(
            self,
            module: nn.Module,
            training_path: Path,
            validation_path: Path,
            testing_path: Path,
            weights_path: Optional[Path] = None,
            dimensions: Optional[Tuple[int, int]] = None,
            epochs: Optional[int] = None,
            batch: Optional[int] = None,
            lr: Optional[Union[float, int]] = None,
            decay: Optional[Union[float, int]] = None,
            gamma: Optional[Union[float, int]] = None,
            momentum: Optional[Union[float, int]] = None,
            workers: Optional[int] = None,
            seed: Optional[int] = None,
            parallelism: bool = False,
    ) -> None:
        """Sets up everything needed to train your model.

        This is where all the magic happens - we check that everything looks good,
        set up your datasets, pick the best optimizer for your specific model type,
        and get everything ready for training.

        Args:
            module: The neural network you want to train
            training_path: Where your training images are stored
            validation_path: Where your validation images are stored
            testing_path: Where your test images are stored
            weights_path: Path to pre-trained weights if you want to start from there
            dimensions: Image size as (height, width) - all images get resized to this
            epochs: How many times to go through the entire training dataset
            batch: How many images to process at once
            lr: Learning rate - how big steps the model takes when learning
            decay: Weight decay for regularization (helps prevent overfitting)
            gamma: How much to reduce learning rate over time
            momentum: Momentum factor for SGD (helps with training stability)
            workers: Number of CPU threads for loading data
            seed: Random seed for reproducible results
            parallelism: Whether to use multiple GPUs if available

        Raises:
            TypeError: If you don't pass in a proper PyTorch model
            ValueError: If any of the paths don't exist or parameters are weird
        """

        # Make sure what you gave us is actually a PyTorch model
        if not isinstance(module, nn.Module):
            logger.error(f"Expected a PyTorch model, but got {type(module)}.")
            raise TypeError(f"Expected a PyTorch model, but got {type(module)}.")
        logger.info(f"Got a {type(module).__name__} model - looks good!")

        # Check that all your data folders actually exist
        if not isinstance(training_path, Path) or not training_path.exists():
            logger.error(f"Can't find training data at: {training_path}")
            raise ValueError(f"Can't find training data at: {training_path}")
        logger.info(f"Found training data at: {training_path}")

        if not isinstance(validation_path, Path) or not validation_path.exists():
            logger.error(f"Can't find validation data at: {validation_path}")
            raise ValueError(f"Can't find validation data at: {validation_path}")
        logger.info(f"Found validation data at: {validation_path}")

        if not isinstance(testing_path, Path) or not testing_path.exists():
            logger.error(f"Can't find test data at: {testing_path}")
            raise ValueError(f"Can't find test data at: {testing_path}")
        logger.info(f"Found test data at: {testing_path}")

        # Make sure image dimensions make sense
        if not isinstance(dimensions, tuple) or len(dimensions) != 2:
            logger.error(f"Image dimensions should be like (224, 224), got: {dimensions}")
            raise ValueError(f"Image dimensions should be like (224, 224), got: {dimensions}")
        logger.info(f"Will resize all images to: {dimensions}")

        # Check training parameters
        if epochs is None or not isinstance(epochs, int) or epochs <= 0:
            logger.error(f"Epochs should be a positive number, got: {epochs}")
            raise ValueError(f"Epochs should be a positive number, got: {epochs}")
        logger.info(f"Training for {epochs} epochs")

        if batch is None or not isinstance(batch, int) or batch <= 0:
            logger.error(f"Batch size should be a positive number, got: {batch}")
            raise ValueError(f"Batch size should be a positive number, got: {batch}")
        logger.info(f"Using batch size of {batch}")

        # Check if pre-trained weights path exists (if provided)
        if weights_path and (not isinstance(weights_path, Path) or not weights_path.exists()):
            logger.warning(f"Can't find weights file at: {weights_path}")
            warnings.warn(f"Can't find weights file at: {weights_path}", UserWarning)
        elif weights_path:
            logger.info(f"Will load pre-trained weights from: {weights_path}")

        # Give warnings if parameter types look wrong
        if lr is not None and not isinstance(lr, (float, int)):
            logger.warning(f"Learning rate should be a number, got: {type(lr)}")
            warnings.warn(f"Learning rate should be a number, got: {type(lr)}", UserWarning)
        elif lr is not None:
            logger.info(f"Using learning rate of {lr}")

        if decay is not None and not isinstance(decay, (float, int)):
            logger.warning(f"Weight decay should be a number, got: {type(decay)}")
            warnings.warn(f"Weight decay should be a number, got: {type(decay)}", UserWarning)
        elif decay is not None:
            logger.info(f"Using weight decay of {decay}")

        if gamma is not None and not isinstance(gamma, (float, int)):
            logger.warning(f"Gamma should be a number, got: {type(gamma)}")
            warnings.warn(f"Gamma should be a number, got: {type(gamma)}", UserWarning)
        elif gamma is not None:
            logger.info(f"Using gamma of {gamma}")

        if momentum is not None and not isinstance(momentum, (float, int)):
            logger.warning(f"Momentum should be a number, got: {type(momentum)}")
            warnings.warn(f"Momentum should be a number, got: {type(momentum)}", UserWarning)
        elif momentum is not None:
            logger.info(f"Using momentum of {momentum}")

        if workers is not None and not isinstance(workers, int):
            logger.warning(f"Number of workers should be an integer, got: {type(workers)}")
            warnings.warn(f"Number of workers should be an integer, got: {type(workers)}", UserWarning)
        elif workers is not None:
            logger.info(f"Using {workers} worker threads")

        if seed is not None and not isinstance(seed, int):
            logger.warning(f"Seed should be an integer, got: {type(seed)}")
            warnings.warn(f"Seed should be an integer, got: {type(seed)}", UserWarning)
        elif seed is not None:
            logger.info(f"Using random seed: {seed}")

        if parallelism is not None and not isinstance(parallelism, bool):
            logger.warning(f"Parallelism should be True or False, got: {type(parallelism)}")
            warnings.warn(f"Parallelism should be True or False, got: {type(parallelism)}", UserWarning)
        else:
            logger.info(f"Multi-GPU training: {parallelism}")

        # Set random seed if provided - this makes results reproducible
        if seed is not None:
            self.seed(seed=seed)
            logger.info(f"Set random seed to {seed} for reproducible results")

        # Figure out if we should use GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Move model to GPU/CPU and store settings
        self.module = module.to(self.device)
        self.dimensions = dimensions
        self.epochs = epochs
        self.workers = workers

        # Create a place to store training and validation losses
        self.cache = {
            "training": [],
            "validation": [],
        }
        logger.info("Set up loss tracking")

        # Use multiple GPUs if requested and available
        if parallelism and torch.cuda.device_count() > 1:
            self.module = nn.parallel.DistributedDataParallel(self.module)
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        else:
            logger.info("Using single GPU/CPU")

        # Load all the datasets
        try:
            self.training_dataset = self.loader(dirpath=training_path, batch=batch)
            logger.info("Successfully loaded training data")
        except Exception as error:
            logger.error(f"Problem loading training data: {error}")
            warnings.warn(f"Problem loading training data: {error}")

        try:
            self.validation_dataset = self.loader(dirpath=validation_path, batch=batch)
            logger.info("Successfully loaded validation data")
        except Exception as error:
            logger.error(f"Problem loading validation data: {error}")
            warnings.warn(f"Problem loading validation data: {error}")

        try:
            self.testing_dataset = self.loader(dirpath=testing_path, batch=batch)
            logger.info("Successfully loaded test data")
        except Exception as error:
            logger.error(f"Problem loading test data: {error}")
            warnings.warn(f"Problem loading test data: {error}")

        # Set up the loss function for classification
        self.criterion = nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss for classification")

        # Choose the best optimizer and scheduler for each model type
        # Different models work better with different settings based on research
        if isinstance(module, modules.AlexNet):
            # AlexNet does well with Adam and step-based learning rate reduction
            decay = decay or 0.0001
            gamma = gamma or 0.5
            momentum = momentum or 0.9
            lr = lr or 0.001

            optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)

        elif isinstance(module, modules.ResNet50):
            # ResNet50 likes SGD with cosine annealing for smooth learning rate changes
            decay = decay or 0.0005
            gamma = gamma or 0.1
            momentum = momentum or 0.9
            lr = lr or 0.01

            optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        elif isinstance(module, modules.ResNet18):
            # ResNet18 works best with SGD and milestone-based learning rate drops
            decay = decay or 0.0001
            gamma = gamma or 0.5
            momentum = momentum or 0.9
            lr = lr or 0.01

            optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=gamma)

        elif isinstance(module, modules.MobileNetLarge):
            # MobileNet models are designed for mobile, so RMSprop with exponential decay works well
            decay = decay or 0.0004
            gamma = gamma or 0.95
            momentum = momentum or 0.9
            lr = lr or 0.045

            optimizer = torch.optim.RMSprop(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        elif isinstance(module, modules.MobileNetSmall):
            # Smaller MobileNet needs faster decay since it has less capacity
            decay = decay or 0.0004
            gamma = gamma or 0.9  # Faster decay for smaller model
            momentum = momentum or 0.9
            lr = lr or 0.045

            optimizer = torch.optim.RMSprop(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        elif isinstance(module, modules.VGG11):
            # VGG models like SGD with scheduled learning rate drops
            decay = decay or 0.0005
            gamma = gamma or 0.1
            momentum = momentum or 0.9
            lr = lr or 0.01

            optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 60, 90], gamma=gamma
            )

        elif isinstance(module, modules.VGG16):
            # Same setup as VGG11 but can handle the deeper network
            decay = decay or 0.0005
            gamma = gamma or 0.1
            momentum = momentum or 0.9
            lr = lr or 0.01

            optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 60, 90], gamma=gamma
            )

        elif isinstance(module, modules.EfficientNetB0):
            # EfficientNets work great with Adam and cosine annealing
            decay = decay or 0.0001
            gamma = gamma or 0.95
            momentum = momentum or 0.9
            lr = lr or 0.001

            optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        elif isinstance(module, modules.EfficientNetB3):
            # Same as B0 but can handle being a bit larger
            decay = decay or 0.0001
            gamma = gamma or 0.95
            momentum = momentum or 0.9
            lr = lr or 0.001

            optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        elif isinstance(module, modules.DenseNet121):
            # DenseNets do well with SGD and milestone scheduling
            decay = decay or 0.0001
            gamma = gamma or 0.1
            momentum = momentum or 0.9
            lr = lr or 0.01

            optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 60, 90], gamma=gamma
            )

        elif isinstance(module, modules.DenseNet169):
            # Same as DenseNet121 but handles the deeper network fine
            decay = decay or 0.0001
            gamma = gamma or 0.1
            momentum = momentum or 0.9
            lr = lr or 0.01

            optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 60, 90], gamma=gamma
            )

        else:
            logger.error(f"Don't know how to train {type(module)} - not supported yet")
            raise ValueError(f"Don't know how to train {type(module)} - not supported yet")

        # Store optimizer and scheduler for training
        self.optimizer = optimizer
        self.scheduler = scheduler

        logger.info(
            f"Set up {module.__class__.__name__} with "
            f"decay={decay}, gamma={gamma}, momentum={momentum}, lr={lr}"
        )

    @staticmethod
    def seed(seed: int) -> None:
        """
        Set random seeds everywhere so you get the same results every time.

        This is really important if you want to be able to reproduce your experiments
        or compare different approaches fairly. It sets the seed for Python's random
        module, NumPy, PyTorch, and GPU operations.

        Args:
            seed: The number to use as the random seed

        Returns:
            None
        """
        try:
            # Set Python's hash randomization
            os.environ["PYTHONHASHSEED"] = str(seed)
            logger.info("Configured Python hash randomization")

            # Set seeds for all the random number generators
            torch.manual_seed(seed=seed)  # PyTorch CPU stuff
            random.seed(a=seed)  # Python's built-in random
            np.random.seed(seed=seed)  # NumPy random numbers
            logger.info("Set all random seeds")

            # If we're using GPU, set those seeds too
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed=seed)  # All GPU devices
                # Make everything deterministic (slower but reproducible)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info("Made GPU operations deterministic")

            logger.info(f"Successfully set random seed to {seed}")
        except Exception as error:
            logger.error(f"Couldn't set seed: {str(error)}")
            warnings.warn(f"Couldn't set seed: {str(error)}")

    def loader(self, dirpath, batch):
        """
        Create a DataLoader that handles loading and preprocessing images efficiently.

        This sets up the pipeline that loads images, resizes them, converts them to
        tensors, and normalizes them. It also uses multiple CPU cores to load data
        in parallel so your GPU doesn't have to wait around.

        Args:
            dirpath: Path to the folder with images
            batch: How many images to load at once

        Returns:
            DataLoader that yields batches of preprocessed images, or None if it fails
        """
        # Set up image preprocessing - resize, convert to tensor, normalize
        transform = transforms.Compose([
            transforms.Resize(size=self.dimensions),  # Make all images the same size
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1] range
        ])
        logger.info("Set up image preprocessing pipeline")

        try:
            # Create the dataset with transformations
            dataset = DatasetLoader(dirpath=dirpath, transform=transform)
            logger.info(f"Created dataset from {dirpath}")

            # Create DataLoader with performance optimizations
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch,
                shuffle=True,  # Randomize order so model doesn't memorize sequence
                num_workers=(2 if self.workers is None else self.workers),  # Parallel loading
                pin_memory=True,  # Faster GPU transfers
            )
            logger.info(
                f"Created DataLoader with batch_size={batch}, "
                f"workers={2 if self.workers is None else self.workers}"
            )
            return dataloader

        except Exception as error:
            logger.error(f"Couldn't load data: {str(error)}")
            warnings.warn(f"Couldn't load data: {str(error)}")
            return None

    async def rehearse(self, dataloader, mode):
        """
        Run one epoch of training or validation.

        This goes through all the data once, either training the model (updating weights)
        or just evaluating it (validation/testing). It includes gradient clipping during
        training to prevent the gradients from getting too big and messing up training.

        Args:
            dataloader: The DataLoader with batched data
            mode: Either "training" or "validation"

        Returns:
            The average loss for this epoch
        """
        # Set model to training or evaluation mode
        self.module.train() if mode == "training" else self.module.eval()
        total_loss = np.float64(0.0)
        logger.info(f"Starting {mode} epoch with {len(dataloader)} batches")

        # Process all batches with a progress bar
        async with Bar(iterations=len(dataloader), title=mode, steps=20) as bar:
            time = asyncio.get_event_loop().time()

            for batch, (inputs, targets) in enumerate(dataloader, start=1):
                # Make sure we actually got tensors
                if not isinstance(inputs, torch.Tensor):
                    logger.warning("Inputs should be tensors")
                    warnings.warn("Inputs should be tensors.")
                    continue

                try:
                    # Move data to GPU/CPU
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    self.optimizer.zero_grad()  # Clear old gradients

                    # Only compute gradients during training
                    with torch.set_grad_enabled(mode == "training"):
                        outputs = self.module(inputs)  # Forward pass
                        loss = self.criterion(outputs, targets)  # Calculate loss

                        # Check if loss is NaN (usually means something's wrong)
                        if torch.isnan(loss):
                            logger.warning("Got NaN loss! Something might be wrong with your data or model")
                            warnings.warn("Got NaN loss! Check your data and model.")
                            continue

                        # During training, compute gradients and update weights
                        if mode == "training":
                            try:
                                loss.backward()  # Compute gradients
                                # Clip gradients to prevent them from getting too big
                                torch.nn.utils.clip_grad_norm_(
                                    parameters=self.module.parameters(), max_norm=1.0
                                )
                                self.optimizer.step()  # Update model weights
                            except Exception as error:
                                logger.error(f"Problem with backward pass: {str(error)}")
                                warnings.warn(f"Problem with backward pass: {str(error)}")

                    # Add up the loss (weighted by batch size)
                    total_loss = np.add(
                        total_loss,
                        np.multiply(np.float64(loss.item()), np.float64(inputs.size(0))),
                    )

                    # Update progress bar
                    await bar.update(batch=batch, time=time)
                    await bar.postfix(loss=np.divide(total_loss, batch))

                except Exception as error:
                    logger.error(f"Problem with batch {batch}: {str(error)}")
                    warnings.warn(f"Problem processing batch: {str(error)}")
                    continue

            # Calculate average loss for this epoch
            average_loss = np.divide(total_loss, np.float64(len(dataloader)))
            logger.info(f"{mode.capitalize()} epoch done, average loss: {average_loss:.4f}")
            return average_loss

    async def train(self):
        """
        Run the complete training process for all epochs.

        This handles the main training loop - it runs training and validation for each
        epoch, watches for overfitting (when validation loss goes up while training loss
        goes down), adjusts the learning rate, and saves checkpoints when needed.

        Returns:
            None
        """
        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.epochs):
            try:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                print(f"Epoch {epoch + 1}/{self.epochs}")

                # Run both training and validation
                for mode, dataloader in [
                    ("training", self.training_dataset),
                    ("validation", self.validation_dataset)
                ]:
                    loss = await self.rehearse(dataloader=dataloader, mode=mode)

                    # Log the results
                    logger.info(f"Epoch {epoch + 1}/{self.epochs}, {mode.capitalize()} Loss: {loss:.4f}")

                    # Save loss for overfitting detection
                    self.cache[mode].append(loss)

                # Check for overfitting after the first epoch
                if np.greater(epoch, 0):
                    if (
                            self.cache["validation"][-1] > self.cache["validation"][-2]
                            and self.cache["training"][-1] < self.cache["training"][-2]
                    ):
                        logger.warning(f"Looks like overfitting at epoch {epoch + 1}")
                        warnings.warn(
                            f"Possible overfitting at epoch {epoch + 1}: "
                            f"validation loss went up while training loss went down."
                        )
                        # Save a checkpoint in case we want to go back
                        checkpoint_path = Path(f"checkpoints/checkpoint-{epoch + 1}.pth")
                        self.save(filepath=checkpoint_path)

                # Update the learning rate based on the scheduler type
                scheduler = type(self.scheduler)
                if scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(epoch)
                elif scheduler in [
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                    torch.optim.lr_scheduler.StepLR,
                    torch.optim.lr_scheduler.MultiStepLR,
                    torch.optim.lr_scheduler.ExponentialLR,
                ]:
                    self.optimizer.step()
                    self.scheduler.step()

                # Show current learning rate
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Learning rate is now: {lr:.6f}")

            except Exception as error:
                logger.error(f"Problem in epoch {epoch + 1}: {str(error)}")
                warnings.warn(f"Problem in epoch {epoch + 1}: {str(error)}")
                continue

        logger.info("Training finished!")

    async def test(self):
        """
        Test the trained model on the test set to see how well it actually performs.

        This runs the model on data it has never seen before to get an honest measure
        of how good it is. It calculates both the loss and accuracy.

        Returns:
            None
        """
        # Set model to evaluation mode (no gradient computation)
        self.module.eval()
        total_loss = np.float64(0.0)

        # Track predictions to calculate accuracy
        all_predictions = np.array([], dtype=np.int64)
        all_targets = np.array([], dtype=np.int64)

        async with Bar(iterations=len(self.testing_dataset), title="Testing", steps=20) as bar:
            time = asyncio.get_event_loop().time()

            for batch, (inputs, targets) in enumerate(self.testing_dataset, start=1):
                try:
                    # Make sure we got tensors
                    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                        warnings.warn(
                            f"Skipping batch: expected tensors but got {type(inputs)}/{type(targets)}"
                        )
                        continue

                    # Move to GPU/CPU
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)

                    # Run inference without computing gradients
                    with torch.no_grad():
                        outputs = self.module(inputs)
                        loss = self.criterion(outputs, targets)

                        # Add up loss
                        total_loss = np.add(
                            total_loss,
                            np.multiply(np.float64(loss.item()), np.float64(inputs.size(0))),
                        )

                        # Get predictions for accuracy
                        _, prediction = torch.max(outputs, 1)  # Get the class with highest score
                        all_predictions = np.concatenate(
                            (all_predictions, prediction.cpu().numpy()), axis=0
                        )
                        all_targets = np.concatenate((all_targets, targets.cpu().numpy()), axis=0)

                    # Update progress bar
                    await bar.update(batch, time)
                    await bar.postfix(loss=np.divide(total_loss, batch))

                except Exception as error:
                    warnings.warn(f"Problem with test batch {batch}: {str(error)}")
                    continue

        # Calculate final metrics
        accuracy = np.multiply(
            np.divide(np.sum((all_predictions == all_targets)), np.size(all_predictions)),
            np.float64(100)
        )
        average_loss = np.divide(total_loss, len(self.testing_dataset))

        # Show results
        print(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        logger.info(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def save(self, filepath=None):
        """
        Save the trained model to disk.

        For .pth files, it saves just the model weights (recommended because it's more
        flexible). For other file types, it saves the entire model including the
        architecture.

        Args:
            filepath: Where to save the model (optional)

        Returns:
            None
        """

        if not filepath:
            # Use default location if none provided
            parent = Path(__file__).parent
            filepath = Path(parent, "module.pt")
        else:
            # Make sure the directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

        # Choose save format based on file extension
        if filepath.suffix == ".pth":
            # Save just the weights (preferred method)
            torch.save(obj=self.module.state_dict(), f=filepath)
            print(f"Saved model weights to: {filepath}")
            logger.info(f"Saved model weights to: {filepath}")
        else:
            # Save the whole model
            torch.save(obj=self.module, f=filepath)
            print(f"Saved complete model to: {filepath}")
            logger.info(f"Saved complete model to: {filepath}")