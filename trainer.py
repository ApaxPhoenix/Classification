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

# Use logger configured in main application
logger = logging.getLogger("trainer")


class Trainer:
    """Classification model training controller with automatic optimization configuration.

    Handles complete training workflows for popular CNN architectures including
    AlexNet, ResNet variants, VGG models, MobileNet, EfficientNet, and DenseNet.
    Automatically selects appropriate optimizers, learning rate schedules, and
    hyperparameters based on the specific model architecture.

    Features include multi-GPU support, detailed logging, overfitting detection,
    automatic checkpointing, and performance monitoring throughout training.
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
        """Initialize training configuration and validate all parameters.

        Validates input parameters, configures datasets, and sets up model-specific
        optimization strategies based on established best practices for each
        architecture type.

        Args:
            module: PyTorch neural network model to train
            training_path: Directory containing training dataset
            validation_path: Directory containing validation dataset
            testing_path: Directory containing test dataset
            weights_path: Optional path to pre-trained model weights
            dimensions: Target image size as (height, width) tuple
            epochs: Number of training epochs to execute
            batch: Batch size for training and evaluation
            lr: Learning rate for optimizer (architecture-specific defaults used if None)
            decay: Weight decay regularization factor
            gamma: Learning rate scheduler decay factor
            momentum: SGD momentum parameter
            workers: Number of data loading worker processes
            seed: Random seed for reproducible results
            parallelism: Whether to enable multi-GPU parallel training

        Raises:
            TypeError: When module is not a PyTorch nn.Module
            ValueError: When paths don't exist or parameters are invalid
        """

        # Validate model type
        if not isinstance(module, nn.Module):
            logger.error(f"Expected nn.Module, received {type(module)}")
            raise TypeError(f"Expected PyTorch model, got {type(module)}")
        logger.info(f"Model validated: {type(module).__name__}")

        # Validate dataset paths
        if not isinstance(training_path, Path) or not training_path.exists():
            logger.error(f"Training path invalid: {training_path}")
            raise ValueError(f"Training directory not found: {training_path}")
        logger.info(f"Training dataset located: {training_path}")

        if not isinstance(validation_path, Path) or not validation_path.exists():
            logger.error(f"Validation path invalid: {validation_path}")
            raise ValueError(f"Validation directory not found: {validation_path}")
        logger.info(f"Validation dataset located: {validation_path}")

        if not isinstance(testing_path, Path) or not testing_path.exists():
            logger.error(f"Test path invalid: {testing_path}")
            raise ValueError(f"Test directory not found: {testing_path}")
        logger.info(f"Test dataset located: {testing_path}")

        # Validate image dimensions
        if not isinstance(dimensions, tuple) or len(dimensions) != 2:
            logger.error(f"Invalid dimensions format: {dimensions}")
            raise ValueError(f"Dimensions must be (height, width) tuple, got: {dimensions}")
        logger.info(f"Image target size: {dimensions}")

        # Validate training parameters
        if epochs is None or not isinstance(epochs, int) or epochs <= 0:
            logger.error(f"Invalid epochs value: {epochs}")
            raise ValueError(f"Epochs must be positive integer, got: {epochs}")
        logger.info(f"Training duration: {epochs} epochs")

        if batch is None or not isinstance(batch, int) or batch <= 0:
            logger.error(f"Invalid batch size: {batch}")
            raise ValueError(f"Batch size must be positive integer, got: {batch}")
        logger.info(f"Batch size configured: {batch}")

        # Validate weights path if provided
        if weights_path and (not isinstance(weights_path, Path) or not weights_path.exists()):
            logger.warning(f"Weights path not found: {weights_path}")
            warnings.warn(f"Pre-trained weights not accessible: {weights_path}", UserWarning)
        elif weights_path:
            logger.info(f"Pre-trained weights specified: {weights_path}")

        # Validate optional parameters with type checking
        if lr is not None and not isinstance(lr, (float, int)):
            logger.warning(f"Learning rate type invalid: {type(lr)}")
            warnings.warn(f"Learning rate should be numeric, got: {type(lr)}", UserWarning)
        elif lr is not None:
            logger.info(f"Learning rate override: {lr}")

        if decay is not None and not isinstance(decay, (float, int)):
            logger.warning(f"Weight decay type invalid: {type(decay)}")
            warnings.warn(f"Weight decay should be numeric, got: {type(decay)}", UserWarning)
        elif decay is not None:
            logger.info(f"Weight decay override: {decay}")

        if gamma is not None and not isinstance(gamma, (float, int)):
            logger.warning(f"Gamma type invalid: {type(gamma)}")
            warnings.warn(f"Gamma should be numeric, got: {type(gamma)}", UserWarning)
        elif gamma is not None:
            logger.info(f"Gamma override: {gamma}")

        if momentum is not None and not isinstance(momentum, (float, int)):
            logger.warning(f"Momentum type invalid: {type(momentum)}")
            warnings.warn(f"Momentum should be numeric, got: {type(momentum)}", UserWarning)
        elif momentum is not None:
            logger.info(f"Momentum override: {momentum}")

        if workers is not None and not isinstance(workers, int):
            logger.warning(f"Workers type invalid: {type(workers)}")
            warnings.warn(f"Worker count should be integer, got: {type(workers)}", UserWarning)
        elif workers is not None:
            logger.info(f"Data loading workers: {workers}")

        if seed is not None and not isinstance(seed, int):
            logger.warning(f"Seed type invalid: {type(seed)}")
            warnings.warn(f"Random seed should be integer, got: {type(seed)}", UserWarning)
        elif seed is not None:
            logger.info(f"Random seed configured: {seed}")

        if parallelism is not None and not isinstance(parallelism, bool):
            logger.warning(f"Parallelism type invalid: {type(parallelism)}")
            warnings.warn(f"Parallelism should be boolean, got: {type(parallelism)}", UserWarning)
        else:
            logger.info(f"Multi-GPU training enabled: {parallelism}")

        # Configure reproducible random state
        if seed is not None:
            self.seed(seed=seed)
            logger.info(f"Random seed applied: {seed}")

        # Determine compute device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Compute device selected: {self.device}")

        # Configure model and store parameters
        self.module = module.to(self.device)
        self.dimensions = dimensions
        self.epochs = epochs
        self.workers = workers

        # Initialize loss tracking storage
        self.cache = {
            "training": [],
            "validation": [],
        }
        logger.info("Loss tracking initialized")

        # Configure multi-GPU processing if requested
        if parallelism and torch.cuda.device_count() > 1:
            self.module = nn.parallel.DistributedDataParallel(self.module)
            logger.info(f"Multi-GPU training configured: {torch.cuda.device_count()} devices")
        else:
            logger.info("Single-device training mode")

        # Initialize dataset loaders
        try:
            self.training_dataset = self.loader(dirpath=training_path, batch=batch)
            logger.info("Training dataset loader created")
        except Exception as error:
            logger.error(f"Training dataset loading failed: {error}")
            warnings.warn(f"Training data loading error: {error}")

        try:
            self.validation_dataset = self.loader(dirpath=validation_path, batch=batch)
            logger.info("Validation dataset loader created")
        except Exception as error:
            logger.error(f"Validation dataset loading failed: {error}")
            warnings.warn(f"Validation data loading error: {error}")

        try:
            self.testing_dataset = self.loader(dirpath=testing_path, batch=batch)
            logger.info("Test dataset loader created")
        except Exception as error:
            logger.error(f"Test dataset loading failed: {error}")
            warnings.warn(f"Test data loading error: {error}")

        # Configure loss function for classification
        self.criterion = nn.CrossEntropyLoss()
        logger.info("CrossEntropyLoss criterion configured")

        # Configure architecture-specific optimization strategies
        # Each model type has empirically determined optimal settings
        if isinstance(module, modules.AlexNet):
            # AlexNet performs well with Adam and step scheduling
            decay = decay or 0.0001
            gamma = gamma or 0.5
            momentum = momentum or 0.9
            lr = lr or 0.001

            optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)

        elif isinstance(module, modules.ResNet50):
            # ResNet50 benefits from SGD with cosine annealing
            decay = decay or 0.0005
            gamma = gamma or 0.1
            momentum = momentum or 0.9
            lr = lr or 0.01

            optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        elif isinstance(module, modules.ResNet18):
            # ResNet18 works well with SGD and milestone scheduling
            decay = decay or 0.0001
            gamma = gamma or 0.5
            momentum = momentum or 0.9
            lr = lr or 0.01

            optimizer = torch.optim.SGD(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=gamma)

        elif isinstance(module, modules.MobileNetLarge):
            # MobileNet architectures optimized for RMSprop with exponential decay
            decay = decay or 0.0004
            gamma = gamma or 0.95
            momentum = momentum or 0.9
            lr = lr or 0.045

            optimizer = torch.optim.RMSprop(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        elif isinstance(module, modules.MobileNetSmall):
            # Smaller MobileNet requires faster decay due to reduced capacity
            decay = decay or 0.0004
            gamma = gamma or 0.9  # Accelerated decay for compact model
            momentum = momentum or 0.9
            lr = lr or 0.045

            optimizer = torch.optim.RMSprop(
                self.module.parameters(), lr=lr, momentum=momentum, weight_decay=decay
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        elif isinstance(module, modules.VGG11):
            # VGG architectures perform well with SGD and milestone scheduling
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
            # VGG16 uses same optimization strategy as VGG11
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
            # EfficientNet models optimized for Adam with cosine annealing
            decay = decay or 0.0001
            gamma = gamma or 0.95
            momentum = momentum or 0.9
            lr = lr or 0.001

            optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        elif isinstance(module, modules.EfficientNetB3):
            # EfficientNet B3 uses same strategy as B0 with scaling accommodation
            decay = decay or 0.0001
            gamma = gamma or 0.95
            momentum = momentum or 0.9
            lr = lr or 0.001

            optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        elif isinstance(module, modules.DenseNet121):
            # DenseNet architectures benefit from SGD with milestone scheduling
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
            # DenseNet169 uses same optimization as DenseNet121
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
            logger.error(f"Unsupported model type: {type(module)}")
            raise ValueError(f"Training configuration not available for: {type(module)}")

        # Store optimization components
        self.optimizer = optimizer
        self.scheduler = scheduler

        logger.info(
            f"Optimization configured for {module.__class__.__name__}: "
            f"decay={decay}, gamma={gamma}, momentum={momentum}, lr={lr}"
        )

    @staticmethod
    def seed(seed: int) -> None:
        """
        Configure deterministic random state across all libraries.

        Ensures reproducible results by setting random seeds for Python's
        random module, NumPy, PyTorch CPU and GPU operations, and configuring
        deterministic CUDA behavior.

        Args:
            seed: Integer value for random seed initialization

        Returns:
            None
        """
        try:
            # Configure Python hash randomization
            os.environ["PYTHONHASHSEED"] = str(seed)
            logger.info("Python hash randomization configured")

            # Set random seeds for all generators
            torch.manual_seed(seed=seed)  # PyTorch CPU operations
            random.seed(a=seed)  # Python built-in random module
            np.random.seed(seed=seed)  # NumPy random operations
            logger.info("Random seeds applied to all generators")

            # Configure GPU determinism if CUDA available
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed=seed)  # All CUDA devices
                # Enable deterministic operations (performance impact)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info("GPU operations configured for determinism")

            logger.info(f"Deterministic random state established: seed={seed}")
        except Exception as error:
            logger.error(f"Random seed configuration failed: {str(error)}")
            warnings.warn(f"Unable to set random seed: {str(error)}")

    def loader(self, dirpath, batch):
        """
        Create DataLoader with preprocessing pipeline and performance optimization.

        Configures image preprocessing (resize, tensor conversion, normalization)
        and DataLoader with multi-process loading for efficient GPU utilization.

        Args:
            dirpath: Directory path containing image dataset
            batch: Batch size for data loading

        Returns:
            Configured DataLoader instance, or None if creation fails
        """
        # Configure image preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize(size=self.dimensions),  # Standardize image dimensions
            transforms.ToTensor(),  # Convert to PyTorch tensor format
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1] range
        ])
        logger.info("Image preprocessing pipeline configured")

        try:
            # Initialize dataset with transforms
            dataset = DatasetLoader(dirpath=dirpath, transform=transform)
            logger.info(f"Dataset created from: {dirpath}")

            # Configure DataLoader with performance optimizations
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch,
                shuffle=True,  # Randomize sample order each epoch
                num_workers=(2 if self.workers is None else self.workers),  # Parallel loading
                pin_memory=True,  # Accelerate GPU memory transfers
            )
            logger.info(
                f"DataLoader configured: batch_size={batch}, "
                f"workers={2 if self.workers is None else self.workers}"
            )
            return dataloader

        except Exception as error:
            logger.error(f"DataLoader creation failed: {str(error)}")
            warnings.warn(f"Data loading error: {str(error)}")
            return None

    async def rehearse(self, dataloader, mode):
        """
        Execute single epoch of training or evaluation.

        Processes all batches in the dataloader, performing forward passes and
        optionally backward passes (training only). Includes gradient clipping
        during training to prevent gradient explosion.

        Args:
            dataloader: DataLoader containing batched dataset
            mode: Training mode - either "training" or "validation"

        Returns:
            Average loss value for the epoch
        """
        # Configure model state based on mode
        self.module.train() if mode == "training" else self.module.eval()
        total_loss = np.float64(0.0)
        logger.info(f"Beginning {mode} epoch: {len(dataloader)} batches")

        # Process all batches with progress tracking
        async with Bar(iterations=len(dataloader), title=mode, steps=20) as bar:
            time = asyncio.get_event_loop().time()

            for batch, (inputs, targets) in enumerate(dataloader, start=1):
                # Validate tensor inputs
                if not isinstance(inputs, torch.Tensor):
                    logger.warning("Non-tensor inputs detected")
                    warnings.warn("Expected tensor inputs for processing")
                    continue

                try:
                    # Transfer data to compute device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    self.optimizer.zero_grad()  # Clear accumulated gradients

                    # Compute forward pass with conditional gradient computation
                    with torch.set_grad_enabled(mode == "training"):
                        outputs = self.module(inputs)  # Model inference
                        loss = self.criterion(outputs, targets)  # Loss calculation

                        # Detect numerical instability
                        if torch.isnan(loss):
                            logger.warning("NaN loss detected - possible numerical instability")
                            warnings.warn("NaN loss indicates potential data or model issues")
                            continue

                        # Execute backward pass during training
                        if mode == "training":
                            try:
                                loss.backward()  # Gradient computation
                                # Apply gradient clipping to prevent explosion
                                torch.nn.utils.clip_grad_norm_(
                                    parameters=self.module.parameters(), max_norm=1.0
                                )
                                self.optimizer.step()  # Parameter update
                            except Exception as error:
                                logger.error(f"Backward pass failed: {str(error)}")
                                warnings.warn(f"Gradient computation error: {str(error)}")

                    # Accumulate batch loss weighted by batch size
                    total_loss = np.add(
                        total_loss,
                        np.multiply(np.float64(loss.item()), np.float64(inputs.size(0))),
                    )

                    # Update progress display
                    await bar.update(batch=batch, time=time)
                    await bar.postfix(loss=np.divide(total_loss, batch))

                except Exception as error:
                    logger.error(f"Batch processing error {batch}: {str(error)}")
                    warnings.warn(f"Batch {batch} processing failed: {str(error)}")
                    continue

            # Calculate epoch average loss
            average_loss = np.divide(total_loss, np.float64(len(dataloader)))
            logger.info(f"{mode.capitalize()} epoch completed - average loss: {average_loss:.4f}")
            return average_loss

    async def train(self):
        """
        Execute complete training process across all epochs.

        Manages the full training loop including training and validation phases,
        overfitting detection, learning rate scheduling, and checkpoint creation
        when potential overfitting is detected.

        Returns:
            None
        """
        logger.info(f"Training initiation: {self.epochs} epochs scheduled")

        for epoch in range(self.epochs):
            try:
                logger.info(f"Epoch progression: {epoch + 1}/{self.epochs}")
                print(f"Epoch {epoch + 1}/{self.epochs}")

                # Execute training and validation phases
                for mode, dataloader in [
                    ("training", self.training_dataset),
                    ("validation", self.validation_dataset)
                ]:
                    loss = await self.rehearse(dataloader=dataloader, mode=mode)

                    # Record epoch results
                    logger.info(f"Epoch {epoch + 1}/{self.epochs} - {mode.capitalize()} Loss: {loss:.4f}")

                    # Store loss values for overfitting analysis
                    self.cache[mode].append(loss)

                # Analyze for overfitting after initial epoch
                if np.greater(epoch, 0):
                    if (
                            self.cache["validation"][-1] > self.cache["validation"][-2]
                            and self.cache["training"][-1] < self.cache["training"][-2]
                    ):
                        logger.warning(f"Potential overfitting detected at epoch {epoch + 1}")
                        warnings.warn(
                            f"Overfitting indication at epoch {epoch + 1}: "
                            f"validation loss increased while training loss decreased"
                        )
                        # Create checkpoint for potential rollback
                        checkpoint_path = Path(f"checkpoints/checkpoint-{epoch + 1}.pth")
                        self.save(filepath=checkpoint_path)

                # Update learning rate based on scheduler configuration
                scheduler_type = type(self.scheduler)
                if scheduler_type == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(epoch)
                elif scheduler_type in [
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                    torch.optim.lr_scheduler.StepLR,
                    torch.optim.lr_scheduler.MultiStepLR,
                    torch.optim.lr_scheduler.ExponentialLR,
                ]:
                    self.optimizer.step()
                    self.scheduler.step()

                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Learning rate updated: {current_lr:.6f}")

            except Exception as error:
                logger.error(f"Epoch {epoch + 1} execution failed: {str(error)}")
                warnings.warn(f"Epoch {epoch + 1} error: {str(error)}")
                continue

        logger.info("Training process completed successfully")

    async def test(self):
        """
        Evaluate trained model performance on test dataset.

        Executes inference on test data to measure model generalization
        capability. Calculates both loss and classification accuracy metrics.

        Returns:
            None
        """
        # Configure model for evaluation mode
        self.module.eval()
        total_loss = np.float64(0.0)

        # Initialize prediction tracking for accuracy calculation
        all_predictions = np.array([], dtype=np.int64)
        all_targets = np.array([], dtype=np.int64)

        async with Bar(iterations=len(self.testing_dataset), title="Testing", steps=20) as bar:
            time = asyncio.get_event_loop().time()

            for batch, (inputs, targets) in enumerate(self.testing_dataset, start=1):
                try:
                    # Validate input tensor types
                    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                        warnings.warn(
                            f"Batch {batch} skipped: expected tensors, got {type(inputs)}/{type(targets)}"
                        )
                        continue

                    # Transfer to compute device
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)

                    # Execute inference without gradient computation
                    with torch.no_grad():
                        outputs = self.module(inputs)
                        loss = self.criterion(outputs, targets)

                        # Accumulate loss values
                        total_loss = np.add(
                            total_loss,
                            np.multiply(np.float64(loss.item()), np.float64(inputs.size(0))),
                        )

                        # Extract predictions for accuracy computation
                        _, prediction = torch.max(outputs, 1)  # Get highest confidence class
                        all_predictions = np.concatenate(
                            (all_predictions, prediction.cpu().numpy()), axis=0
                        )
                        all_targets = np.concatenate((all_targets, targets.cpu().numpy()), axis=0)

                    # Update progress tracking
                    await bar.update(batch, time)
                    await bar.postfix(loss=np.divide(total_loss, batch))

                except Exception as error:
                    warnings.warn(f"Test batch {batch} processing failed: {str(error)}")
                    continue

        # Compute final performance metrics
        accuracy = np.multiply(
            np.divide(np.sum((all_predictions == all_targets)), np.size(all_predictions)),
            np.float64(100)
        )
        average_loss = np.divide(total_loss, len(self.testing_dataset))

        # Display test results
        print(f"Test Results - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        logger.info(f"Test evaluation completed - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def save(self, filepath=None):
        """
        Persist trained model to storage.

        Saves model weights (.pth files) or complete model architecture based on
        file extension. Creates necessary directories if they don't exist.

        Args:
            filepath: Target save location (uses default if None)

        Returns:
            None
        """

        if not filepath:
            # Use default save location
            parent = Path(__file__).parent
            filepath = Path(parent, "module.pt")
        else:
            # Ensure target directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

        # Select save format based on file extension
        if filepath.suffix == ".pth":
            # Save state dictionary (weights only - recommended)
            torch.save(obj=self.module.state_dict(), f=filepath)
            print(f"Model weights saved: {filepath}")
            logger.info(f"Model state dictionary saved: {filepath}")
        else:
            # Save complete model including architecture
            torch.save(obj=self.module, f=filepath)
            print(f"Complete model saved: {filepath}")
            logger.info(f"Full model architecture saved: {filepath}")