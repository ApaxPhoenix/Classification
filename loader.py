import torch
from PIL import Image, UnidentifiedImageError
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import logging
import warnings

# Use logger configured in main application
logger = logging.getLogger("loader")


class DatasetLoader(Dataset):
    """
    Dataset loader for classification tasks with folder-based organization.

    Handles image datasets organized in directories where each subdirectory
    represents a different class. Built on PyTorch's ImageFolder with enhanced
    error handling for corrupted or unreadable images.

    Expected directory structure:
        dataset/
        ├── class1/
        │   ├── class001.jpg
        │   ├── class002.png
        │   └── ...
        └── class2/
            ├── class001.jpg
            ├── class002.png
            └── ...

    The subdirectory names become class labels, and all supported image
    formats within each directory are loaded as training samples.
    """

    def __init__(
        self,
        dirpath: Path,
        transform: Optional[Callable[[Image], torch.Tensor]] = None,
    ) -> None:
        """
        Initialize dataset loader with directory path and transforms.

        Args:
            dirpath: Root directory containing class subdirectories
            transform: Optional preprocessing pipeline for images

        Raises:
            ValueError: When directory doesn't exist or contains no class folders
        """
        # Validate directory existence
        if not dirpath.is_dir():
            logger.error(f"Directory not found: {dirpath}")
            raise ValueError(f"Directory {dirpath} does not exist or is not accessible")

        # Check for empty directory
        if not any(dirpath.iterdir()):
            logger.error(f"Empty directory: {dirpath}")
            raise ValueError(f"Directory {dirpath} is empty - requires class subdirectories")

        # Store transformation pipeline
        self.transform: Optional[Callable[[Image], torch.Tensor]] = transform

        # Initialize PyTorch ImageFolder for directory traversal
        self.dataset: ImageFolder = ImageFolder(root=str(dirpath))

        logger.info(f"Dataset initialized: {len(self.dataset)} images from {dirpath}")

    def __len__(self) -> int:
        """
        Return total number of images in the dataset.

        Returns:
            Count of all images across all class directories
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Union[Tuple[Image, int], Tuple[None, None]]:
        """
        Retrieve image and label at specified index with error recovery.

        Attempts to load the requested image, automatically skipping corrupted
        files and trying subsequent indices until a valid image is found or
        all images are exhausted.

        Args:
            idx: Index of image to retrieve

        Returns:
            Tuple containing (image, label) on success,
            or (None, None) if no valid images remain
        """
        try:
            # Attempt to load image and associated label
            image: Union[Image, torch.Tensor]
            label: int
            image, label = self.dataset[idx]

            # Ensure PIL Image format for consistency
            if not isinstance(image, Image):
                image = transforms.ToPILImage()(image)

            # Apply preprocessing transforms if specified
            if self.transform:
                image = self.transform(image)

            return image, label

        except (UnidentifiedImageError, OSError) as error:
            # Handle corrupted or unreadable image files
            logger.error(f"Failed to load image at index {idx}: {error}")
            warnings.warn(f"Skipping corrupted image at index {idx}: {str(error)}")

            # Prevent infinite loop when all images are corrupted
            if idx + 1 >= len(self):
                logger.error("All remaining images unreadable - dataset may be corrupted")
                warnings.warn("No valid images available - verify dataset integrity")
                return None, None

            # Recursively try the next index
            return self.__getitem__(idx + 1)
