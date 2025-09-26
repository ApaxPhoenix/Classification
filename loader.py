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

# Get our logger for tracking what's happening
logger = logging.getLogger("loader")


class DatasetLoader(Dataset):
    """
    A simple dataset loader for images organized in folders.

    This basically takes a folder structure where each subfolder is a different
    class (like "cats", "dogs", "cars") and loads all the images from those
    folders. It's built on top of PyTorch's ImageFolder but with better error
    handling so it won't crash on corrupted images.

    Your folder should look like:
    my_dataset/
    ├── cats/
    │   ├── cat1.jpg
    │   ├── cat2.png
    │   └── ...
    ├── dogs/
    │   ├── dog1.jpg
    │   ├── dog2.png
    │   └── ...
    └── cars/
        ├── car1.jpg
        └── ...
    """

    def __init__(
        self,
        dirpath: Path,
        transform: Optional[Callable[[Image], torch.Tensor]] = None,
    ) -> None:
        """
        Set up the dataset loader.

        Args:
            dirpath: Path to your main dataset folder (the one with subfolders for each class)
            transform: Any image transformations you want to apply (resize, normalize, etc.)

        Your directory needs to have subfolders - each subfolder name becomes a class label.
        """
        # Make sure the directory actually exists
        if not dirpath.is_dir():
            logger.error(f"Can't find directory: {dirpath}")
            raise ValueError(f"Directory {dirpath} doesn't exist or isn't accessible")

        # Make sure it's not empty
        if not any(dirpath.iterdir()):
            logger.error(f"Directory {dirpath} is empty - no class folders found")
            raise ValueError(f"Directory {dirpath} is empty - need subfolders for each class")

        # Store the transform function
        self.transform: Optional[Callable[[Image], torch.Tensor]] = transform

        # Use PyTorch's ImageFolder to do the heavy lifting
        self.dataset: ImageFolder = ImageFolder(root=str(dirpath))

        logger.info(f"Dataset loaded: {len(self.dataset)} images from {dirpath}")

    def __len__(self) -> int:
        """
        How many images are in the dataset?

        Returns:
            Total number of images we found
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Union[Tuple[Image, int], Tuple[None, None]]:
        """
        Get a specific image and its class label.

        This tries to load the image at the given index. If that image is
        corrupted or can't be read, it'll try the next one, and so on.
        If we run out of images, it returns None.

        Args:
            idx: Which image to get (starting from 0)

        Returns:
            A tuple of (image, class_number) if successful, or (None, None) if failed
        """
        # Keep trying until we get a good image or run out
        while True:
            try:
                # Try to load the image and its label
                image: Union[Image, torch.Tensor]
                label: int
                image, label = self.dataset[idx]

                # Make sure we have a PIL Image (some datasets return tensors)
                if not isinstance(image, Image):
                    image = transforms.ToPILImage()(image)

                # Apply any transforms if we have them
                if self.transform:
                    image = self.transform(image)

                return image, label

            except (UnidentifiedImageError, OSError) as error:
                # This image is corrupted or unreadable
                logger.error(f"Couldn't load image at index {idx}: {error}")
                warnings.warn(f"Skipping corrupted image at index {idx}: {str(error)}")

                # Try the next image
                idx += 1

                # Don't go on forever - if we've tried everything, give up
                if idx >= len(self):
                    logger.error("Ran out of images to try - they might all be corrupted")
                    warnings.warn("No valid images found - check your dataset for corruption")
                    return None, None