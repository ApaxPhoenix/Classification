import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Union, cast
from pathlib import Path
import warnings
import logging
import numpy as np

# Use the proper logging configuration from main
logger = logging.getLogger("loader")


def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    """A collation function for semantic segmentation dataset batching.

    Implements robust batch processing by filtering failed data loads and stacking
    valid tensors for PyTorch DataLoader compatibility. Handles real-world datasets
    containing corrupted files or malformed annotations.

    Args:
        batch: Collection of (image, mask) tuples returned from dataset __getitem__
            method. May contain None values from failed loads.

    Returns:
        Tuple containing (image, mask) as stacked tensors with
        batch dimension as first axis. Returns None if entire batch fails validation.

    Note:
        Batch dimension ordering follows PyTorch convention: (N, C, H, W) for images
        and (N, H, W) for segmentation masks where N is batch size.
    """
    # Filter None values from corrupted files or parsing failures
    batch: List[Tuple[torch.Tensor, torch.Tensor]] = [item for item in batch if item is not None]

    # Handle complete batch failure to prevent downstream training errors
    if not batch:
        warnings.warn(message="All items in batch failed to load")
        logger.warning("Complete batch failure: All items in batch failed to load")
        return None

    # Separate image and mask tensors for independent stacking
    images: Tuple[torch.Tensor, ...]
    masks: Tuple[torch.Tensor, ...]
    images, masks = zip(*batch)

    # Stack tensors along batch dimension for model compatibility
    return torch.stack(tensors=images), torch.stack(tensors=masks)


class DatasetLoader(Dataset):
    """PyTorch Dataset for semantic segmentation with XML polygon annotations.

    Provides standardized interface for loading image-annotation pairs where
    segmentation masks are generated from polygon coordinates in XML format.
    Designed for computer vision training pipelines requiring pixel-level
    classification.

    The dataset expects structured directory layout with separate subdirectories
    for images and XML annotation files. Polygon coordinates are converted to
    dense segmentation masks during loading.

    Directory Structure:
        root_directory/
        ├── images/          # Image files (.jpg, .jpeg, .png, .tiff)
        └── annotations/     # XML annotation files with matching base names

    XML Annotation Format:
        Each XML file contains <object> elements with <name> tags for class
        labels and <polygon> elements with coordinate pairs as separate tags.

    Attributes:
        images (Path): Directory path containing training images.
        annotations (Path): Directory path containing XML annotation files.
        transform (transforms.Compose): Optional image preprocessing pipeline.
        files (List[Tuple[Path, Path]]): Matched image-annotation file pairs.
        classes (Dict[str, int]): Mapping from class names to integer labels.
    """

    def __init__(
            self, dirpath: Path, transform: Optional[transforms.Compose] = None
    ) -> None:
        """Initialize dataset by discovering and validating image-annotation pairs.

        Performs comprehensive dataset validation including directory structure
        verification, file matching, and class vocabulary construction from
        all available annotations.

        Args:
            dirpath: Root directory containing required 'images' and 'annotations'
                subdirectories following standard dataset organization.
            transform: Optional torchvision transform pipeline for image preprocessing
                including augmentation and normalization operations.

        Raises:
            UserWarning: When required directories are missing or no valid pairs found.
        """
        # Establish paths to required dataset subdirectories
        self.images: Path = Path(dirpath, "images")
        self.annotations: Path = Path(dirpath, "annotations")
        self.transform: Optional[transforms.Compose] = transform

        # Validate dataset directory structure and initialize empty collections
        if not self.images.exists():
            warnings.warn(message=f"Images directory not found at {self.images}")
            logger.warning(f"Dataset initialization failed: Images directory not found at {self.images}")
            self.files: List[Tuple[Path, Path]] = []
            self.classes: Dict[str, int] = {}
            return

        if not self.annotations.exists():
            warnings.warn(
                message=f"Annotations directory not found at {self.annotations}"
            )
            logger.warning(f"Dataset initialization failed: Annotations directory not found at {self.annotations}")
            self.files: List[Tuple[Path, Path]] = []
            self.classes: Dict[str, int] = {}
            return

        # Discover matching image-annotation pairs through filename matching
        self.files: List[Tuple[Path, Path]] = [
            (image, annotation)
            for annotation in self.annotations.glob(pattern="*.xml")
            for pattern in ["*.jpg", "*.jpeg", "*.png", "*.tiff"]
            for image in self.images.glob(pattern=pattern)
            if image.stem == annotation.stem
        ]

        # Construct class vocabulary by parsing all XML annotations using cast
        self.classes: Dict[str, int] = {
            name: label
            for label, name in enumerate(
                dict.fromkeys(
                    cast(ET.Element, object.find(path="name")).text
                    for _, annotation in self.files
                    for object in ET.parse(source=annotation)
                    .getroot()
                    .findall(path="object")
                )
            )
        }

        # Validate dataset completeness and log initialization results
        if not self.files:
            warnings.warn(
                message="No matching image-annotation pairs found. Check file names and extensions."
            )
            logger.warning("Dataset initialization incomplete: No matching image-annotation pairs found. Check file names and extensions.")
            return

        logger.info(f"Found {len(self.files)} valid image-annotation pairs")
        logger.info(f"Found {len(self.classes)} classes: {self.classes}")

    def __len__(self) -> int:
        """Return total number of available dataset samples.

        Required implementation for PyTorch Dataset interface enabling
        proper DataLoader iteration and sampling strategies.

        Returns:
            Integer count of valid image-annotation pairs in dataset.
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor], None]:
        """Load and process single dataset sample with comprehensive error handling.

        Implements complete data loading pipeline including image loading, XML parsing,
        polygon-to-mask conversion, and optional preprocessing. Handles multiple objects
        per image with proper mask overlay semantics for dense segmentation tasks.

        Args:
            index: Zero-based dataset index for sample retrieval.

        Returns:
            Tuple of (preprocessed_image, segmentation_mask) as PyTorch tensors
            with appropriate dtypes for training. Returns None on processing failure
            to enable graceful error handling in batch processing.

        Processing Pipeline:
            1. Load source image and extract spatial dimensions
            2. Parse XML annotation for polygon coordinates and class labels
            3. Convert polygon coordinates to dense segmentation mask
            4. Apply optional image transformations with mask resizing
            5. Convert to PyTorch tensors with proper dtype conventions
        """
        # Validate index bounds to prevent array access errors
        if index >= len(self.files):
            logger.error(f"Index out of bounds: {index} >= {len(self.files)}")
            return None

        # Retrieve file paths for current dataset sample
        image: Path
        annotation: Path
        image, annotation = self.files[index]

        try:
            # Load source image and capture original spatial dimensions
            image: Image.Image = Image.open(fp=image)
            width: int
            height: int
            width, height = image.size

            # Parse XML annotation structure for object definitions
            tree: ET.ElementTree = ET.parse(source=annotation)
            root: ET.Element = tree.getroot()

            # Initialize segmentation mask canvas with background class (0)
            mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
            canvas: Image.Image = Image.fromarray(mask)
            draw: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)

            # Process all annotated objects in current sample
            objects: List[ET.Element] = root.findall(path="object")

            # Extract class labels using cast
            labels: List[int] = [
                self.classes[cast(ET.Element, object.find(path="name")).text]
                for object in objects
            ]

            # Extract polygon coordinates using cast
            polygons: List[List[Tuple[float, float]]] = [
                [(float(cast(ET.Element, polygon)[i].text),
                  float(cast(ET.Element, polygon)[i + 1].text))
                 for i in range(0, len(cast(ET.Element, polygon)), 2)]
                for polygon in [cast(ET.Element, object.find("polygon")) for object in objects]
            ]

            # Rasterize polygons to segmentation mask with class labels
            for label, points in zip(labels, polygons):
                draw.polygon(points, fill=label)

            # Convert PIL canvas back to numpy array for tensor processing
            mask = np.array(canvas)

            # Apply image preprocessing pipeline if specified
            if self.transform:
                image: Tensor = self.transform(image)

                # Determine target dimensions from transform pipeline
                mHeight: int = height
                mWidth: int = width
                for transform in self.transform.transforms:
                    if hasattr(transform, 'size'):
                        mHeight, mWidth = transform.size
                        break

                # Resize segmentation mask using nearest neighbor interpolation
                scaley: np.ndarray = np.divide(mHeight, height)
                scalex: np.ndarray = np.divide(mWidth, width)

                # Generate coordinate mappings for mask resizing
                yindex: np.ndarray = np.arange(mHeight)
                xindex: np.ndarray = np.arange(mWidth)

                # Map target coordinates back to source image space
                origy: np.ndarray = np.divide(yindex, scaley).astype(np.int32)
                origx: np.ndarray = np.divide(xindex, scalex).astype(np.int32)

                # Clamp coordinates to valid source image bounds
                origy = np.clip(origy, 0, np.subtract(height, 1))
                origx = np.clip(origx, 0, np.subtract(width, 1))

                # Apply coordinate mapping to resize mask
                ygrid: np.ndarray
                xgrid: np.ndarray
                ygrid, xgrid = np.meshgrid(origy, origx, indexing='ij')
                mask = mask[ygrid, xgrid]
            else:
                # Apply default tensor conversion for images without preprocessing
                image: Tensor = transforms.ToTensor()(image)

            # Convert segmentation mask to PyTorch tensor with long dtype
            mask: torch.Tensor = torch.from_numpy(mask).long()

            return image, mask

        except Exception as error:
            # Log processing failures for debugging and monitoring
            logger.error(f"Failed to load item {index} ({image}): {error}")
            warnings.warn(f"Data loading failed for item {index}: {str(error)}")
            return None