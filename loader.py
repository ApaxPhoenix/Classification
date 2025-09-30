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
