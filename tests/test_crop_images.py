import os
from PIL import Image
import pandas as pd
import numpy as np
import pytest
from src.bhu_text_mining.preprocess_imgs import (
    crop_and_save_images,
)


@pytest.fixture
def setup_test_images(tmp_path):
    """
    Pytest fixture to create a temporary directory with test images.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # Create a sample image
    img1 = Image.new("RGB", (500, 500), color="red")
    img1.save(image_dir / "test_image1.jpg")

    img2 = Image.new("RGB", (600, 600), color="blue")
    img2.save(image_dir / "test_image2.png")

    return image_dir


def test_crop_and_save_images_valid(setup_test_images):
    """
    Test cropping and saving images with valid inputs.
    """
    image_dir = setup_test_images
    crop_coordinates = (50, 50, 300, 300)
    output_prefix = "cropped"

    crop_and_save_images(str(image_dir), crop_coordinates, output_prefix)

    # Check that cropped images are saved
    saved_files = list(image_dir.glob("cropped_*.*"))
    assert len(saved_files) == 2  # Two images should be processed

    # Check that the cropped images have correct dimensions
    for cropped_file in saved_files:
        img = Image.open(cropped_file)
        assert img.size == (250, 250)  # width=right-left, height=lower-upper


def test_crop_and_save_images_empty_directory(tmp_path):
    """
    Test the function when the input directory is empty.
    """
    image_dir = tmp_path / "empty_images"
    image_dir.mkdir()

    crop_coordinates = (50, 50, 300, 300)
    output_prefix = "cropped"

    crop_and_save_images(str(image_dir), crop_coordinates, output_prefix)

    # No files should be saved
    saved_files = list(image_dir.glob("cropped_*.jpg"))
    assert len(saved_files) == 0


def test_crop_and_save_images_invalid_coordinates(setup_test_images):
    """
    Test the function with invalid crop coordinates.
    """
    image_dir = setup_test_images
    crop_coordinates = (600, 600, 300, 300)  # Invalid coordinates
    output_prefix = "cropped"

    with pytest.raises(ValueError):
        crop_and_save_images(str(image_dir), crop_coordinates, output_prefix)


def test_crop_and_save_images_unsupported_extension(setup_test_images):
    """
    Test that unsupported file extensions are ignored.
    """
    image_dir = setup_test_images

    # Create an unsupported file format (e.g., .bmp)
    img_unsupported = Image.new("RGB", (500, 500), color="green")
    img_unsupported.save(image_dir / "unsupported_image.bmp")

    crop_coordinates = (50, 50, 300, 300)
    output_prefix = "cropped"

    crop_and_save_images(str(image_dir), crop_coordinates, output_prefix)

    # Check that no cropped file is created for the unsupported format
    saved_files = list(image_dir.glob("cropped_*.*"))  # Include all formats
    unsupported_files = [f for f in saved_files if f.name.endswith(".bmp")]

    assert (
        len(unsupported_files) == 0
    ), "Unsupported file format was processed!"


def test_crop_and_save_images_output_naming(setup_test_images):
    """
    Test the naming convention of the output files.
    """
    image_dir = setup_test_images
    crop_coordinates = (50, 50, 300, 300)
    output_prefix = "cropped"

    crop_and_save_images(str(image_dir), crop_coordinates, output_prefix)

    # Check the naming of the saved files
    saved_files = list(image_dir.glob("cropped_*.jpg"))
    for saved_file in saved_files:
        assert output_prefix in saved_file.name
