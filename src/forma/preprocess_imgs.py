"""Image preprocessing for scanned answer sheets (crop, split, rename)."""

from __future__ import annotations

from typing import Any

from PIL import Image
import os
import re


def show_image(image_path: str) -> tuple[int, int, int, int]:
    """Display an image and capture crop coordinates via mouse clicks.

    Opens the specified image in a matplotlib window and allows the user
    to click two points (top-left and bottom-right corners of the crop area).

    Args:
        image_path: Path to the image file to display.

    Returns:
        Crop coordinates as (left, upper, right, lower).

    Raises:
        ImportError: If Qt5 backend is not available.
        ValueError: If two points were not selected.
    """
    import matplotlib

    try:
        matplotlib.use("Qt5Agg")
        import matplotlib.pyplot as plt
        plt.figure()  # 백엔드 실제 로드 확인
        plt.close()
    except ImportError:
        raise ImportError(
            "Cannot load Qt5 backend. "
            "NixOS: add python3Packages.pyqt5 to nix-shell. "
            "Other: pip install PyQt5"
        ) from None

    coordinates: list[tuple[int, int]] = []

    def onclick(event: Any) -> None:
        """Handle mouse click events to record crop coordinates."""
        if event.xdata and event.ydata:
            coordinates.append((int(event.xdata), int(event.ydata)))
            print(
                f"The clicked coordinates: ({int(event.xdata)}, {int(event.ydata)})"
            )

            if len(coordinates) == 2:  # Two points captured, disconnect event
                print("Confirmed coordinates:", coordinates)
                plt.gcf().canvas.mpl_disconnect(
                    cid
                )  # Disconnect the event listener
                plt.close()  # Close the image window

    # Open and display the image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title("Click on the top left and bottom right corners")
    plt.axis("on")

    # Connect the click event
    cid = plt.gcf().canvas.mpl_connect("button_press_event", onclick)

    # Display the image and wait for interaction
    plt.show()

    # Ensure exactly two coordinates are captured
    if len(coordinates) == 2:
        left, upper = coordinates[0]
        right, lower = coordinates[1]
        return left, upper, right, lower
    else:
        raise ValueError("Two points were not selected.")


def crop_and_save_images(
    image_dir: str,
    crop_coordinates: tuple[int, int, int, int],
    output_prefix: str,
) -> None:
    """Crop and save all images in a directory using the given coordinates.

    Iterates through supported image files (jpg, jpeg, png), crops each
    using the provided coordinates, and saves with the given prefix.
    Already-cropped files (starting with ``q<N>_``) are skipped.

    Args:
        image_dir: Path to the directory containing images.
        crop_coordinates: Crop region as (left, upper, right, lower).
        output_prefix: Prefix for the output filenames.
    """

    supported_formats = (".jpg", ".jpeg", ".png")
    _cropped_re = re.compile(r"^q\d+_")

    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(supported_formats) and not _cropped_re.match(filename):
            image_path = os.path.join(image_dir, filename)
            img = Image.open(image_path)
            cropped_img = img.crop(crop_coordinates)
            ## timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_extension = os.path.splitext(filename)[-1]
            output_filename = f"{output_prefix}_{filename.split('.')[0]}{file_extension}"
            ## output_filename = f"{output_prefix}_{timestamp}_{filename.split('.')[0]}{file_extension}"
            output_path = os.path.join(image_dir, output_filename)
            cropped_img.save(output_path)
            print(f"Saved the cropped image to: {output_path}")
