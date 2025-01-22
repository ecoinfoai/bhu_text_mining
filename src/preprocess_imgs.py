from PIL import Image
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")


def show_image(image_path):
    """
    Displays an image and captures crop coordinates.

    Opens the specified image file in a new window, allows the user to click on
    two points (top-left and bottom-right corners of the crop area), and returns
    the coordinates as a tuple.

    Args:
        image_path (str): The file path of the image to be displayed.

    Returns:
        Tuple[int, int, int, int]: The crop coordinates (left, upper, right, lower).

    Example:
        >>> image_path = "example.jpg"
        >>> crop_coords = show_image(image_path)
        Click on the top left corner: (50, 100)
        Click on the bottom right corner: (400, 300)
        Confirmed coordinates: [(50, 100), (400, 300)]
        >>> print(crop_coords)
        (50, 100, 400, 300)
    """
    # Local variable to store coordinates
    coordinates = []

    def onclick(event):
        """
        Handles mouse click events to record coordinates of the clicked points.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The mouse event object containing
                the coordinates and metadata of the click event.
        """
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
):
    """
    Crops and saves multiple images from a specified directory.

    Iterates through all images in the given directory, crops each image using
    the provided coordinates, and saves the cropped images with a new filename.

    Args:
        image_dir (str): Path to the directory containing images to process.
        crop_coordinates (tuple[int, int, int, int]): Coordinates for cropping
            in the format (left, upper, right, lower).
        output_prefix (str): Prefix for the output file names.

    Example:
        >>> image_dir = "images/"
        >>> crop_coordinates = (50, 100, 400, 300)
        >>> output_prefix = "cropped"
        >>> crop_and_save_images(image_dir, crop_coordinates, output_prefix)
        Saved the cropped image to: images/cropped_20250111123045_image1.jpg
        Saved the cropped image to: images/cropped_20250111123046_image2.jpg
    """

    supported_formats = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(image_dir, filename)
            img = Image.open(image_path)
            cropped_img = img.crop(crop_coordinates)
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_extension = os.path.splitext(filename)[-1]
            output_filename = f"{output_prefix}_{timestamp}_{filename.split('.')[0]}{file_extension}"
            output_path = os.path.join(image_dir, output_filename)
            cropped_img.save(output_path)
            print(f"Saved the cropped image to: {output_path}")
