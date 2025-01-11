from PIL import Image
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")

coordinates = []

def onclick(event):
    """
    Handles mouse click events to record coordinates of the clicked points.

    This function captures the x and y coordinates of the mouse click within the 
    displayed image and stores them in a global list `coordinates`. The function 
    expects two clicks (top-left and bottom-right corners of the crop area). Once 
    both coordinates are recorded, it disconnects the event listener and closes 
    the image window.

    Args:
        event (matplotlib.backend_bases.MouseEvent): The mouse event object containing
            the coordinates and metadata of the click event.

    Returns:
        None
    """
    if event.xdata and event.ydata:
        coordinates.append((int(event.xdata), int(event.ydata)))
        print(f"The clicked coordinates: ({int(event.xdata)}, {int(event.ydata)})")

        if len(coordinates) == 2:  # 두 개의 좌표를 얻으면 이벤트 연결 해제
            print("Confirmed coordinates:", coordinates)
            plt.gcf().canvas.mpl_disconnect(cid)  # 이벤트 연결 해제
            plt.close()  # 창 닫기

def show_image(image_path):
    """
    Displays an image in a separate window for user interaction.

    Opens the specified image file, displays it in a new window, and allows 
    users to click on points to define a crop area. The function does not 
    handle click events directly but works with an event listener 
    (e.g., `onclick`) to process mouse interactions.

    Args:
        image_path (str): The file path of the image to be displayed.

    Returns:
        None
    """
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title("Click on the top left and bottom right corners")
    plt.axis("on")

def crop_and_save_images(
    image_dir: str,
    crop_coordinates: tuple[int, int, int, int],
    output_prefix: str
):
    supported_formats = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(supported_formats):
           image_path = os.path.join(image_dir, filename)
           img = Image.open(image_path)
           cropped_img = img.crop(crop_coordinates)
           timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
           output_filename = f"{output_prefix}_{timestamp}_{filename}"
           output_path = os.path.join(image_dir, output_filename)
           cropped_img.save(output_path)
           print(f"Saved the cropped image to: {output_path}")


# 백엔드 변경 확인
print("Current backend:", matplotlib.get_backend())

# 이미지 경로
image_path = "/home/kjeong/localgit/bhu_text_mining/data/fig01.jpg"

# 클릭 이벤트 연결
show_image(image_path)
cid = plt.gcf().canvas.mpl_connect("button_press_event", onclick)
plt.show()








def crop_and_save_images(image):
    
