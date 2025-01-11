from src.bhu_text_mining.ocr import show_image, crop_and_save_images
import matplotlib

# Verify backend changes
print("Current backend:", matplotlib.get_backend())

# Setup parameters
image_path = "/home/kjeong/localgit/bhu_text_mining/data"
sample_image_file = image_path + "/fig01.jpg"

# Crop your image
crop_coordinates = show_image(sample_image_file)
crop_and_save_images(image_path, crop_coordinates, "cropped")


# Initial settings
config_path = "~/.config/naver_ocr/naver_ocr_config.json"
image_file = [
    "/home/kjeong/localgit/bhu_text_mining/data/fig01.jpg",
    "/home/kjeong/localgit/bhu_text_mining/data/fig02.jpg",
]

secret_key, api_url = load_naver_ocr_env(config_path)

request_json = create_request_json(image_file)
print(request_json)

response_data = send_images_receive_ocr(api_url, secret_key, image_file)
print(response_data)

extracted_data = extract_text(response_data)
print(extracted_data)
