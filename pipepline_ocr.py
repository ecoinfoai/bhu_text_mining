from src.bhu_text_mining.preprocess_imgs import (
    show_image,
    crop_and_save_images,
)
from src.bhu_text_mining.naver_ocr import (
    load_naver_ocr_env,
    create_request_json,
    prepare_image_files_list,
    send_images_receive_ocr,
    extract_text,
)


# Setup parameters
config_path = "~/.config/naver_ocr/naver_ocr_config.json"
image_path = "/home/kjeong/localgit/bhu_text_mining/data"
sample_image_file = image_path + "/fig01.jpg"
prefix = "cropped_"

# Crop your image
crop_coordinates = show_image(sample_image_file)
crop_and_save_images(image_path, crop_coordinates, prefix)


# Initial settings
image_files = prepare_image_files_list(image_path, prefix)
secret_key, api_url = load_naver_ocr_env(config_path)
request_json = create_request_json(image_files)
response_data = send_images_receive_ocr(api_url, secret_key, image_files)
extracted_data = extract_text(response_data)

print(extracted_data)
