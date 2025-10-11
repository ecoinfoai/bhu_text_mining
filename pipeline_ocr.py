from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
import yaml
from src.preprocess_imgs import (
    show_image,
    crop_and_save_images,
)
from src.naver_ocr import (
    load_naver_ocr_env,
    create_request_json,
    prepare_image_files_list,
    send_images_receive_ocr,
    extract_text,
)
from src.network_analysis import (
    load_stopwords,
    extract_keywords,
    build_network_from_keywords,
    visualize_network,
)


## Setup parameters

# Analysis environment setup
config_path = "~/.config/naver_ocr/naver_ocr_config.json"
font_path = "/usr/share/fonts/nanum/NanumGothic.ttf"
stopwords_path = "./data/stopwords-ko.txt"

# Data for the analysis
image_path = "/home/kjeong/localgit/bhu_text_mining/data/W6_1A"
image_file = image_path + "/W6_1A_0001.jpg"
prefix = "cropped"
yaml_path = "/home/kjeong/localgit/bhu_text_mining/data/text_data.yaml"

# If you have scanned images from students' documents, please use the
# following Image OCR pipeline to extract text and build a network graph
# based on the extracted keywords.

## OCR your Images

# Crop your image
crop_coordinates = show_image(image_file)
crop_and_save_images(image_path, crop_coordinates, prefix)

# Text OCR and extraction
image_files = prepare_image_files_list(image_path, prefix)
secret_key, api_url = load_naver_ocr_env(config_path)
request_json = create_request_json(image_files)
response_data = send_images_receive_ocr(api_url, secret_key, image_files)
extracted_data = extract_text(response_data)

print(extracted_data)  # Just for check
