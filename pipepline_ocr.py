from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
import yaml
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
from src.bhu_text_mining.network_analysis import (
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
image_path = "/home/kjeong/localgit/bhu_text_mining/data"
sample_image_file = image_path + "/fig01.jpg"
prefix = "cropped"
yaml_path = "/home/kjeong/localgit/bhu_text_mining/data/text_data.yaml"

# If you have scanned images from students' documents, please use the
# following Image OCR pipeline to extract text and build a network graph
# based on the extracted keywords.

## Image OCR process
# Crop your image
crop_coordinates = show_image(sample_image_file)
crop_and_save_images(image_path, crop_coordinates, prefix)

# Text OCR and extraction
image_files = prepare_image_files_list(image_path, prefix)
secret_key, api_url = load_naver_ocr_env(config_path)
request_json = create_request_json(image_files)
response_data = send_images_receive_ocr(api_url, secret_key, image_files)
extracted_data = extract_text(response_data)

print(extracted_data)  # Just for check

# If your data is in YAML format, you can save it as a YAML file, please us the

with open(yaml_path, "r", encoding="UTF-8") as f:
    extracted_data = yaml.safe_load(f)

type(extracted_data)  # Just for check
print(extracted_data)

## Network analysis
# Load stopwords data
stopwords = load_stopwords(stopwords_path)

# Build the keywords networks
extracted_keywords_dict = {}
for filename, text in extracted_data.items():
    filtered_keywords = extract_keywords(text, stopwords)
    extracted_keywords_dict[filename] = filtered_keywords

print(extracted_keywords_dict)

network_graph = build_network_from_keywords(
    extracted_keywords_dict, "professor", 5
)

# Korean font setting on the network node figure
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
plt.rcParams["axes.unicode_minus"] = False

# Draw network graphs
visualize_network(
    network_graph, font_prop, title="Keywords Network from Dictionary"
)
