import os
import requests
import uuid
import time
import json
from typing import Dict, List, Tuple
import base64


def load_naver_ocr_env(config_path: str) -> Tuple[str, str]:
    """
    Settings for Naver OCR API. Load the secret key and API URL from the
    configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Tuple[str, str]:
            - secret_key (str): Secret key for Naver OCR API.
            - api_url (str): API URL for Naver OCR API.

    Examples:
        >>> secret_key, api_url = load_naver_ocr_env("path/to/config_file")
        >>> print(secret_key)
        >>> print(api_url)
    """

    expanded_path = os.path.expanduser(config_path)
    with open(expanded_path, "r") as f:
        config = json.load(f)
    secret_key = config["secret_key"]
    api_url = config["api_url"]

    return secret_key, api_url


def create_request_json(image_files: List[str]) -> Dict:
    """
    Create a JSON payload for an OCR API request. This function generates
    a JSON object required for an OCR API request. It processes a list of
    image file paths and includes information such as file format, file name,
    a unique request ID, and a timestamp.

    Args:
        image_files (List[str]): A list of file paths to the images.

    Returns:
        Dict: A JSON object containing the API request details, including:
            - "images": List of dictionaries, each with:
                - "format": The file format (e.g., "jpg", "png").
                - "name": The file name without the extension.
            - "requestId": A unique identifier for the request.
            - "version": The API version (fixed as "V2").
            - "timestamp": The current timestamp in milliseconds.

    Examples:
        >>> image_files = ["path/to/image1.jpg", "path/to/image2.png"]
        >>> request_json = create_request_json(image_files)
        >>> print(request_json)
        {
            "images": [
                {"format": "jpg", "name": "image1"},
                {"format": "png", "name": "image2"}
            ],
            "requestId": "123e4567-e89b-12d3-a456-426614174000",
            "version": "V2",
            "timestamp": 1736513806003
        }
    """
    return {
        "images": [
            {
                "format": os.path.splitext(os.path.basename(image_file))[1][
                    1:
                ],
                "name": os.path.splitext(os.path.basename(image_file))[0],
            }
            for image_file in image_files
        ],
        "requestId": str(uuid.uuid4()),
        "version": "V2",
        "timestamp": int(round(time.time() * 1000)),
    }


def send_images_receive_ocr(
    api_url: str, secret_key: str, image_files: List[str]
) -> List[Dict]:
    """
    Send images to the OCR API and receive OCR results.

    This function processes a list of image file paths, encodes each image
    in Base64 format, and sends it to the OCR API. It returns the OCR
    results for all the images as a list of dictionaries.

    Args:
        api_url (str): The URL of the OCR API.
        secret_key (str): The secret key for authenticating with the OCR API.
        image_files (List[str]): A list of file paths to the images to be processed.

    Returns:
        List[Dict]: A list of dictionaries containing the OCR results for
        each image. Each dictionary corresponds to the API's response for
        a single image.

    Examples:
        >>> api_url = "https://ocr.api.example.com/v1/process"
        >>> secret_key = "your_secret_key"
        >>> image_files = ["path/to/image1.jpg", "path/to/image2.png"]
        >>> results = send_images_receive_ocr(api_url, secret_key, image_files)
        >>> for result in results:
        ...     print(result)
        [
            {
                "images": [
                    {"name": "image1.jpg", "text": "Recognized text here"}
                ],
                "requestId": "123e4567-e89b-12d3-a456-426614174000"
            },
            {
                "images": [
                    {"name": "image2.png", "text": "Another recognized text"}
                ],
                "requestId": "123e4567-e89b-12d3-a456-426614174001"
            }
        ]
    """
    results = []

    for image_file in image_files:
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"File not found: {image_file}")

        with open(image_file, "rb") as file:
            file_content = file.read()
            file_data_base64 = base64.b64encode(file_content).decode("utf-8")

        request_json = {
            "images": [
                {
                    "format": os.path.splitext(image_file)[1][1:],  # 확장자
                    "name": os.path.basename(image_file),  # 파일 이름
                    "data": file_data_base64,  # Base64 데이터
                }
            ],
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": int(round(time.time() * 1000)),
        }

        headers = {"X-OCR-SECRET": secret_key}
        response = requests.post(
            api_url,
            headers=headers,
            json=request_json,  # JSON 데이터를 직접 전송
        )
        response.raise_for_status()
        results.append(response.json())

    return results


def extract_text(responses: List[Dict]) -> Dict[str, str]:
    """
    Extract and aggregate text from OCR API responses.

    This function processes the responses from an OCR API, extracting recognized
    text for each image and aggregating the text into a single string for each image.

    Args:
        responses (List[Dict]): A list of dictionaries containing OCR API responses.
            Each dictionary should include a "images" key, where each image entry
            contains "name" (image name) and "fields" (recognized text fields).

    Returns:
        Dict[str, str]: A dictionary where keys are image file names and values
        are aggregated text recognized from the respective images.

    Examples:
        >>> result = extract_text(responses)
        >>> print(result)
        {
            "image1.jpg": "Hello World",
            "image2.jpg": "Python OCR"
        }
    """
    extracted_data = {}

    for response_data in responses:
        for image_result in response_data.get("images", []):
            image_name = image_result["name"]
            extracted_texts = [
                field.get("inferText", "").replace("\n", " ").strip()
                for field in image_result.get("fields", [])
            ]
            aggregated_text = " ".join(extracted_texts).strip()
            extracted_data[image_name] = aggregated_text

    return extracted_data


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
