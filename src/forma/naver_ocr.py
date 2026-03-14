"""Naver CLOVA OCR API client for scanned answer sheet text extraction."""

from __future__ import annotations

import os
import requests
import uuid
import time
import json
import base64


def load_naver_ocr_env(config_path: str = "") -> tuple[str, str]:
    """
    Settings for Naver OCR API. Load the secret key and API URL from the
    configuration file.

    If ``config_path`` is provided and exists, it is used directly
    (legacy behaviour).  Otherwise falls back to the unified config
    system (``~/.config/formative-analysis/forma.json`` → legacy paths).

    Args:
        config_path (str): Path to the configuration file (optional).

    Returns:
        tuple[str, str]:
            - secret_key (str): Secret key for Naver OCR API.
            - api_url (str): API URL for Naver OCR API.

    Examples:
        >>> secret_key, api_url = load_naver_ocr_env("path/to/config_file")
        >>> print(secret_key)
        >>> print(api_url)
    """
    # Try explicit path first (legacy behaviour)
    if config_path:
        expanded_path = os.path.expanduser(config_path)
        if os.path.isfile(expanded_path):
            with open(expanded_path, "r") as f:
                config = json.load(f)
            return config["secret_key"], config["api_url"]

    # Fall back to unified config system
    from forma.config import get_naver_ocr_config, load_config

    config = load_config(config_path if config_path else None)
    return get_naver_ocr_config(config)


def create_request_json(image_files: list[str]) -> dict:
    """
    Create a JSON payload for an OCR API request. This function generates
    a JSON object required for an OCR API request. It processes a list of
    image file paths and includes information such as file format, file name,
    a unique request ID, and a timestamp.

    Args:
        image_files (list[str]): A list of file paths to the images.

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


def prepare_image_files_list(image_path: str, prefix: str) -> list[str]:
    """
    Get a list of image files with a specific prefix in the given directory.

    Only includes files with extensions .jpg, .jpeg, or .png that start with
    the specified prefix. Returns full paths to the matching image files.

    Args:
        image_path (str): Path to the directory containing image files.
        prefix (str): Prefix to filter image files. Default is "cropped_".

    Returns:
        list: A list of image file paths with the specified prefix.

    Example:
        >>> image_path = "/home/user/images"
        >>> get_cropped_images(image_path, prefix="cropped_")
        ["/home/user/images/cropped_image1.jpg", "/home/user/images/cropped_image2.png"]
    """
    image_file = []
    for file_name in os.listdir(image_path):
        if file_name.startswith(prefix) and file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            full_path = os.path.join(image_path, file_name)
            image_file.append(full_path)

    return image_file


def send_images_receive_ocr(
    api_url: str, secret_key: str, image_files: list[str]
) -> list[dict]:
    """
    Send images to the OCR API and receive OCR results.

    This function processes a list of image file paths, encodes each image
    in Base64 format, and sends it to the OCR API. It returns the OCR
    results for all the images as a list of dictionaries.

    Args:
        api_url (str): The URL of the OCR API.
        secret_key (str): The secret key for authenticating with the OCR API.
        image_files (list[str]): A list of file paths to the images to be processed.

    Returns:
        list[Dict]: A list of dictionaries containing the OCR results for
        each image. Each dictionary corresponds to the API's response for
        a single image.

    Raises:
        ValueError: If api_url does not use HTTPS.

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
    # FR-010: HTTPS enforcement
    if not api_url.startswith("https://"):
        raise ValueError("OCR API URL must use HTTPS")

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


def extract_text_with_confidence(responses: list[dict]) -> dict[str, dict]:
    """Extract text and confidence statistics from OCR API responses.

    Processes OCR API responses, extracting recognized text and
    ``inferConfidence`` values for each image.  When confidence data is
    available, mean and minimum confidence are computed; otherwise they
    are returned as ``None``.

    Args:
        responses: A list of dictionaries containing OCR API responses.
            Each dictionary should include an ``"images"`` key, where each
            image entry contains ``"name"`` (image name) and ``"fields"``
            (recognized text fields with optional ``"inferConfidence"``).

    Returns:
        A dictionary keyed by image name.  Each value is a dict with:
            - ``text`` (str): Aggregated recognized text.
            - ``confidence_mean`` (float | None): Mean of available
              ``inferConfidence`` values, or ``None`` if none present.
            - ``confidence_min`` (float | None): Minimum ``inferConfidence``,
              or ``None`` if none present.
            - ``field_count`` (int): Number of text fields in the image.
    """
    extracted_data: dict[str, dict] = {}

    for response_data in responses:
        for image_result in response_data.get("images", []):
            image_name = image_result["name"]
            fields = image_result.get("fields", [])

            extracted_texts = [
                field.get("inferText", "").replace("\n", " ").strip()
                for field in fields
            ]
            aggregated_text = " ".join(extracted_texts).strip()

            confidences = [
                field.get("inferConfidence")
                for field in fields
                if field.get("inferConfidence") is not None
            ]

            if confidences:
                confidence_mean = sum(confidences) / len(confidences)
                confidence_min = min(confidences)
            else:
                confidence_mean = None
                confidence_min = None

            extracted_data[image_name] = {
                "text": aggregated_text,
                "confidence_mean": confidence_mean,
                "confidence_min": confidence_min,
                "field_count": len(fields),
            }

    return extracted_data


def extract_text(responses: list[dict]) -> dict[str, str]:
    """Extract and aggregate text from OCR API responses.

    Convenience wrapper around :func:`extract_text_with_confidence` that
    returns only the recognized text strings, preserving the original
    return type for backward compatibility.

    Args:
        responses: A list of dictionaries containing OCR API responses.
            Each dictionary should include a ``"images"`` key, where each
            image entry contains ``"name"`` (image name) and ``"fields"``
            (recognized text fields).

    Returns:
        A dictionary where keys are image file names and values are
        aggregated text recognized from the respective images.

    Examples:
        >>> result = extract_text(responses)
        >>> print(result)
        {
            "image1.jpg": "Hello World",
            "image2.jpg": "Python OCR"
        }
    """
    full_data = extract_text_with_confidence(responses)
    return {name: entry["text"] for name, entry in full_data.items()}
