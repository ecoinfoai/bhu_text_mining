import os
import json
import pytest
from unittest.mock import patch, mock_open
from src.bhu_text_mining.naver_ocr import (
    load_naver_ocr_env,
    create_request_json,
    prepare_image_files_list,
    send_images_receive_ocr,
    extract_text,
)


@pytest.fixture
def mock_image_directory(tmpdir):
    """Create a temporary directory with mock image files."""
    image_dir = tmpdir.mkdir("images")
    # Create files with and without the 'cropped_' prefix
    cropped_files = [
        "cropped_image1.jpg",
        "cropped_image2.png",
        "cropped_image3.jpeg",
    ]
    other_files = ["image4.jpg", "image5.png", "cropped_text.txt"]
    for file_name in cropped_files + other_files:
        image_dir.join(file_name).write("")  # Create empty files
    return image_dir


def test_load_naver_ocr_env():
    # Mock configuration file content
    mock_config = {
        "secret_key": "mock_secret_key",
        "api_url": "https://mock.api.url",
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
        secret_key, api_url = load_naver_ocr_env("mock/config/path")

    assert secret_key == "mock_secret_key"
    assert api_url == "https://mock.api.url"


def test_create_request_json():
    image_files = ["/path/to/image1.jpg", "/path/to/image2.png"]
    result = create_request_json(image_files)

    assert "images" in result
    assert "requestId" in result
    assert "version" in result
    assert "timestamp" in result

    assert result["images"] == [
        {"format": "jpg", "name": "image1"},
        {"format": "png", "name": "image2"},
    ]
    assert result["version"] == "V2"


def test_send_images_receive_ocr():
    api_url = "https://mock.api.url"
    secret_key = "mock_secret_key"
    image_files = ["/path/to/image1.jpg"]

    mock_response = {
        "images": [
            {
                "name": "image1.jpg",
                "fields": [{"inferText": "Mock Text"}],
            }
        ]
    }

    with patch("builtins.open", mock_open(read_data=b"mock_image_data")):
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.status_code = 200
            with patch(
                "os.path.exists", return_value=True
            ):  # os.path.exists 패치
                results = send_images_receive_ocr(
                    api_url, secret_key, image_files
                )

    assert len(results) == 1
    assert results[0] == mock_response


def test_send_images_receive_ocr_file_not_found():
    api_url = "https://mock.api.url"
    secret_key = "mock_secret_key"
    image_files = ["/path/to/nonexistent_image.jpg"]

    with pytest.raises(FileNotFoundError):
        send_images_receive_ocr(api_url, secret_key, image_files)


def test_extract_text():
    mock_responses = [
        {
            "images": [
                {
                    "name": "image1.jpg",
                    "fields": [
                        {"inferText": "Hello"},
                        {"inferText": "World"},
                    ],
                },
                {
                    "name": "image2.jpg",
                    "fields": [
                        {"inferText": "Python"},
                        {"inferText": "OCR"},
                    ],
                },
            ]
        }
    ]

    result = extract_text(mock_responses)

    assert result == {
        "image1.jpg": "Hello World",
        "image2.jpg": "Python OCR",
    }


def test_extract_text_empty_fields():
    mock_responses = [
        {
            "images": [
                {
                    "name": "image1.jpg",
                    "fields": [],
                }
            ]
        }
    ]

    result = extract_text(mock_responses)

    assert result == {"image1.jpg": ""}


def test_prepare_image_files_list(mock_image_directory):
    """Test the prepare_image_files_list function."""
    image_path = str(mock_image_directory)
    result = prepare_image_files_list(image_path, prefix="cropped_")
    assert (
        len(result) == 3
    ), "The number of filtered files does not match the expected count."

    for file_path in result:
        assert os.path.basename(file_path).startswith(
            "cropped_"
        ), "File does not have the required prefix."
        assert file_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        ), "File does not have a valid extension."

    for file_path in result:
        assert os.path.isfile(
            file_path
        ), f"Path {file_path} is not a valid file."


def test_invalid_directory():
    """Test handling of invalid directory."""
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        prepare_image_files_list("/invalid/path", prefix="cropped")


def test_empty_directory(tmpdir):
    """Test function behavior with an empty directory."""
    empty_dir = tmpdir.mkdir("empty_images")
    result = prepare_image_files_list(str(empty_dir), prefix="cropped_")
    assert (
        result == []
    ), "The result should be an empty list for an empty directory."


if __name__ == "__main__":
    pytest.main()
