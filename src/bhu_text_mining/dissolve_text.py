import yaml
from typing import List, Dict, Union, Set
from pathlib import Path


def load_yaml(file_path: Union[str, Path]) -> List[Dict]:

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def clean_data(
    raw_data: List[Dict], required_fields: List[str]
) -> List[Dict[str, str]]:

    cleaned_data = []
    for record in raw_data:
        record_fields = set(record.keys())
        required_set = set(required_fields)
        missing = required_set - record_fields

        if missing:
            print(
                f"Warning: Missing fields {list(missing)} for record {record.get('id', 'Unknown')}"
            )
            continue

        cleaned_record = {}
        for field in required_fields:
            value = record[field]
            if isinstance(value, (int, float)):
                value = str(value)
            elif value is None:
                value = ""
            if isinstance(value, str):
                value = value.strip()
            cleaned_record[field] = value
        cleaned_data.append(cleaned_record)

    return cleaned_data


def get_text_for_analysis(
    cleaned_data: List[Dict[str, str]],
    fields: List[str] = None,
    exclude_fields: Set[str] = {"id", "name", "gender", "age", "dept"},
) -> List[str]:
    """
    지정된 필드의 텍스트를 추출하여 KoNLPy 분석용 리스트로 변환

    Args:
        data: process_student_data 함수로 처리된 데이터
        fields: 분석할 필드명 리스트. None인 경우 텍스트성 필드 전체 사용
        exclude_fields: 분석에서 제외할 필드 집합

    Returns:
        List[str]: 분석할 텍스트 리스트
    """
    if not cleaned_data:
        return []

    # fields가 지정되지 않은 경우, 텍스트성 필드 전체 사용
    if fields is None:
        fields = [
            field
            for field in cleaned_data[0].keys()
            if field not in exclude_fields
        ]

    texts = []
    for student in cleaned_data:
        # 지정된 필드의 텍스트를 모두 이어붙임
        student_text = " ".join(student[field] for field in fields)
        texts.append(student_text)

    return texts


file_path = "data/self_introduction_2024.yaml"
required_fields = [
    "dept",
    "id",
    "name",
    "gender",
    "age",
    "growth",
    "character",
    "pros_cons",
    "values",
    "school_life",
    "career",
    "motivation",
]

textdata = load_yaml(file_path)
cleaned_data = clean_data(textdata, required_fields)
texts = get_text_for_analysis(cleaned_data)

len(cleaned_data)
type(cleaned_data)

texts
