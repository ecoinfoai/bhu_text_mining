import pytest
import yaml
from bertopic import BERTopic
import pandas as pd
from unittest.mock import MagicMock
from src.bhu_text_mining.topic_analysis import (
    load_yaml_data,
    split_sentences,
    generate_topic_dataframe,
    generate_topic_keywords_table,
)


# Fixture: Sample YAML data
@pytest.fixture
def sample_yaml_data():
    return {
        "professor": "This is the professor's answer. It has two sentences.",
        "student1": "Student1's answer is here. It has three sentences. Here's another one.",
    }


# Fixture: Temporary YAML file
@pytest.fixture
def yaml_file(tmp_path, sample_yaml_data):
    file_path = tmp_path / "sample.yaml"
    with open(file_path, "w", encoding="UTF-8") as f:
        yaml.dump(sample_yaml_data, f, allow_unicode=True)
    return file_path


# Fixture: BERTopic model
@pytest.fixture
def topic_model():
    # Mocking BERTopic for testing
    mock_model = MagicMock(spec=BERTopic)
    mock_model.get_topic_info.return_value = pd.DataFrame(
        {
            "Topic": [0, 1],
            "Count": [10, 15],
            "Name": ["Topic 0", "Topic 1"],
        }
    )
    mock_model.get_topic.side_effect = lambda topic_id: [
        ("keyword1", 0.5),
        ("keyword2", 0.3),
        ("keyword3", 0.2),
    ]
    return mock_model


# Test: load_yaml_data
def test_load_yaml_data(yaml_file, sample_yaml_data):
    data = load_yaml_data(yaml_file)
    assert data == sample_yaml_data


# Test: split_sentences
def test_split_sentences(sample_yaml_data):
    sentences, keys = split_sentences(sample_yaml_data)
    assert len(sentences) == 5
    assert sentences[0] == "This is the professor's answer."
    assert keys == [
        "professor",
        "professor",
        "student1",
        "student1",
        "student1",
    ]


# Test: generate_topic_dataframe
def test_generate_topic_dataframe():
    sentences = [
        "Sentence 1 from professor.",
        "Sentence 2 from professor.",
        "Sentence 1 from student1.",
    ]
    topics = [0, 1, 0]
    keys = ["professor", "professor", "student1"]
    df = generate_topic_dataframe(sentences, topics, keys)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.iloc[0]["Person"] == "professor"
    assert df.iloc[0]["Topic No."] == 0
    assert df.iloc[0]["Sentence No."] == "professor_0000"
    assert df.iloc[0]["Sentence"] == "Sentence 1 from professor."


# Test: generate_topic_keywords_table
def test_generate_topic_keywords_table(topic_model):
    sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
    topics = [0, 1, 0]
    df = generate_topic_keywords_table(topic_model, sentences, topics)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.iloc[0]["Topic"] == 0
    assert df.iloc[0]["Keywords"] == "keyword1, keyword2, keyword3"
