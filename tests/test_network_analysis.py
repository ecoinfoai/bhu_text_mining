import pytest
import networkx as nx
from src.bhu_text_mining.network_analysis import (
    load_stopwords,
    extract_keywords,
    create_network,
    build_network_from_keywords,
)


@pytest.fixture
def sample_stopwords_file(tmp_path):
    stopwords_content = "이\n그\n저\n그리고\n그래서\n"
    stopwords_file = tmp_path / "stopwords.txt"
    stopwords_file.write_text(stopwords_content, encoding="utf-8")
    return str(stopwords_file)


@pytest.fixture
def sample_text():
    return "이 프로그램은 Python을 사용하여 키워드 네트워크를 생성합니다."


@pytest.fixture
def stopwords():
    return {"이", "그", "저", "그리고", "그래서"}


def test_load_stopwords(sample_stopwords_file):
    stopwords = load_stopwords(sample_stopwords_file)
    assert "이" in stopwords
    assert "Python" not in stopwords
    assert isinstance(stopwords, set)


def test_extract_keywords(sample_text, stopwords):
    keywords = extract_keywords(sample_text, stopwords)
    assert "프로그램" in keywords
    assert "이" not in keywords
    assert all(len(word) > 1 for word in keywords)


def test_create_network():
    keywords = ["Python", "키워드", "네트워크", "생성", "Python"]
    G = create_network(keywords, window_size=2)
    assert isinstance(G, nx.Graph)
    assert ("Python", "키워드") in G.edges
    assert G["Python"]["키워드"]["weight"] == 1
    assert G["키워드"]["네트워크"]["weight"] == 1


def test_build_network_from_keywords():
    keywords_dict = {
        "sample.txt": ["Python", "키워드", "네트워크", "생성", "Python"]
    }
    G = build_network_from_keywords(keywords_dict, "sample.txt", window_size=2)
    assert isinstance(G, nx.Graph)
    assert ("Python", "키워드") in G.edges
    assert G["Python"]["키워드"]["weight"] == 1
    assert G["키워드"]["네트워크"]["weight"] == 1
