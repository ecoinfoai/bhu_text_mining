import pytest
import networkx as nx
from matplotlib.font_manager import FontProperties
from src.network_analysis import (
    load_stopwords,
    extract_keywords,
    create_network,
    build_network_from_keywords,
    visualize_network,
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


@pytest.fixture
def sample_graph():
    # 샘플 그래프 생성
    G = nx.Graph()
    G.add_edge("키워드1", "키워드2", weight=2)
    G.add_edge("키워드2", "키워드3", weight=3)
    G.add_edge("키워드3", "키워드4", weight=1)
    G.nodes["키워드1"]["frequency"] = 5
    G.nodes["키워드2"]["frequency"] = 10
    G.nodes["키워드3"]["frequency"] = 7
    G.nodes["키워드4"]["frequency"] = 3
    return G


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


def test_node_sizes(sample_graph):
    frequencies = nx.get_node_attributes(sample_graph, "frequency")
    max_frequency = max(frequencies.values())
    node_sizes = [
        1000 * (freq / max_frequency) for freq in frequencies.values()
    ]

    assert len(node_sizes) == len(frequencies)
    assert max(node_sizes) == 1000
    assert min(node_sizes) > 0


def test_font_sizes(sample_graph):
    frequencies = nx.get_node_attributes(sample_graph, "frequency")
    max_frequency = max(frequencies.values())
    font_sizes = [
        10 + 15 * (freq / max_frequency) for freq in frequencies.values()
    ]

    assert len(font_sizes) == len(frequencies)
    assert max(font_sizes) > 10
    assert min(font_sizes) >= 10


def test_edge_widths(sample_graph):
    weights = nx.get_edge_attributes(sample_graph, "weight").values()
    max_weight = max(weights)
    edge_widths = [3 * (weight / max_weight) for weight in weights]

    assert len(edge_widths) == len(weights)
    assert max(edge_widths) == 3
    assert min(edge_widths) > 0


def test_visualize_network(sample_graph):
    font_path = "/path/to/font.ttf"
    font_prop = FontProperties(fname=font_path)

    try:
        visualize_network(sample_graph, font_prop, title="Test Network")
    except Exception as e:
        pytest.fail(f"Visualize network function failed: {e}")
