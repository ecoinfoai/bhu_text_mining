from konlpy.tag import Okt
from collections import Counter
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Dict


def load_stopwords(file_path: str) -> set:
    with open(file_path, "r", encoding="UTF-8") as f:
        stopwords = f.read().splitlines()

    return set(stopwords)


def extract_keywords(text: str, stopwords: set) -> List[str]:
    okt = Okt()
    nouns = okt.nouns(text)

    extracted_keywords = []
    for word in nouns:
        if word not in stopwords and len(word) > 1:
            extracted_keywords.append(word)

    return extracted_keywords


def create_network(keywords: List[str], window_size=2) -> nx.Graph:
    edges = Counter()
    for i in range(len(keywords) - window_size + 1):
        window = keywords[i : i + window_size]
        edges.update(itertools.combinations(window, 2))

    G = nx.Graph()
    for (word1, word2), weight in edges.items():
        G.add_edge(word1, word2, weight=weight)

    return G


def build_network_from_keywords(
    keywords_dict: Dict[str, List], target_filename: str, window_size=2
) -> nx.Graph:
    keywords = keywords_dict[target_filename]
    return create_network(keywords, window_size=window_size)


def visualize_network(G, font_prop, title="Keyword Network"):
    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    weights = nx.get_edge_attributes(G, "weight").values()  # Get edges weights
    max_weight = max(weights) if weights else 1  # for weights normalization
    edge_widths = [
        3 * (weight / max_weight) for weight in weights
    ]  # thickness of edges

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        font_size=10,
        edge_color="gray",
        font_family=font_prop.get_name(),
        width=edge_widths,
    )

    # labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title(title, fontproperties=font_prop)
    plt.show()
