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


def create_network(keywords: List[str], window_size=2):
    edges = Counter()
    keyword_counts = Counter(keywords)

    for i in range(len(keywords) - window_size + 1):
        window = keywords[i : i + window_size]
        edges.update(itertools.combinations(window, 2))

    G = nx.Graph()
    for (word1, word2), weight in edges.items():
        G.add_edge(word1, word2, weight=weight)

    for word, count in keyword_counts.items():
        G.nodes[word]["frequency"] = count

    return G


def build_network_from_keywords(
    keywords_dict: Dict[str, List], target_filename: str, window_size=2
) -> nx.Graph:

    keywords = keywords_dict[target_filename]
    return create_network(keywords, window_size=window_size)


def visualize_network(G, font_prop, title="Keyword Network"):
    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    # Extract frequencies for node size
    frequencies = nx.get_node_attributes(G, "frequency")
    max_frequency = max(frequencies.values()) if frequencies else 1

    # Node sizes proportional to frequency
    node_sizes = [
        1000 * (freq / max_frequency) for freq in frequencies.values()
    ]

    # Edge widths proportional to weight
    weights = nx.get_edge_attributes(G, "weight").values()
    max_weight = max(weights) if weights else 1
    edge_widths = [3 * (weight / max_weight) for weight in weights]

    # Draw nodes and edges
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.9
    )
    nx.draw_networkx_edges(
        G, pos, width=edge_widths, edge_color="gray", alpha=0.5
    )

    # Draw labels with individual font sizes
    for node, (x, y) in pos.items():
        freq = frequencies[node]
        font_size = 10 + 15 * (
            freq / max_frequency
        )  # Font size proportional to frequency
        plt.text(
            x,
            y,
            s=node,
            fontsize=font_size,
            fontproperties=font_prop,
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.title(title, fontproperties=font_prop)
    plt.axis("off")
    plt.show()
