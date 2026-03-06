from konlpy.tag import Okt
from collections import Counter
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict


def load_stopwords(file_path: str) -> set:
    """
    Load stopwords from a text file into a set.

    Reads a text file containing stopwords (one per line) and returns them
    as a set for efficient filtering during text processing.

    Args:
        file_path (str): The path to the file containing stopwords.

    Returns:
        set: A set of stopwords.

    Examples:
        >>> stopwords = load_stopwords("stopwords.txt")
        >>> "and" in stopwords
        True
    """
    with open(file_path, "r", encoding="UTF-8") as f:
        stopwords = f.read().splitlines()

    return set(stopwords)


def extract_keywords(text: str, stopwords: set) -> List[str]:
    """
    Extract keywords from text, excluding stopwords.

    Uses the Okt tokenizer to extract nouns from the input text and filters
    out words that are either in the stopwords set or have a length of one character.

    Args:
        text (str): The input text to process.
        stopwords (set): A set of words to exclude from the results.

    Returns:
        List[str]: A list of extracted keywords.

    Examples:
        >>> text = "Python은 데이터 분석에 매우 유용한 언어입니다."
        >>> stopwords = {"은", "에", "언어"}
        >>> extract_keywords(text, stopwords)
        ['Python', '데이터', '분석']
    """
    okt = Okt()
    nouns = okt.nouns(text)

    extracted_keywords = []
    for word in nouns:
        if word not in stopwords and len(word) > 1:
            extracted_keywords.append(word)

    return extracted_keywords


def create_network(keywords: List[str], window_size=2):
    """
    Create a keyword co-occurrence network.

    Constructs a graph where nodes represent keywords and edges represent
    co-occurrences within a sliding window of the specified size. Node
    attributes include keyword frequencies, and edge weights reflect
    co-occurrence counts.

    Args:
        keywords (List[str]): A list of keywords to process.
        window_size (int): The size of the sliding window for co-occurrence
            calculations (default is 2).

    Returns:
        nx.Graph: A network graph with keywords as nodes and co-occurrence
        relationships as edges.

    Examples:
        >>> keywords = ["Python", "data", "analysis", "Python", "analysis"]
        >>> G = create_network(keywords, window_size=2)
        >>> G.nodes["Python"]["frequency"]
        2
        >>> G["Python"]["analysis"]["weight"]
        1
    """
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
    """
    Build a network graph from preprocessed keywords for a specific target.

    Fetches keywords for the specified filename from the input dictionary and
    constructs a co-occurrence network using the given window size.

    Args:
        keywords_dict (Dict[str, List]): A dictionary where keys are filenames
            and values are lists of keywords.
        target_filename (str): The filename whose keywords are used to build the network.
        window_size (int): The size of the sliding window for co-occurrence
            calculations (default is 2).

    Returns:
        nx.Graph: A network graph of keywords for the specified target file.

    Examples:
        >>> keywords_dict = {"file1.txt": ["Python", "data", "analysis"]}
        >>> G = build_network_from_keywords(keywords_dict, "file1.txt")
        >>> G.nodes["Python"]["frequency"]
        1
    """
    keywords = keywords_dict[target_filename]
    return create_network(keywords, window_size=window_size)


def visualize_network(G, font_prop, title="Keyword Network"):
    """
    Visualize a keyword co-occurrence network.

    Displays a network graph with nodes sized proportionally to keyword
    frequencies and edges weighted by co-occurrence strength. Node labels
    are displayed with font sizes proportional to their frequency.

    Args:
        G (nx.Graph): The network graph to visualize.
        font_prop: Font properties for rendering node labels.
        title (str): The title of the plot (default is "Keyword Network").

    Examples:
        >>> G = create_network(["Python", "data", "analysis", "Python"], window_size=2)
        >>> font_prop = FontProperties(fname="path/to/font.ttf")
        >>> visualize_network(G, font_prop)
    """
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
