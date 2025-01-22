import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from src.network_analysis import (
    create_network,
    extract_keywords,
)


# Develop knowledge graph
def build_knowledge_graph(
    topic_df: pd.DataFrame, stopwords: set, window_size=2
) -> nx.Graph:
    """

    Build a knowledge graph from text data using keyword co-occurrence.

    Creates a network graph where nodes are keywords and edges represent
    co-occurrence within the window size.

    Args:
       topic_df (pd.DataFrame): DataFrame containing 'Sentence' column.
       stopwords (set): Set of words to exclude from keyword extraction.
       window_size (int, optional): Size of co-occurrence window. Defaults to 2.

    Returns:
       nx.Graph: NetworkX graph with keywords as nodes and co-occurrences as edges.
    """
    all_keywords = []
    for sentence in topic_df["Sentence"]:
        keywords = extract_keywords(sentence, stopwords)
        all_keywords.extend(keywords)

    graph = create_network(all_keywords, window_size=window_size)
    return graph


def create_student_knowledge_graph(
    student_name: str, topic_df: pd.DataFrame, stopwords: set, window_size=2
) -> nx.Graph:
    """
    Create a knowledge graph from a specific student's text data.

    Filters data for given student and builds their knowledge graph.

    Args:
        student_name (str): Name of student to filter data for.
        topic_df (pd.DataFrame): DataFrame with 'Person' and 'Sentence' columns.
        stopwords (set): Set of words to exclude from keyword extraction.
        window_size (int, optional): Size of co-occurrence window. Defaults to 2.

    Returns:
        nx.Graph: NetworkX graph representing student's knowledge structure.
    """
    student_df = topic_df[topic_df["Person"] == student_name]
    return build_knowledge_graph(student_df, stopwords, window_size)


def create_reference_knowledge_graph(
    topic_df: pd.DataFrame, stopwords: set, window_size=2
) -> nx.Graph:
    """
    Create a reference knowledge graph from professor's text data.

    Filters for professor entries and builds a reference knowledge graph.

    Args:
        topic_df (pd.DataFrame): DataFrame with 'Person' and 'Sentence' columns.
        stopwords (set): Set of words to exclude from keyword extraction.
        window_size (int, optional): Size of co-occurrence window. Defaults to 2.

    Returns:
        nx.Graph: NetworkX graph representing reference knowledge structure.
    """
    reference_df = topic_df[topic_df["Person"] == "professor"]
    return build_knowledge_graph(reference_df, stopwords, window_size)


def compare_graphs(graph1: nx.Graph, graph2: nx.Graph) -> Dict[str, List]:
    """
    Compare two knowledge graphs and identify differences.

    Finds missing/extra nodes and edges between two graphs.

    Args:
        graph1 (nx.Graph): First graph to compare.
        graph2 (nx.Graph): Second graph to compare (reference).

    Returns:
        Dict[str, List]: Dictionary containing:
            - missing_nodes: Nodes in graph2 but not in graph1
            - extra_nodes: Nodes in graph1 but not in graph2
            - missing_edges: Edges in graph2 but not in graph1
            - extra_edges: Edges in graph1 but not in graph2
    """
    results = {
        "missing_nodes": list(set(graph2.nodes()) - set(graph1.nodes())),
        "extra_nodes": list(set(graph1.nodes()) - set(graph2.nodes())),
        "missing_edges": list(set(graph2.edges()) - set(graph1.edges())),
        "extra_edges": list(set(graph1.edges()) - set(graph2.edges())),
    }
    return results


# Tabulation of comparison results
def display_comparison_results(
    comparison_results: Dict[str, Dict[str, List]]
) -> pd.DataFrame:
    """
    Create a summary DataFrame of knowledge graph comparisons.

    Takes a dictionary of comparison results between reference and student knowledge
    graphs and transforms it into a DataFrame. For each student, combines node
    differences as comma-separated strings and counts edge differences. Useful for
    identifying key structural differences in students' knowledge representation.

    Args:
        comparison_results (Dict[str, Dict[str, List]]): Dictionary containing
            comparison results for each student, with missing/extra nodes and edges.

    Returns:
        pd.DataFrame: DataFrame with columns for student name, missing/extra nodes
            (as comma-separated strings), and missing/extra edge counts.

    Examples:
        >>> results = {
        ...     'student1': {
        ...         'missing_nodes': ['concept1', 'concept2'],
        ...         'extra_nodes': ['concept3'],
        ...         'missing_edges': [('concept1', 'concept2')],
        ...         'extra_edges': []
        ...     }
        ... }
        >>> display_comparison_results(results)
           Student         Missing Nodes Extra Nodes  Missing Edges  Extra Edges
        0  student1  concept1, concept2    concept3             1            0
    """
    all_data = []
    for student, results in comparison_results.items():
        all_data.append(
            {
                "Student": student,
                "Missing Nodes": ", ".join(results["missing_nodes"]),
                "Extra Nodes": ", ".join(results["extra_nodes"]),
                "Missing Edges": len(results["missing_edges"]),
                "Extra Edges": len(results["extra_edges"]),
            }
        )
    results_df = pd.DataFrame(all_data)

    return results_df


# Visualization of superimposed graph
def visualize_superimposed_graph(
    reference_graph: nx.Graph,
    student_graph: nx.Graph,
    font_prop,
    title="Superimposed Graph Comparison",
):
    """
    Create visualization comparing reference and student knowledge graphs.

    Displays nodes and edges from both graphs with color coding:
    - Reference: light gray
    - Student: blue
    - Missing elements: red
    - Extra elements: green

    Args:
        reference_graph (nx.Graph): Reference knowledge graph (professor's)
        student_graph (nx.Graph): Student's knowledge graph
        font_prop: Font properties for text rendering
        title (str, optional): Plot title. Defaults to "Superimposed Graph Comparison"
    """
    plt.figure(figsize=(12, 10))

    # Create commone layouts (including all nodes)
    combined_nodes = set(reference_graph.nodes()).union(
        set(student_graph.nodes())
    )
    combined_graph = nx.Graph()
    combined_graph.add_nodes_from(combined_nodes)
    pos = nx.random_layout(combined_graph)
    # pos = nx.spring_layout(combined_graph, seed=42)

    # Reference Graph
    nx.draw_networkx_nodes(
        reference_graph,
        pos,
        node_color="lightgray",
        node_size=700,
        alpha=0.6,
        label="Reference Nodes",
    )
    nx.draw_networkx_edges(
        reference_graph,
        pos,
        edge_color="lightgray",
        alpha=0.6,
        label="Reference Edges",
    )

    # Student Graph
    nx.draw_networkx_nodes(
        student_graph,
        pos,
        node_color="blue",
        node_size=700,
        alpha=0.9,
        label="Student Nodes",
    )
    nx.draw_networkx_edges(
        student_graph, pos, edge_color="blue", alpha=0.8, label="Student Edges"
    )

    # Emphasizing the NODE difference (Extra/Missing Nodes)
    missing_nodes = set(reference_graph.nodes()) - set(student_graph.nodes())
    extra_nodes = set(student_graph.nodes()) - set(reference_graph.nodes())
    nx.draw_networkx_nodes(
        combined_graph,
        pos,
        nodelist=list(missing_nodes),
        node_color="red",
        node_size=800,
        alpha=0.9,
        label="Missing Nodes",
    )
    nx.draw_networkx_nodes(
        combined_graph,
        pos,
        nodelist=list(extra_nodes),
        node_color="green",
        node_size=800,
        alpha=0.9,
        label="Extra Nodes",
    )

    # Emphasizing the EDGE diffference (Extra/Missing Edges)
    missing_edges = set(reference_graph.edges()) - set(student_graph.edges())
    extra_edges = set(student_graph.edges()) - set(reference_graph.edges())
    nx.draw_networkx_edges(
        combined_graph,
        pos,
        edgelist=list(missing_edges),
        edge_color="red",
        width=2,
        alpha=0.8,
        label="Missing Edges",
    )
    nx.draw_networkx_edges(
        combined_graph,
        pos,
        edgelist=list(extra_edges),
        edge_color="green",
        width=2,
        alpha=0.8,
        label="Extra Edges",
    )

    # Label the nodes
    nx.draw_networkx_labels(
        combined_graph, pos, font_family=font_prop.get_name()
    )

    plt.title(title, fontproperties=font_prop)
    plt.legend()
    plt.axis("off")
    plt.show()
