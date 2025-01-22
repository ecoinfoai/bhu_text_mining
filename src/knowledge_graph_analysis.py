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
):

    all_keywords = []
    for sentence in topic_df["Sentence"]:
        keywords = extract_keywords(sentence, stopwords)
        all_keywords.extend(keywords)

    graph = create_network(all_keywords, window_size=window_size)
    return graph


def create_student_knowledge_graph(
    student_name: str, topic_df: pd.DataFrame, stopwords: set, window_size=2
):
    student_df = topic_df[topic_df["Person"] == student_name]
    return build_knowledge_graph(student_df, stopwords, window_size)


def create_reference_knowledge_graph(
    topic_df: pd.DataFrame, stopwords: set, window_size=2
):
    reference_df = topic_df[topic_df["Person"] == "professor"]
    return build_knowledge_graph(reference_df, stopwords, window_size)


def compare_graphs(graph1: nx.Graph, graph2: nx.Graph) -> Dict[str, List]:
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
