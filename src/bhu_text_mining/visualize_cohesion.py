import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm
from collections import defaultdict
from math import ceil
from src.bhu_text_mining.cohesion_analysis import (
    extract_keywords_from_sentences,
)
from src.bhu_text_mining.network_analysis import (
    create_network,
    visualize_network,
)


def compare_topic_lengths_with_xy(topic_df: pd.DataFrame):
    """
    Compare topic-wise sentence count and average length per person,
    and plot XY graphs for each Person in groups of 6.

    Args:
        topic_df (pd.DataFrame): DataFrame containing topic assignments, sentences, and person identifiers.

    Returns:
        pd.DataFrame: A summary table of topic-wise statistics per person.
    """
    # Initialize data structure to store stats
    topic_stats = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "total_length": 0})
    )

    # Calculate counts and character lengths
    for _, row in topic_df.iterrows():
        topic_no = row["Topic No."]
        person = row["Person"]
        char_count = len(row["Sentence"])

        topic_stats[person][topic_no]["count"] += 1
        topic_stats[person][topic_no]["total_length"] += char_count

    # Convert stats to a DataFrame for easier analysis
    rows = []
    for person, topics in topic_stats.items():
        for topic_no, stats in topics.items():
            avg_length = (
                stats["total_length"] / stats["count"]
                if stats["count"] > 0
                else 0
            )
            rows.append(
                {
                    "Person": person,
                    "Topic No.": topic_no,
                    "Sentence Count": stats["count"],
                    "Avg Sentence Length": avg_length,
                }
            )
    summary_df = pd.DataFrame(rows)

    # Get unique persons
    unique_persons = summary_df["Person"].unique()

    # Determine global axis limits
    x_min = summary_df["Sentence Count"].min()
    x_max = summary_df["Sentence Count"].max()
    y_min = summary_df["Avg Sentence Length"].min()
    y_max = summary_df["Avg Sentence Length"].max()

    # Plot XY graphs in groups of 6
    group_size = 6
    num_groups = ceil(len(unique_persons) / group_size)

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min(start_idx + group_size, len(unique_persons))
        persons_in_group = unique_persons[start_idx:end_idx]

        # Create a 3x2 subplot grid
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        axes = axes.flatten()

        for i, person in enumerate(persons_in_group):
            ax = axes[i]
            person_data = summary_df[summary_df["Person"] == person]

            # Scatter plot for the current person
            scatter = ax.scatter(
                person_data["Sentence Count"],
                person_data["Avg Sentence Length"],
                label=person,
            )

            # Annotate points with Topic No.
            for _, row in person_data.iterrows():
                ax.annotate(
                    str(row["Topic No."]),
                    (row["Sentence Count"], row["Avg Sentence Length"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="center",
                    fontsize=9,
                )

            ax.set_title(f"Person: {person}")
            ax.set_xlabel("Sentence Count")
            ax.set_ylabel("Avg Sentence Length")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True)

        # Remove unused subplots
        for j in range(len(persons_in_group), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    return summary_df


def generate_person_networks_from_sentences(
    topic_df: pd.DataFrame, stopwords: set, font_prop, window_size=2
):
    """
    Generate and visualize network graphs for each person based on their sentences.

    Args:
        topic_df (pd.DataFrame): DataFrame with columns ['Person', 'Sentence'].
        stopwords (set): Set of stopwords to exclude.
        font_prop: FontProperties object for Korean font rendering.
        window_size (int): Window size for creating edges in the network graph.

    Returns:
        None: Displays network graphs for each person.
    """
    # Extract keywords grouped by person
    keywords_dict = extract_keywords_from_sentences(topic_df, stopwords)

    # Generate and visualize network graphs
    for person, keywords in keywords_dict.items():
        G = create_network(keywords, window_size=window_size)
        visualize_network(G, font_prop, title=f"Keyword Network for {person}")
