import yaml
from typing import Dict, List, Tuple
from bertopic import BERTopic
from konlpy.tag import Okt, Mecab
from collections import Counter
import itertools
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import kss
import pandas as pd
from bert_score import score
from matplotlib import font_manager as fm
from collections import defaultdict
from math import ceil
from src.bhu_text_mining.network_analysis import (
    load_stopwords,
    extract_keywords,
    create_network,
    visualize_network,
)


# It is strongly recommended to install Mecab on your linux environment,
# before using kss.split_sentences. The kss library prefers to use
# Mecab for Korean sentence splitting.


# Prepare your data
def load_yaml_data(yaml_path: str) -> Dict[str, str]:
    with open(yaml_path, "r", encoding="UTF-8") as f:
        return yaml.safe_load(f)


def split_sentences(yaml_data: Dict[str, str]) -> List[str]:
    all_sentences = []
    keys = []
    for key, text in yaml_data.items():
        sentences = kss.split_sentences(text)
        all_sentences.extend(sentences)
        keys.extend([key] * len(sentences))
    return all_sentences, keys


# BERTopic model
def configure_bertopic(env_config: Dict):
    umap_model = UMAP(**env_config["umap"])
    hdbscan_model = HDBSCAN(**env_config["hdbscan"])
    embedding_model = SentenceTransformer(env_config["embedding_model"])

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
    )


def analyze_topics_with_bertopic(topic_model: BERTopic, sentences: List[str]):
    topics, probs = topic_model.fit_transform(sentences)
    return topics, probs


# Examine topics from BERTopic model
def generate_topic_dataframe(
    sentences: List[str], topics: List[int], keys: List[str]
) -> pd.DataFrame:
    data = []
    for i, (sentence, topic, key) in enumerate(zip(sentences, topics, keys)):
        data.append(
            {
                "Person": key,
                "Topic No.": topic,
                "Sentence No.": f"{key}_{i:04}",
                "Sentence": sentence,
            }
        )
    df = pd.DataFrame(data)

    return df


def generate_topic_keywords_table(
    topic_model: BERTopic, sentences: List[str], topics: List[int]
):
    topic_keywords = topic_model.get_topic_info()
    topic_keywords_table = []

    for topic_id in topic_keywords["Topic"]:
        if topic_id == -1:
            continue  # Ignore outliers
        keywords = topic_model.get_topic(topic_id)
        topic_keywords_table.append(
            {
                "Topic": topic_id,
                "Keywords": ", ".join([word[0] for word in keywords]),
            }
        )

    df = pd.DataFrame(topic_keywords_table)
    return df


# Examine texts similarity using BERTscore
def calculate_bertscore(
    reference: List[str], candidate: List[str], lang="ko"
) -> Tuple[float, float, float]:
    P, R, F1 = score(candidate, reference, lang=lang)
    return P.mean().item(), R.mean().item(), F1.mean().item()


def calculate_overall_similarity(
    professor_text: List[str], student_texts: List[List[str]], lang="ko"
) -> pd.DataFrame:
    """
    Calculate BERTScore for professor's full text compared to each student's full text.

    Args:
        professor_text (List[str]): List of sentences from the professor's answers.
        student_texts (List[List[str]]): List of lists, where each inner list contains sentences from a student's answers.
        lang (str): Language code for BERTScore.

    Returns:
        pd.DataFrame: A DataFrame with overall BERTScore results for each student.
    """
    results = []
    professor_full_text = " ".join(
        professor_text
    )  # Join professor's text into a single string
    for i, student_text in enumerate(student_texts):
        student_full_text = " ".join(
            student_text
        )  # Join student's text into a single string

        # Calculate BERTScore
        P, R, F1 = calculate_bertscore(
            reference=[professor_full_text],
            candidate=[student_full_text],
            lang=lang,
        )
        results.append(
            {
                "Person": f"Student{i + 1}",
                "Precision": P,
                "Recall": R,
                "F1": F1,
            }
        )

    return pd.DataFrame(results)


def calculate_topicwise_similarity(
    topic_df: pd.DataFrame, lang="ko"
) -> pd.DataFrame:
    """
    Calculate BERTScore for each topic comparing professor's text and students' texts.

    Args:
        topic_df (pd.DataFrame): DataFrame with columns ['Person', 'Topic No.', 'Sentence'].
        lang (str): Language code for BERTScore.

    Returns:
        pd.DataFrame: Topic-wise BERTScore results sorted by student and topic.
    """
    results = []

    # Get unique topics and persons, including -1
    topics = sorted(topic_df["Topic No."].unique())
    persons = sorted(topic_df["Person"].unique())

    for person in persons:
        if person == "professor":
            continue  # Skip professor's data

        for topic in topics:
            # Professor's text for the topic
            professor_topic_text = " ".join(
                topic_df[
                    (topic_df["Person"] == "professor")
                    & (topic_df["Topic No."] == topic)
                ]["Sentence"]
            )

            # Student's text for the topic
            student_topic_text = " ".join(
                topic_df[
                    (topic_df["Person"] == person)
                    & (topic_df["Topic No."] == topic)
                ]["Sentence"]
            )

            # If either professor or student text is empty, skip
            if (
                not professor_topic_text.strip()
                or not student_topic_text.strip()
            ):
                continue

            # Calculate BERTScore
            P, R, F1 = score(
                [student_topic_text], [professor_topic_text], lang=lang
            )

            # Store results
            results.append(
                {
                    "Person": person,
                    "Topic No.": topic,
                    "Precision": P.mean().item(),
                    "Recall": R.mean().item(),
                    "F1": F1.mean().item(),
                }
            )

    # Create DataFrame and sort by Person and Topic No.
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by=["Person", "Topic No."]).reset_index(
        drop=True
    )

    return result_df


def compare_students_overall(
    student_texts: List[List[str]], lang="ko"
) -> pd.DataFrame:
    """
    Compare the overall text of each pair of students using BERTScore.

    Args:
        student_texts (List[List[str]]): List of lists, where each inner list contains sentences from a student's answers.
        lang (str): Language code for BERTScore.

    Returns:
        pd.DataFrame: A DataFrame with pairwise BERTScore results between students.
    """
    results = []
    student_ids = [f"Student{i+1}" for i in range(len(student_texts))]

    # Generate all pair combinations of students
    for (i, student1_text), (j, student2_text) in combinations(
        enumerate(student_texts), 2
    ):
        student1_full_text = " ".join(student1_text)
        student2_full_text = " ".join(student2_text)

        # Calculate BERTScore
        P, R, F1 = score([student1_full_text], [student2_full_text], lang=lang)

        # Store results
        results.append(
            {
                "Person 1": student_ids[i],
                "Person 2": student_ids[j],
                "Precision": P.mean().item(),
                "Recall": R.mean().item(),
                "F1": F1.mean().item(),
            }
        )

    return pd.DataFrame(results)  # Visualize the results


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


def extract_keywords_from_sentences(
    topic_df: pd.DataFrame, stopwords: set
) -> Dict[str, List[str]]:
    """
    Extract keywords from sentences in topic_df and group them by person.

    Args:
        topic_df (pd.DataFrame): DataFrame with columns ['Person', 'Sentence'].
        stopwords (set): Set of stopwords to exclude.

    Returns:
        Dict[str, List[str]]: Dictionary where keys are person names and values are lists of keywords.
    """
    okt = Okt()
    keywords_dict = {}

    for person in topic_df["Person"].unique():
        # Filter sentences by person
        person_sentences = topic_df[topic_df["Person"] == person]["Sentence"]

        # Extract keywords from all sentences
        all_keywords = []
        for sentence in person_sentences:
            nouns = okt.nouns(sentence)  # Extract nouns
            filtered_nouns = [
                word
                for word in nouns
                if word not in stopwords and len(word) > 1
            ]
            all_keywords.extend(filtered_nouns)

        keywords_dict[person] = all_keywords

    return keywords_dict


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
