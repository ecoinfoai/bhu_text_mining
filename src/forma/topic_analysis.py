"""BERTopic-based topic modeling for lecture transcript analysis."""

from __future__ import annotations

import warnings

import yaml
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import kss
import pandas as pd


def load_yaml_data(yaml_path: str) -> dict[str, str]:
    """
    Load data from a YAML file into a dictionary.

    Reads the contents of a YAML file and returns it as a dictionary where
    keys represent identifiers and values are associated text data.

    Args:
        yaml_path (str): The path to the YAML file.

    Returns:
        dict[str, str]: A dictionary containing the YAML file's key-value pairs.

    Examples:
        >>> yaml_data = load_yaml_data("data.yaml")
        >>> yaml_data["person1"]
        "Sample text for person1."
    """
    with open(yaml_path, "r", encoding="UTF-8") as f:
        return yaml.safe_load(f)


def split_sentences(yaml_data: dict[str, str]) -> tuple[list[str], list[str]]:
    """
    Split the text data in a YAML dictionary into individual sentences.

    Processes the text data associated with each key in the input dictionary
    by splitting it into sentences using the KSS library. Returns the list
    of sentences and their corresponding keys.

    Args:
        yaml_data (dict[str, str]): A dictionary where keys are identifiers
            and values are text data.

    Returns:
        tuple[list[str], list[str]]: A tuple containing a list of all
        sentences and a list of corresponding keys.

    Examples:
        >>> yaml_data = {"person1": "Hello. How are you?"}
        >>> sentences, keys = split_sentences(yaml_data)
        >>> sentences
        ["Hello.", "How are you?"]
        >>> keys
        ["person1", "person1"]
    """
    all_sentences = []
    keys = []
    for key, text in yaml_data.items():
        sentences = kss.split_sentences(text)
        all_sentences.extend(sentences)
        keys.extend([key] * len(sentences))
    return all_sentences, keys


def configure_bertopic(env_config: dict) -> BERTopic:
    """
    Configure a BERTopic model using environment settings.

    Initializes a BERTopic model with custom configurations for the UMAP,
    HDBSCAN, and Sentence Transformer embedding models based on the input
    dictionary.

    Args:
        env_config (Dict): A dictionary containing configuration parameters
            for UMAP, HDBSCAN, and embedding models.

    Returns:
        BERTopic: An initialized BERTopic model instance.

    Examples:
        >>> env_config = {
        ...     "umap": {"n_neighbors": 15, "n_components": 5},
        ...     "hdbscan": {"min_cluster_size": 10},
        ...     "embedding_model": "all-MiniLM-L6-v2"
        ... }
        >>> topic_model = configure_bertopic(env_config)
    """
    umap_model = UMAP(**env_config["umap"])
    hdbscan_model = HDBSCAN(**env_config["hdbscan"])
    embedding_model = SentenceTransformer(env_config["embedding_model"])

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
    )


def analyze_topics_with_bertopic(
    topic_model: BERTopic, sentences: list[str]
) -> tuple[list[int], list[float]]:
    """
    Analyze topics in a set of sentences using BERTopic.

    Fits the BERTopic model to the input sentences and generates topic
    assignments and probabilities for each sentence.

    Args:
        topic_model (BERTopic): The BERTopic model instance to use.
        sentences (list[str]): A list of sentences to analyze.

    Returns:
        tuple[list[int], list[float]]: A tuple containing a list of topic
        assignments and a list of topic probabilities.

    Examples:
        >>> sentences = ["This is a test sentence.", "Another example sentence."]
        >>> topics, probs = analyze_topics_with_bertopic(topic_model, sentences)
    """
    topics, probs = topic_model.fit_transform(sentences)
    return topics, probs


def generate_topic_dataframe(
    sentences: list[str], topics: list[int], keys: list[str]
) -> pd.DataFrame:
    """
    Generate a DataFrame summarizing topic assignments for sentences.

    Constructs a DataFrame where each row represents a sentence, its topic
    assignment, and associated metadata (key and sentence number).

    Args:
        sentences (list[str]): A list of sentences.
        topics (list[int]): A list of topic assignments for the sentences.
        keys (list[str]): A list of keys corresponding to each sentence.

    Returns:
        pd.DataFrame: A DataFrame with columns for keys, topic numbers,
        sentence numbers, and sentences.

    Examples:
        >>> sentences = ["Hello world.", "This is a test."]
        >>> topics = [0, 1]
        >>> keys = ["person1", "person2"]
        >>> df = generate_topic_dataframe(sentences, topics, keys)
        >>> df.head()
          Person  Topic No. Sentence No.             Sentence
        0  person1          0    person1_0000       Hello world.
        1  person2          1    person2_0001  This is a test.
    """
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
    topic_model: BERTopic, sentences: list[str], topics: list[int]
) -> pd.DataFrame:
    """
    Generate a table of topic keywords from the BERTopic model.

    Extracts the top keywords for each topic identified by the BERTopic model
    and organizes them into a DataFrame. Outlier topics are excluded.

    Args:
        topic_model (BERTopic): The BERTopic model instance.
        sentences (list[str]): A list of sentences (used for reference).
        topics (list[int]): A list of topic assignments for the sentences.

    Returns:
        pd.DataFrame: A DataFrame with columns for topic numbers and their
        associated keywords.

    Examples:
        >>> topic_keywords_table = generate_topic_keywords_table(topic_model, sentences, topics)
        >>> topic_keywords_table.head()
           Topic           Keywords
        0      0  data, analysis, test
        1      1         example, word
    """
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
