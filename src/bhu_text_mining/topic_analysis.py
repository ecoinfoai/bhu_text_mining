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
