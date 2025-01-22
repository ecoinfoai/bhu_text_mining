import pandas as pd
from matplotlib import font_manager as fm
from typing import Dict
from src.network_analysis import load_stopwords
from src.topic_analysis import (
    load_yaml_data,
    split_sentences,
    configure_bertopic,
    analyze_topics_with_bertopic,
    generate_topic_dataframe,
    generate_topic_keywords_table,
)
from src.cohesion_analysis import (
    calculate_overall_similarity,
    calculate_topicwise_similarity,
    compare_students_overall,
    extract_keywords_from_sentences,
    calculate_pairwise_sentence_similarity,
    calculate_topic_statistics,
)
from src.visualize_cohesion import (
    compare_topic_lengths_with_xy,
    generate_person_networks_from_sentences,
)
from src.knowledge_graph_analysis import (
    create_student_knowledge_graph,
    create_reference_knowledge_graph,
    compare_graphs,
    display_comparison_results,
    visualize_superimposed_graph,
)


def make_bertopic_model(yaml_path: str, env_config: Dict):

    # Setup your BERTopic model with data
    yaml_data = load_yaml_data(yaml_path)
    all_sentences, keys = split_sentences(yaml_data)
    topic_model = configure_bertopic(env_config)
    topics, probs = analyze_topics_with_bertopic(topic_model, all_sentences)
    topic_df = generate_topic_dataframe(all_sentences, topics, keys)

    topic_table = generate_topic_keywords_table(
        topic_model, all_sentences, topics
    )
    topic_lengths_df = compare_topic_lengths_with_xy(topic_df)

    professor_text = yaml_data.get("professor", "").split("\n")
    student_texts = [
        text.split("\n")
        for key, text in yaml_data.items()
        if key != "professor"
    ]
    overall_results = calculate_overall_similarity(
        professor_text, student_texts
    )

    topicwise_results = calculate_topicwise_similarity(topic_df)
    pairwise_results = compare_students_overall(student_texts)

    return (
        topic_df,
        topic_table,
        topic_lengths_df,
        overall_results,
        topicwise_results,
        pairwise_results,
    )


# Setup environment for topic modelling
yaml_path = "/home/kjeong/localgit/bhu_text_mining/data/text_data.yaml"
env_config = {
    "embedding_model": "jhgan/ko-sroberta-multitask",
    "umap": {
        "n_neighbors": 10,
        "min_dist": 0.1,
        "n_components": 5,
        "random_state": 42,
    },
    "hdbscan": {
        "min_cluster_size": 4,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True,
    },
}

# Setup environment for network analysis
stopwords_path = "./data/stopwords-ko.txt"
font_path = "/usr/share/fonts/nanum/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)

# Build your BERTopic model and extract import topics
(
    topic_df,
    topic_table,
    topic_lengths_df,
    overall_results,
    topicwise_results,
    pairwise_results,
) = make_bertopic_model(yaml_path, env_config)

# Examine topics
print(topic_df)
print(topic_table)
print(topic_lengths_df)

# Examine cohesion
# BERTscore from all sentences of each student compared with the professor's
print(overall_results)

# BERTscore from sentences of each topic of each student compared with the professor's
print(topicwise_results)

# BERTscore from all sentences of each student compared with each other
print(pairwise_results)

# Pairwise sentence comparison per topic per student
pairwise_cohesion = calculate_pairwise_sentence_similarity(topic_df)
print(pairwise_cohesion)

# Calculate avg and std for student's pairwise BERTscore
topic_statistics_df = calculate_topic_statistics(pairwise_cohesion)
print(topic_statistics_df)

# Create network graphs for each person
stopwords = load_stopwords(stopwords_path)
keywords_dict = extract_keywords_from_sentences(topic_df, stopwords)
generate_person_networks_from_sentences(topic_df, stopwords, font_prop)


# Knowledge graph analysis
reference_graph = create_reference_knowledge_graph(topic_df, stopwords)
students = ["student1", "student2"]

comparison_results = {}
for student in students:
    student_graph = create_student_knowledge_graph(
        student, topic_df, stopwords
    )
    results = compare_graphs(reference_graph, student_graph)
    comparison_results[student] = results


display_comparison_results(comparison_results)  # in DataFrame

for student in students:  # in graphs
    student_graph = create_student_knowledge_graph(
        student, topic_df, stopwords
    )
    visualize_superimposed_graph(
        reference_graph,
        student_graph,
        font_prop,
        title=f"Superimposed Graph: {student}",
    )
