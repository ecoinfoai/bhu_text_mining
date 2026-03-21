"""Text cohesion analysis using BERTScore sentence similarity."""

from __future__ import annotations

from itertools import combinations
from konlpy.tag import Okt
import pandas as pd
from bert_score import score


def calculate_bertscore(
    reference: list[str], candidate: list[str], lang: str = "ko"
) -> tuple[float, float, float]:
    """Calculate BERTScore between reference and candidate text lists.

    Args:
        reference: Reference text sentences.
        candidate: Candidate text sentences.
        lang: Language code for BERTScore.

    Returns:
        Tuple of (precision, recall, F1) mean scores.
    """
    P, R, F1 = score(candidate, reference, lang=lang)
    return P.mean().item(), R.mean().item(), F1.mean().item()


def calculate_overall_similarity(
    professor_text: list[str], student_texts: list[list[str]], lang: str = "ko"
) -> pd.DataFrame:
    """
    Calculate BERTScore for professor's full text compared to each student's full text.

    Args:
        professor_text (list[str]): List of sentences from the professor's answers.
        student_texts (list[list[str]]): List of lists, where each inner list
            contains sentences from a student's answers.
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
    topic_df: pd.DataFrame, lang: str = "ko"
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
    student_texts: list[list[str]], lang: str = "ko"
) -> pd.DataFrame:
    """
    Compare the overall text of each pair of students using BERTScore.

    Args:
        student_texts (list[list[str]]): List of lists, where each inner list
            contains sentences from a student's answers.
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


def extract_keywords_from_sentences(
    topic_df: pd.DataFrame, stopwords: set
) -> dict[str, list[str]]:
    """
    Extract keywords from sentences in topic_df and group them by person.

    Args:
        topic_df (pd.DataFrame): DataFrame with columns ['Person', 'Sentence'].
        stopwords (set): Set of stopwords to exclude.

    Returns:
        dict[str, list[str]]: Dictionary where keys are person names and values are lists of keywords.
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


def calculate_pairwise_sentence_similarity(
    topic_df: pd.DataFrame, lang: str = "ko"
) -> pd.DataFrame:
    """Calculate pairwise sentence BERTScore within each student's topics.

    For each student, computes BERTScore between all sentence pairs within
    each topic assignment.

    Args:
        topic_df: DataFrame with columns ['Person', 'Topic No.', 'Sentence', 'Sentence No.'].
        lang: Language code for BERTScore.

    Returns:
        DataFrame with pairwise BERTScore results per student per topic.
    """
    results = []
    # Exclude professor data
    students_df = topic_df[topic_df["Person"] != "professor"]

    # Process each student
    for person in students_df["Person"].unique():
        person_df = students_df[students_df["Person"] == person]

        # Get topic numbers for this student
        person_topics = person_df["Topic No."].unique()

        for topic in person_topics:
            # Extract sentences for this topic (preserve original index)
            topic_sentences_df = person_df[person_df["Topic No."] == topic]

            # Only compare if 2+ sentences exist
            if len(topic_sentences_df) < 2:
                continue

            # Generate and compare sentence pairs
            for (idx1, row1), (idx2, row2) in combinations(
                topic_sentences_df.iterrows(), 2
            ):
                sentence1 = row1["Sentence"]
                sentence2 = row2["Sentence"]
                sentence_no1 = row1["Sentence No."]
                sentence_no2 = row2["Sentence No."]

                P, R, F1 = score([sentence1], [sentence2], lang=lang)

                results.append(
                    {
                        "Person": person,
                        "Topic No.": topic,
                        "Sentence Pair": f"{sentence_no1} vs {sentence_no2}",
                        "Precision": P.mean().item(),
                        "Recall": R.mean().item(),
                        "F1": F1.mean().item(),
                    }
                )

    # Build DataFrame and sort
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by=["Person", "Topic No."]).reset_index(
        drop=True
    )

    return result_df


def calculate_topic_statistics(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and standard deviation of pairwise BERTScore per student per topic.

    Args:
        pairwise_df: Pairwise sentence similarity result DataFrame with
            columns ['Person', 'Topic No.', 'Precision', 'Recall', 'F1'].

    Returns:
        DataFrame with per-student per-topic mean and std statistics.
    """
    # Compute grouped statistics using relevant columns
    grouped = (
        pairwise_df.groupby(["Person", "Topic No."])
        .agg(
            Precision_Mean=("Precision", "mean"),
            Precision_Std=("Precision", "std"),
            Recall_Mean=("Recall", "mean"),
            Recall_Std=("Recall", "std"),
            F1_Mean=("F1", "mean"),
            F1_Std=("F1", "std"),
        )
        .reset_index()
    )

    return grouped
