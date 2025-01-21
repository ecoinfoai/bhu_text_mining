from typing import Dict, List, Tuple
from itertools import combinations
from konlpy.tag import Okt, Mecab
import pandas as pd
from bert_score import score


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


def calculate_pairwise_sentence_similarity(
    topic_df: pd.DataFrame, lang="ko"
) -> pd.DataFrame:
    """
    각 학생의 문장들을 자신의 주제별로 pairwise 비교하여 BERTScore 계산.
    Args:
        topic_df (pd.DataFrame): DataFrame with columns ['Person', 'Topic No.', 'Sentence', 'Sentence No.'].
        lang (str): Language code for BERTScore.
    Returns:
        pd.DataFrame: 각 학생의 각 주제별 문장 간 BERTScore를 포함한 DataFrame.
    """
    results = []
    # 교수 데이터 제외
    students_df = topic_df[topic_df["Person"] != "professor"]

    # 각 학생별로 데이터 처리
    for person in students_df["Person"].unique():
        # 해당 학생의 데이터 필터링
        person_df = students_df[students_df["Person"] == person]

        # 해당 학생이 실제 작성한 문장의 토픽 번호만 가져오기
        person_topics = person_df["Topic No."].unique()

        for topic in person_topics:
            # 주제별 문장들을 DataFrame으로 추출 (원래 인덱스 유지)
            topic_sentences_df = person_df[person_df["Topic No."] == topic]

            # 문장이 2개 이상인 경우만 비교
            if len(topic_sentences_df) < 2:
                continue

            # 문장 쌍 생성 및 비교
            for (idx1, row1), (idx2, row2) in combinations(
                topic_sentences_df.iterrows(), 2
            ):
                sentence1 = row1["Sentence"]
                sentence2 = row2["Sentence"]
                sentence_no1 = row1["Sentence No."]
                sentence_no2 = row2["Sentence No."]

                P, R, F1 = score([sentence1], [sentence2], lang=lang)

                # 결과 저장
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

    # DataFrame 생성 및 정렬
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by=["Person", "Topic No."]).reset_index(
        drop=True
    )

    return result_df


def calculate_topic_statistics(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 학생의 각 주제별 pairwise BERTScore의 평균과 표준편차 계산.

    Args:
        pairwise_df (pd.DataFrame): pairwise sentence similarity 결과 DataFrame
                                    Columns: ['Person', 'Topic No.', 'Precision', 'Recall', 'F1'].

    Returns:
        pd.DataFrame: 각 학생의 각 주제별 평균 및 표준편차를 포함한 DataFrame.
    """
    # 필요한 컬럼만 사용하여 그룹별 통계 계산
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
