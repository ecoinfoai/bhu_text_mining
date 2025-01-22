import pytest
import pandas as pd
from typing import Dict, List

from src.knowledge_graph_analysis import (
    display_comparison_results,
)


@pytest.fixture
def sample_comparison_results():
    return {
        "student1": {
            "missing_nodes": ["node1", "node2"],
            "extra_nodes": ["node3"],
            "missing_edges": [("node1", "node4")],
            "extra_edges": [("node3", "node5")],
        },
        "student2": {
            "missing_nodes": ["node6"],
            "extra_nodes": ["node7", "node8"],
            "missing_edges": [("node6", "node9"), ("node10", "node11")],
            "extra_edges": [("node7", "node12")],
        },
    }


def test_display_comparison_results(sample_comparison_results):
    # Execution
    results_df = display_comparison_results(sample_comparison_results)

    # Verify the results
    assert isinstance(
        results_df, pd.DataFrame
    ), "Output should be a pandas DataFrame"
    assert len(results_df) == len(
        sample_comparison_results
    ), "DataFrame should have the same number of rows as the input dictionary"

    # Verify by students
    student1_row = results_df[results_df["Student"] == "student1"]
    assert not student1_row.empty, "Student1 should be in the DataFrame"
    assert (
        student1_row["Missing Nodes"].values[0] == "node1, node2"
    ), "Missing Nodes for student1 should match"
    assert (
        student1_row["Extra Nodes"].values[0] == "node3"
    ), "Extra Nodes for student1 should match"
    assert (
        student1_row["Missing Edges"].values[0] == 1
    ), "Missing Edges count for student1 should match"
    assert (
        student1_row["Extra Edges"].values[0] == 1
    ), "Extra Edges count for student1 should match"

    student2_row = results_df[results_df["Student"] == "student2"]
    assert not student2_row.empty, "Student2 should be in the DataFrame"
    assert (
        student2_row["Missing Nodes"].values[0] == "node6"
    ), "Missing Nodes for student2 should match"
    assert (
        student2_row["Extra Nodes"].values[0] == "node7, node8"
    ), "Extra Nodes for student2 should match"
    assert (
        student2_row["Missing Edges"].values[0] == 2
    ), "Missing Edges count for student2 should match"
    assert (
        student2_row["Extra Edges"].values[0] == 1
    ), "Extra Edges count for student2 should match"
