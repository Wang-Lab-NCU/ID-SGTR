import ast
from pathlib import Path

from knowledge_graph.experiments.terminal_recovery import (
    build_terminal_query,
    deduplicate_chunk_ids_by_text,
    parse_terminal_response,
    select_terminal_chunks,
)


def test_terminal_query_contains_runtime_state():
    text = build_terminal_query(
        "Who is the spouse?",
        entities=["Green", "Steve Hillage"],
        facts=["Green -> Steve Hillage"],
    )
    assert "Who is the spouse?" in text
    assert "Steve Hillage" in text
    assert "Green -> Steve Hillage" in text


def test_terminal_selection_preserves_path_core_under_budget():
    selected = select_terminal_chunks(
        ["9", "8", "7", "6", "5"], ["1", "2", "3"], budget=5
    )
    assert len(selected) == 5
    assert selected[:3] == ["9", "8", "7"]
    assert selected[-2:] == ["1", "2"]


def test_cross_context_duplicate_text_is_removed_with_path_id_preferred():
    texts = {
        "10": "  The SAME\npassage. ",
        "99": "the same passage.",
        "20": "Different evidence",
    }
    deduplicated = deduplicate_chunk_ids_by_text(
        ["10", "20", "99"], texts.get, preferred_ids=["99"]
    )
    assert deduplicated == ["99", "20"]


def test_terminal_budget_is_not_consumed_by_duplicate_text():
    texts = {"1": "same", "2": " SAME ", "3": "other", "4": "last"}
    selected = select_terminal_chunks(
        ["1", "2", "3", "4"], [], budget=3, text_getter=texts.get
    )
    assert selected == ["1", "3", "4"]


def test_terminal_response_parser():
    draft = parse_terminal_response(
        "Final Answer: Miquette Giraudy\n"
        "Supporting Refs: Ref 10, 5\n"
        "Confidence: HIGH"
    )
    assert draft.answer == "Miquette Giraudy"
    assert draft.supporting_refs == ["10", "5"]
    assert draft.confidence == "high"


def test_terminal_response_parser_accepts_empty_answer_field():
    draft = parse_terminal_response(
        "Final Answer:\nSupporting Refs:\nConfidence: LOW"
    )
    assert draft.answer == ""
    assert draft.supporting_refs == []
    assert draft.confidence == "low"


def test_global_terminal_timing_dependency_is_imported():
    root = Path(__file__).resolve().parents[1] / "knowledge_graph"
    for dataset in ("hotpot", "2wiki", "musique"):
        source = (root / dataset / "query_global.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = any(
            isinstance(node, ast.Import)
            and any(alias.name == "time" for alias in node.names)
            for node in tree.body
        )
        assert imported, dataset


def test_musique_restores_topology_final_synthesis():
    root = Path(__file__).resolve().parents[1] / "knowledge_graph" / "musique"
    for filename in ("query_local.py", "query_global.py"):
        source = (root / filename).read_text(encoding="utf-8")
        assert '"Topology-Final-Synthesis"' in source
        assert "_answer_from_folded_context(query, last_folded_chunks)" in source
        assert "ID_SGTR_MUSIQUE_FINAL_POLICY" in source
        assert '"Terminal-Recovery-v2.1"' in source
