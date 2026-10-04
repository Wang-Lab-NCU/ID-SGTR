import networkx as nx

from knowledge_graph.experiments.evidence import EvidenceItem
from knowledge_graph.experiments.stage0_gate import (
    parse_stage0_confidence,
    parse_supporting_refs,
    verify_stage0_answer,
)


def _item(chunk_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(chunk_id=chunk_id, text=text)


def test_grounded_single_reference_is_accepted():
    result = verify_stage0_answer(
        query="Where was Ada Lovelace born?",
        answer="London",
        cited_refs=["1"],
        stage0_items=[_item("1", "Ada Lovelace was born in London.")],
        policy="evidence_verified",
    )
    assert result.accepted
    assert result.answer_grounded


def test_missing_reference_is_inferred_but_invalid_reference_is_rejected():
    missing = verify_stage0_answer(
        query="Where was Ada Lovelace born?", answer="London", cited_refs=[],
        stage0_items=[_item("1", "Ada Lovelace was born in London.")],
        policy="evidence_verified",
    )
    invalid = verify_stage0_answer(
        query="Where was Ada Lovelace born?", answer="London", cited_refs=["9"],
        stage0_items=[_item("1", "Ada Lovelace was born in London.")],
        policy="evidence_verified",
    )
    assert missing.accepted
    assert missing.refs_inferred
    assert missing.cited_refs == ["1"]
    assert invalid.rejection_reason == "invalid_citations"


def test_ungrounded_answer_is_rejected():
    result = verify_stage0_answer(
        query="Where was Ada Lovelace born?", answer="Paris", cited_refs=["1"],
        stage0_items=[_item("1", "Ada Lovelace was born in London.")],
        policy="evidence_verified",
    )
    assert not result.accepted
    assert result.rejection_reason == "answer_not_grounded"


def test_boolean_answer_requires_connected_multiple_references():
    graph = nx.Graph()
    graph.add_edge("Scott", "American", chunk_ids=["1"])
    graph.add_edge("Ed", "American", chunk_ids=["4"])
    graph.add_edge("Scott", "Ed", chunk_ids=["bridge"])
    items = [
        _item("1", "Scott Derrickson is American."),
        _item("4", "Ed Wood was an American filmmaker."),
    ]
    result = verify_stage0_answer(
        query="Were Scott Derrickson and Ed Wood of the same nationality?",
        answer="yes", cited_refs=["1", "4"], stage0_items=items, graph=graph,
        policy="evidence_verified",
    )
    assert result.accepted
    assert result.path_connected


def test_boolean_answer_rejects_one_reference():
    result = verify_stage0_answer(
        query="Were Scott Derrickson and Ed Wood of the same nationality?",
        answer="yes", cited_refs=["1"],
        stage0_items=[_item("1", "Scott Derrickson is American.")],
        policy="evidence_verified",
    )
    assert not result.accepted
    assert result.rejection_reason == "boolean_requires_multiple_refs"


def test_supporting_reference_parser_accepts_ref_prefixes():
    assert parse_supporting_refs("Final Answer: yes\nSupporting Refs: [Ref 1, Ref 4]") == ["1", "4"]


def test_boolean_support_is_inferred_from_connected_stage0_chunks():
    graph = nx.Graph()
    graph.add_edge("Scott", "American", chunk_ids=["1"])
    graph.add_edge("Ed", "American", chunk_ids=["4"])
    result = verify_stage0_answer(
        query="Were Scott Derrickson and Ed Wood of the same nationality?",
        answer="yes",
        cited_refs=[],
        stage0_items=[
            _item("1", "Scott Derrickson is American."),
            _item("4", "Ed Wood was an American filmmaker."),
        ],
        graph=graph,
        policy="evidence_verified",
    )
    assert result.accepted
    assert result.refs_inferred
    assert set(result.cited_refs) == {"1", "4"}


def test_confidence_parser_requires_explicit_label():
    assert parse_stage0_confidence("Final Answer: London\nConfidence: HIGH") == "high"
    assert parse_stage0_confidence("DEFER") == "defer"
    assert parse_stage0_confidence("I am very confident.") == ""


def test_confidence_grounded_accepts_high_with_valid_grounded_citation():
    result = verify_stage0_answer(
        query="Where was Ada Lovelace born?",
        answer="London",
        cited_refs=["1"],
        stage0_items=[_item("1", "Ada Lovelace was born in London.")],
        policy="confidence_grounded",
        confidence="high",
    )
    assert result.accepted
    assert result.model_confidence == "high"


def test_confidence_grounded_rejects_low_or_ungrounded_support_and_infers_refs():
    item = _item("1", "Ada Lovelace was born in London.")
    low = verify_stage0_answer(
        query="Where was Ada Lovelace born?", answer="London",
        cited_refs=["1"], stage0_items=[item],
        policy="confidence_grounded", confidence="low",
    )
    missing = verify_stage0_answer(
        query="Where was Ada Lovelace born?", answer="London",
        cited_refs=[], stage0_items=[item],
        policy="confidence_grounded", confidence="high",
    )
    ungrounded = verify_stage0_answer(
        query="Where was Ada Lovelace born?", answer="Paris",
        cited_refs=["1"], stage0_items=[item],
        policy="confidence_grounded", confidence="high",
    )
    assert low.rejection_reason == "model_not_high_confidence"
    assert missing.accepted
    assert missing.refs_inferred
    assert missing.cited_refs == ["1"]
    assert ungrounded.rejection_reason == "answer_not_grounded"


def test_confidence_coverage_rejects_high_confidence_with_one_reference():
    result = verify_stage0_answer(
        query="Who is the mother of the director of Polish-Russian War?",
        answer="Malgorzata Braunek",
        cited_refs=["2"],
        stage0_items=[
            _item("1", "Polish-Russian War was directed by Xawery Zulawski."),
            _item("2", "Malgorzata Braunek was the mother of Xawery Zulawski."),
        ],
        policy="confidence_coverage",
        confidence="high",
    )
    assert not result.accepted
    assert result.rejection_reason == "insufficient_supporting_refs"


def test_confidence_coverage_accepts_connected_complete_support():
    graph = nx.Graph()
    graph.add_edge("film", "director", chunk_ids=["1"])
    graph.add_edge("director", "mother", chunk_ids=["2"])
    result = verify_stage0_answer(
        query="Who is the mother of the director of Polish-Russian War?",
        answer="Malgorzata Braunek",
        cited_refs=["1", "2"],
        stage0_items=[
            _item("1", "Polish-Russian War was directed by Xawery Zulawski."),
            _item("2", "Malgorzata Braunek was the mother of Xawery Zulawski."),
        ],
        graph=graph,
        policy="confidence_coverage",
        confidence="high",
    )
    assert result.accepted
    assert result.path_connected
