from __future__ import annotations

from knowledge_graph.experiments.query_reranker import (
    QueryCrossEncoder,
    bind_rerank_query,
    current_rerank_query,
)
from knowledge_graph.experiments.evidence import EvidenceAssembler, EvidenceItem


class FakeReranker(QueryCrossEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.cross_weight = 0.8
        self.context_cross_weight = 0.8

    def score(self, query: str, documents: list[str]) -> list[float]:
        assert query
        return [float(document.count("relevant")) for document in documents]


def test_query_binding_is_scoped() -> None:
    assert current_rerank_query() == ""
    with bind_rerank_query("question"):
        assert current_rerank_query() == "question"
    assert current_rerank_query() == ""


def test_chunk_rank_fuses_cross_encoder_and_base_score() -> None:
    reranker = FakeReranker()
    ranked = reranker.rank(
        "question",
        ["a", "b", "c"],
        ["irrelevant", "relevant relevant", "other"],
        base_scores=[1.0, 0.0, 0.5],
    )
    assert ranked[0].item_id == "b"
    assert {item.item_id for item in ranked} == {"a", "b", "c"}


def test_context_reranking_aggregates_passages_and_keeps_unique_context() -> None:
    reranker = FakeReranker()
    ranked = reranker.rerank_contexts(
        "question",
        ["10", "20", "30"],
        [
            ["noise", "other"],
            ["relevant relevant", "relevant"],
            ["relevant", "noise"],
        ],
        base_scores=[0.9, 0.2, 0.5],
    )
    assert ranked[0].item_id == "20"
    assert len({item.item_id for item in ranked}) == 3


def test_path_constrained_budget_keeps_connected_second_hop(monkeypatch) -> None:
    monkeypatch.setenv("ID_SGTR_PATH_CONSTRAINED_SELECTION", "true")
    assembler = EvidenceAssembler(budget=2)
    items = [
        EvidenceItem("a", "a", score=1.0, triple="A --[r1]--> B", hop=1),
        EvidenceItem("noise", "noise", score=0.95, triple="X --[r]--> Y", hop=1),
        EvidenceItem("b", "b", score=0.92, triple="B --[r2]--> C", hop=2),
    ]
    selected = assembler.select(items, "graph_naive", query_id="multi hop query")
    assert [item.chunk_id for item in selected] == ["a", "b"]


def test_conservative_gate_protects_core_and_allows_one_replacement() -> None:
    reranker = FakeReranker()
    reranker.conservative_gate = True
    reranker.evidence_budget = 3
    reranker.protected_base_count = 2
    reranker.max_replacements = 1
    reranker.replacement_margin = 0.05

    gated = reranker._apply_conservative_gate(
        ["core-a", "core-b", "old-third", "new"],
        [0.75, 0.70, 0.40, 0.90],
        [1.00, 0.90, 0.80, 0.10],
    )
    order = sorted(range(4), key=lambda index: -gated[index])
    assert {order[0], order[1], order[2]} == {0, 1, 3}


def test_conservative_gate_rejects_low_margin_replacement() -> None:
    reranker = FakeReranker()
    reranker.conservative_gate = True
    reranker.evidence_budget = 3
    reranker.protected_base_count = 2
    reranker.max_replacements = 1
    reranker.replacement_margin = 0.05

    gated = reranker._apply_conservative_gate(
        ["core-a", "core-b", "old-third", "new"],
        [0.75, 0.70, 0.60, 0.64],
        [1.00, 0.90, 0.80, 0.10],
    )
    order = sorted(range(4), key=lambda index: -gated[index])
    assert {order[0], order[1], order[2]} == {0, 1, 2}
