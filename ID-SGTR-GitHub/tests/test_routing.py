import pandas as pd
import networkx as nx

from knowledge_graph.experiments.routing import (
    context_rerank_prompt,
    parse_selected_ids,
    rank_seed_candidates,
    rank_seed_entities,
    seed_rerank_decision,
)


def test_parse_selected_ids_only_accepts_explicit_array():
    assert parse_selected_ids("Selected IDs: [3, 1, 3]", 5) == [3, 1]
    assert parse_selected_ids("IDs 3 and 1; use at most 5", 6) == []
    assert parse_selected_ids("[9, 1]", 3) == [1]


def test_rank_seed_entities_penalizes_generic_hubs():
    graph = nx.Graph()
    graph.add_edges_from(("hub", f"node-{index}") for index in range(20))
    graph.add_edge("Specific Person", "answer")
    frame = pd.DataFrame([
        {"Standard_Entity": "hub", "Score": 0.50},
        {"Standard_Entity": "Specific Person", "Score": 0.50},
    ])
    assert rank_seed_entities(frame, graph, limit=2)[0] == "Specific Person"


def test_context_rerank_prompt_uses_opaque_candidate_ids():
    prompt = context_rerank_prompt("Who?", [["Alpha"], ["Beta", "Gamma"]])
    assert "Candidate 0" in prompt
    assert "Candidate 1" in prompt
    assert "[1]" in prompt


def test_seed_candidates_are_deduplicated_and_limited():
    frame = pd.DataFrame([
        {"Standard_Entity": "Ada Lovelace", "Score": 0.9},
        {"Standard_Entity": "ada lovelace", "Score": 0.8},
        *[
            {"Standard_Entity": f"Entity {index}", "Score": 0.7 - index / 100}
            for index in range(30)
        ],
    ])
    ranked = rank_seed_candidates(
        frame, nx.Graph(), query="Where was Ada Lovelace born?", limit=20,
    )
    assert len(ranked) == 20
    assert ranked[0].name == "Ada Lovelace"
    assert ranked[0].exact_query_match


def test_seed_reranker_is_conditional():
    frame = pd.DataFrame([
        {"Standard_Entity": "Ada Lovelace", "Score": 0.95},
        {"Standard_Entity": "London", "Score": 0.40},
        {"Standard_Entity": "Mathematics", "Score": 0.30},
        {"Standard_Entity": "Charles Babbage", "Score": 0.20},
        {"Standard_Entity": "Analytical Engine", "Score": 0.10},
        {"Standard_Entity": "England", "Score": 0.05},
    ])
    candidates = rank_seed_candidates(
        frame, nx.Graph(), query="Where was Ada Lovelace born?", limit=20,
    )
    use_llm, reason, margin, risk = seed_rerank_decision(candidates)
    assert not use_llm
    assert reason == "deterministic_high_confidence"
    assert margin > 0.04
    assert not risk
