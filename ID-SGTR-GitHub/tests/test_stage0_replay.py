from __future__ import annotations

import pandas as pd

from knowledge_graph.experiments.stage0_replay import (
    load_stage0_replay,
    replay_decision,
    replay_seeds,
)


def test_stage0_replay_round_trip(tmp_path):
    source = tmp_path / "baseline.csv"
    pd.DataFrame([{
        "question": "Who directed the film?",
        "stage0_seed_entities": ["Film", "Director"],
        "stage0_decision": {
            "is_final": False,
            "answer": "",
            "relevant_nodes": ["Film"],
            "next_nodes": ["Director"],
        },
    }]).to_csv(source, sep="|", index=False)

    replay = load_stage0_replay(source)
    seeds, seeds_applied = replay_seeds(" Who directed  the film? ", ["Other"], replay)
    decision, decision_applied = replay_decision(
        "Who directed the film?",
        {"is_final": True, "answer": "wrong"},
        replay,
    )

    assert seeds_applied is True
    assert seeds == ["Film", "Director"]
    assert decision_applied is True
    assert decision["is_final"] is False
    assert decision["relevant_nodes"] == ["Film"]
    assert decision["next_nodes"] == ["Director"]


def test_stage0_replay_uses_query_id_for_duplicate_questions(tmp_path):
    source = tmp_path / "baseline.csv"
    pd.DataFrame([
        {
            "query_id": "q:1",
            "question": "Repeated question?",
            "stage0_seed_entities": ["First"],
            "stage0_decision": {"is_final": False, "next_nodes": ["First"]},
        },
        {
            "query_id": "q:2",
            "question": "Repeated question?",
            "stage0_seed_entities": ["Second"],
            "stage0_decision": {"is_final": True, "answer": "Second"},
        },
    ]).to_csv(source, sep="|", index=False)

    replay = load_stage0_replay(source)
    first, _ = replay_seeds(
        "Repeated question?", [], replay, query_id="q:1"
    )
    second, _ = replay_seeds(
        "Repeated question?", [], replay, query_id="q:2"
    )
    decision, applied = replay_decision(
        "Repeated question?", {}, replay, query_id="q:2"
    )

    assert first == ["First"]
    assert second == ["Second"]
    assert applied is True
    assert decision["answer"] == "Second"
