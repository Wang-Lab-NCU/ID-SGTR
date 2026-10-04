import json

import pandas as pd
import pytest

from knowledge_graph.experiments.run_p0 import build_parser, command_stratify
from knowledge_graph.experiments.stratified_analysis import stratify_results


def _frame():
    return pd.DataFrame([
        {
            "query_id": "q:0", "em": 1.0, "f1": 1.0,
            "foldable": True, "folding_fallback": False,
            "path_continuous": True, "branch_complete": True,
            "manifest_order_changed": True,
            "folding_fallback_reason": "",
            "intent_type": "Reasoning", "path_length": 3,
            "selected_count": 3, "complete_evidence_set": 1.0,
        },
        {
            "query_id": "q:1", "em": 0.0, "f1": 0.5,
            "foldable": None, "folding_fallback": True,
            "path_continuous": False, "branch_complete": False,
            "manifest_order_changed": False,
            "folding_fallback_reason": "no_continuous_relation_path",
            "intent_type": "Unexpected", "path_length": 4,
            "selected_count": 2, "complete_evidence_set": None,
        },
    ])


def test_stratify_uses_fixed_declared_slices_and_unknowns():
    result = stratify_results(_frame())
    assert set(result["stratum"]) == {
        "overall", "foldable", "folding_fallback", "path_continuous",
        "branch_complete", "manifest_order_changed", "intent_type",
        "path_length", "selected_count", "complete_evidence_set",
        "folding_fallback_reason",
    }
    overall = result.loc[result["stratum"].eq("overall")].iloc[0]
    assert overall["count"] == 2
    assert overall["f1"] == 0.75
    assert overall["coverage"] == 1.0
    path = result.loc[result["stratum"].eq("path_length")]
    assert set(path["value"]) == {"0", "1", "2", "3", "4+", "unknown"}
    assert path.set_index("value").loc["3", "count"] == 1
    assert path.set_index("value").loc["4+", "count"] == 1
    foldable = result.loc[result["stratum"].eq("foldable")].set_index("value")
    assert foldable.loc["unknown", "count"] == 1
    intent = result.loc[result["stratum"].eq("intent_type")].set_index("value")
    assert intent.loc["unknown", "count"] == 1
    complete = result.loc[
        result["stratum"].eq("complete_evidence_set")
    ].set_index("value")
    assert complete.loc["unknown", "count"] == 1
    for stratum in result["stratum"].unique():
        if stratum == "overall":
            continue
        assert result.loc[result["stratum"].eq(stratum), "count"].sum() == 2


def test_stratify_rejects_empty_and_duplicate_results():
    with pytest.raises(ValueError, match="must not be empty"):
        stratify_results(pd.DataFrame(columns=["query_id"]))
    duplicated = _frame().copy()
    duplicated.loc[1, "query_id"] = "q:0"
    with pytest.raises(ValueError, match="unique query_id"):
        stratify_results(duplicated)


def test_stratify_command_scores_raw_pipe_csv_and_writes_outputs(tmp_path):
    source = tmp_path / "raw.csv"
    output = tmp_path / "strata.csv"
    scored = tmp_path / "scored.csv"
    json_output = tmp_path / "strata.json"
    raw = _frame().drop(columns=["em", "f1", "complete_evidence_set"])
    raw["pred_answer"] = ["Paris", "London"]
    raw["gold_answer"] = ["Paris", "Rome"]
    raw["retrieved_evidence"] = ["['a']", "['b']"]
    raw["gold_evidence"] = ["['a']", "['c']"]
    raw.to_csv(source, sep="|", index=False)
    args = build_parser().parse_args([
        "stratify", "--results", str(source), "--output", str(output),
        "--scored-output", str(scored), "--json-output", str(json_output),
    ])
    command_stratify(args)
    written = pd.read_csv(output, sep="|")
    assert "f1" in written
    assert "f1" in pd.read_csv(scored, sep="|")
    assert json.loads(json_output.read_text(encoding="utf-8"))[0]["stratum"] == "overall"


def test_cli_exposes_stratify():
    assert "stratify" in build_parser().format_help()
