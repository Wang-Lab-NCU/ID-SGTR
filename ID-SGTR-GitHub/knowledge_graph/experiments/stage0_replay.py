"""Freeze and replay pre-treatment Stage0 state for paired dynamic experiments."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def question_key(question: Any) -> str:
    normalized = " ".join(str(question).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def query_id_key(query_id: Any) -> str:
    return f"id:{str(query_id).strip()}"


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return {}
    parsed = ast.literal_eval(text)
    return dict(parsed) if isinstance(parsed, dict) else {}


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            return []
        parsed = ast.literal_eval(text)
        raw = parsed if isinstance(parsed, (list, tuple, set)) else []
    output: list[str] = []
    seen: set[str] = set()
    for item in raw:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output


def load_stage0_replay(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Stage0 replay file does not exist: {source}")
    frame = pd.read_csv(source, sep="|")
    required = {"question", "stage0_seed_entities", "stage0_decision"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Stage0 replay file is missing columns: {sorted(missing)}")

    output: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        value = {
            "seeds": _parse_list(row["stage0_seed_entities"]),
            "decision": _parse_mapping(row["stage0_decision"]),
        }
        raw_query_id = row.get("query_id")
        has_query_id = pd.notna(raw_query_id) and str(raw_query_id).strip()
        if has_query_id:
            id_key = query_id_key(raw_query_id)
            if id_key in output:
                raise ValueError("Stage0 replay file contains duplicate query IDs")
            output[id_key] = value

        key = question_key(row["question"])
        if key not in output:
            output[key] = value
        elif not has_query_id:
            raise ValueError(
                "Stage0 replay file contains duplicate questions without query IDs"
            )
    return output


def _lookup(
    question: Any,
    replay: dict[str, dict[str, Any]],
    query_id: Any = None,
) -> dict[str, Any] | None:
    if query_id is not None:
        frozen = replay.get(query_id_key(query_id))
        if frozen is not None:
            return frozen
    return replay.get(question_key(question))


def replay_seeds(
    question: Any,
    seeds: list[str],
    replay: dict[str, dict[str, Any]],
    query_id: Any = None,
) -> tuple[list[str], bool]:
    frozen = _lookup(question, replay, query_id=query_id)
    if not frozen:
        return list(seeds), False
    values = _parse_list(frozen.get("seeds", []))
    return (values or list(seeds)), True


def replay_decision(
    question: Any,
    decision: dict[str, Any],
    replay: dict[str, dict[str, Any]],
    query_id: Any = None,
) -> tuple[dict[str, Any], bool]:
    frozen = _lookup(question, replay, query_id=query_id)
    if not frozen:
        return dict(decision), False
    source = _parse_mapping(frozen.get("decision", {}))
    if not source:
        return dict(decision), False
    normalized = {
        "is_final": bool(source.get("is_final", False)),
        "answer": str(source.get("answer", "")),
        "relevant_nodes": _parse_list(source.get("relevant_nodes", [])),
        "next_nodes": _parse_list(source.get("next_nodes", [])),
    }
    return normalized, True
