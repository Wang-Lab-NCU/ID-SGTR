"""Map benchmark supporting passages to repository chunk IDs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _key(value: object) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value).casefold()).split())


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array in {path}")
    return value


def _record_titles(dataset: str, record: dict[str, Any]) -> set[str]:
    if dataset in {"hotpot", "2wiki"}:
        titles = {str(item[0]) for item in record.get("supporting_facts", []) if item}
    elif dataset == "musique":
        titles = {
            str(paragraph.get("title", ""))
            for paragraph in record.get("paragraphs", [])
            if paragraph.get("is_supporting")
        }
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return {_key(title) for title in titles if title}


def supporting_titles(dataset: str, raw_path: Path) -> dict[str, set[str]]:
    """Return normalized-question -> supporting-title set for all three datasets."""
    result: dict[str, set[str]] = {}
    for record in _load_json(raw_path):
        question = _key(record.get("question", ""))
        if question:
            result[question] = _record_titles(dataset, record)
    return result


def annotate_gold_evidence(
    subset: pd.DataFrame,
    chunks: pd.DataFrame,
    *,
    dataset: str,
    raw_path: Path,
) -> pd.DataFrame:
    required_subset = {"question", "context_id"}
    required_chunks = {"context_id", "chunk_id", "title"}
    if missing := required_subset.difference(subset.columns):
        raise ValueError(f"subset is missing columns: {sorted(missing)}")
    if missing := required_chunks.difference(chunks.columns):
        raise ValueError(f"chunk table is missing columns: {sorted(missing)}")

    records = _load_json(raw_path)
    title_map = supporting_titles(dataset, raw_path)
    chunk_copy = chunks.copy()
    chunk_copy["context_id"] = chunk_copy["context_id"].astype(str)
    chunk_copy["_title_key"] = chunk_copy["title"].map(_key)
    by_context = {context: group for context, group in chunk_copy.groupby("context_id")}

    annotated = subset.copy()
    gold_ids: list[list[str]] = []
    gold_titles: list[list[str]] = []
    for _, row in annotated.iterrows():
        question_key = _key(row["question"])
        titles = title_map.get(question_key, set())
        # The generated context_id is the source record index. Prefer it when
        # available so duplicate question strings cannot overwrite one another.
        context_value = str(row["context_id"])
        if context_value.isdigit() and int(context_value) < len(records):
            record = records[int(context_value)]
            if _key(record.get("question", "")) == question_key:
                titles = _record_titles(dataset, record)
        context_chunks = by_context.get(str(row["context_id"]))
        if context_chunks is None or not titles:
            ids: list[str] = []
        else:
            ids = context_chunks.loc[context_chunks["_title_key"].isin(titles), "chunk_id"].astype(str).tolist()
        gold_ids.append(list(dict.fromkeys(ids)))
        gold_titles.append(sorted(titles))
    annotated["gold_evidence"] = gold_ids
    annotated["gold_supporting_titles"] = gold_titles
    annotated["gold_mapping_complete"] = [bool(ids) and len(ids) >= len(titles) for ids, titles in zip(gold_ids, gold_titles)]
    return annotated
