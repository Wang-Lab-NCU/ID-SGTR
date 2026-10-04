"""Offline structural audit for explicit and hybrid local knowledge graphs."""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from typing import Iterable

import pandas as pd


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def parse_ids(value: object) -> list[str]:
    """Parse a serialized evidence-ID collection while preserving order."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            values = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (SyntaxError, ValueError):
            values = text.split(",")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        item = _clean(item)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _edge_index(frame: pd.DataFrame) -> dict[str, list[tuple[str, str, str]]]:
    required = {"context_id", "node_1", "node_2", "chunk_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"edge table is missing columns: {sorted(missing)}")
    result: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for context, left, right, chunk in frame[
        ["context_id", "node_1", "node_2", "chunk_id"]
    ].itertuples(index=False, name=None):
        context = _clean(context)
        left = _clean(left)
        right = _clean(right)
        chunk = _clean(chunk)
        if context and left and right:
            result[context].append((left, right, chunk))
    return result


def _connectivity(
    edges: Iterable[tuple[str, str, str]], gold_chunks: list[str]
) -> dict[str, object]:
    union_find = _UnionFind()
    chunk_nodes: dict[str, set[str]] = defaultdict(set)
    gold_set = set(gold_chunks)
    edge_count = 0
    for left, right, chunk in edges:
        union_find.union(left, right)
        edge_count += 1
        if chunk in gold_set:
            chunk_nodes[chunk].update((left, right))

    mapped = [chunk for chunk in gold_chunks if chunk_nodes.get(chunk)]
    component_sets: list[set[str]] = []
    for chunk in gold_chunks:
        nodes = chunk_nodes.get(chunk, set())
        component_sets.append({union_find.find(node) for node in nodes})

    all_mapped = bool(gold_chunks) and len(mapped) == len(gold_chunks)
    shared_components = (
        set.intersection(*component_sets)
        if all_mapped and component_sets else set()
    )
    return {
        "mapped_chunks": mapped,
        "mapped_recall": len(mapped) / len(gold_chunks) if gold_chunks else 0.0,
        "connected": bool(shared_components),
        "shared_component_count": len(shared_components),
        "edge_count": edge_count,
    }


def audit_hybrid_connectivity(
    subset: pd.DataFrame,
    explicit_graph: pd.DataFrame,
    implicit_graph: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    """Classify each query as explicit-complete, implicitly repaired, or unrepaired.

    Connectivity is measured inside the full Reasoning-Setting context. A query is
    complete when every gold chunk maps to graph nodes and all gold-chunk node sets
    share at least one connected component.
    """
    required = {"query_id", "context_id", "gold_evidence"}
    missing = required.difference(subset.columns)
    if missing:
        raise ValueError(f"subset is missing columns: {sorted(missing)}")

    explicit_by_context = _edge_index(explicit_graph)
    implicit_by_context = _edge_index(implicit_graph)
    rows: list[dict[str, object]] = []
    for row in subset.to_dict(orient="records"):
        context = _clean(row["context_id"])
        gold = parse_ids(row["gold_evidence"])
        explicit_edges = explicit_by_context.get(context, [])
        hybrid_edges = explicit_edges + implicit_by_context.get(context, [])
        explicit = _connectivity(explicit_edges, gold)
        hybrid = _connectivity(hybrid_edges, gold)

        explicit_complete = bool(explicit["connected"])
        hybrid_complete = bool(hybrid["connected"])
        if explicit_complete:
            group = "explicit_complete"
        elif hybrid_complete:
            group = "implicit_repaired"
        else:
            group = "unrepaired"

        rows.append({
            "query_id": _clean(row["query_id"]),
            "dataset": dataset,
            "context_id": context,
            "gold_evidence": gold,
            "gold_evidence_count": len(gold),
            "explicit_mapped_evidence": explicit["mapped_chunks"],
            "explicit_mapped_recall": explicit["mapped_recall"],
            "explicit_complete": explicit_complete,
            "explicit_broken": not explicit_complete,
            "hybrid_mapped_evidence": hybrid["mapped_chunks"],
            "hybrid_mapped_recall": hybrid["mapped_recall"],
            "hybrid_complete": hybrid_complete,
            "implicit_repaired": (not explicit_complete) and hybrid_complete,
            "unrepaired": not hybrid_complete,
            "structural_group": group,
            "explicit_context_edges": explicit["edge_count"],
            "implicit_context_edges": len(implicit_by_context.get(context, [])),
        })
    return pd.DataFrame(rows)


def summarize_hybrid_audit(frame: pd.DataFrame, *, dataset: str) -> dict[str, object]:
    count = len(frame)

    def count_true(column: str) -> int:
        return int(frame[column].astype(bool).sum())

    explicit_complete = count_true("explicit_complete")
    explicit_broken = count_true("explicit_broken")
    implicit_repaired = count_true("implicit_repaired")
    unrepaired = count_true("unrepaired")
    denominator = count or 1
    return {
        "dataset": dataset,
        "count": count,
        "explicit_complete": explicit_complete,
        "explicit_complete_rate": explicit_complete / denominator,
        "explicit_broken": explicit_broken,
        "explicit_broken_rate": explicit_broken / denominator,
        "implicit_repaired": implicit_repaired,
        "implicit_repaired_rate": implicit_repaired / denominator,
        "repair_rate_among_explicit_broken": (
            implicit_repaired / explicit_broken if explicit_broken else 0.0
        ),
        "unrepaired": unrepaired,
        "unrepaired_rate": unrepaired / denominator,
        "mean_explicit_gold_mapping_recall": float(frame["explicit_mapped_recall"].mean()),
        "mean_hybrid_gold_mapping_recall": float(frame["hybrid_mapped_recall"].mean()),
        "connectivity_definition": (
            "all gold-chunk node sets intersect one connected component within "
            "the query's complete local context"
        ),
    }
