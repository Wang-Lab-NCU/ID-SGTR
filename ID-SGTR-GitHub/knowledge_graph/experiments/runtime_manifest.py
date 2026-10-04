"""Freeze source-aligned manifests from actual ID-SGTR runtime trajectories.

Unlike :mod:`topology_folding_v2`, this module does not reconstruct a path
from the complete local graph.  It accepts only graph traversals recorded by
the online engine before answer synthesis, validates their continuity, and
freezes the final evidence IDs for representation-only replay.  Gold answers,
gold evidence, and model predictions are intentionally never read.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .path_manifest import FoldManifest, PathStep


RUNTIME_IMPLEMENTATION_VERSION = (
    "topology-folding-runtime-trace-v1.5-factorial-path-order"
)
_TRIPLE_RE = re.compile(r"^\s*(.*?)\s+--\[(.*?)\]-->\s+(.*?)\s*$", re.DOTALL)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _clean_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    text = _clean_id(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [part.strip() for part in text.split(",") if part.strip()]
    return list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]


def _ids(value: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in _sequence(value):
        candidate = _clean_id(item)
        if candidate and candidate not in seen:
            output.append(candidate)
            seen.add(candidate)
    return output


def _score(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _parse_triple(value: Any) -> tuple[str, str, str] | None:
    match = _TRIPLE_RE.match(str(value or ""))
    if not match:
        return None
    source, relation, target = (part.strip() for part in match.groups())
    if not source or not relation or not target:
        return None
    return source, relation, target


def _phrase_in_question(entity: str, question: str) -> bool:
    tokens = tuple(token.casefold() for token in _WORD_RE.findall(entity))
    if not tokens:
        return False
    normalized = " ".join(token.casefold() for token in _WORD_RE.findall(question))
    return " ".join(tokens) in normalized


def _intent(strategy: Any) -> str:
    match = re.match(
        r"^\((Retrieval|Reasoning|Comparative|Default)\s*,",
        str(strategy or "").strip(),
    )
    return match.group(1) if match else "Default"


@dataclass(frozen=True)
class RuntimeEdge:
    source: str
    relation: str
    target: str
    chunk_id: str
    score: float
    hop: int
    position: int
    canonical_source: str
    canonical_target: str
    direction: str
    explicitly_chosen: bool = False

    @property
    def signature(self) -> tuple[str, str, str, str]:
        return self.source, self.relation, self.target, self.chunk_id


class RuntimeManifestBuilder:
    """Build immutable manifests solely from pre-answer runtime telemetry."""

    def __init__(
        self,
        chunks: pd.DataFrame,
        graph: pd.DataFrame,
        *,
        dataset: str,
        budget: int = 3,
        source_token_budget: int = 1400,
        trace_token_budget: int = 96,
        total_evidence_token_budget: int = 1536,
    ) -> None:
        required = {"context_id", "chunk_id"}
        missing = required.difference(chunks.columns)
        if missing:
            raise ValueError(f"chunks table is missing columns: {sorted(missing)}")
        if budget < 1:
            raise ValueError("budget must be positive")
        self.dataset = str(dataset)
        self.implementation_version = RUNTIME_IMPLEMENTATION_VERSION
        self.budget = int(budget)
        self.source_token_budget = int(source_token_budget)
        self.trace_token_budget = int(trace_token_budget)
        self.total_evidence_token_budget = int(total_evidence_token_budget)
        frame = chunks.loc[:, ["context_id", "chunk_id"]].copy()
        frame["context_id"] = frame["context_id"].astype(str)
        frame["chunk_id"] = frame["chunk_id"].astype(str)
        self.context_chunks = {
            context_id: tuple(dict.fromkeys(group["chunk_id"].tolist()))
            for context_id, group in frame.groupby("context_id", sort=False)
        }
        graph_required = {"context_id", "chunk_id", "node_1", "edge", "node_2"}
        graph_missing = graph_required.difference(graph.columns)
        if graph_missing:
            raise ValueError(
                f"graph table is missing columns: {sorted(graph_missing)}"
            )
        self.graph_edges: dict[tuple[str, str, str, str], list[str]] = {}
        for row in graph.itertuples(index=False):
            key = (
                str(row.context_id), str(row.node_1).strip(),
                str(row.edge).strip(), str(row.node_2).strip(),
            )
            values = self.graph_edges.setdefault(key, [])
            chunk_id = str(row.chunk_id)
            if chunk_id not in values:
                values.append(chunk_id)

    @staticmethod
    def _candidate_records(value: Any) -> list[dict[str, Any]]:
        return [item for item in _sequence(value) if isinstance(item, dict)]

    def _edge_records(
        self,
        candidates: Sequence[Mapping[str, Any]],
        selected_ids: Sequence[str],
        context_id: str,
    ) -> tuple[list[RuntimeEdge], dict[str, float]]:
        edges: dict[tuple[str, str, str, str], RuntimeEdge] = {}
        scores: dict[str, float] = {}
        selected = set(selected_ids)
        for order, candidate in enumerate(candidates):
            chunk_id = _clean_id(candidate.get("chunk_id"))
            if not chunk_id:
                continue
            candidate_score = _score(candidate.get("score"))
            scores[chunk_id] = max(scores.get(chunk_id, -math.inf), candidate_score)
        selected_rank = {chunk_id: index for index, chunk_id in enumerate(selected_ids)}
        for order, candidate in enumerate(candidates):
            chunk_id = _clean_id(candidate.get("chunk_id"))
            if not chunk_id:
                continue
            candidate_score = _score(candidate.get("score"))
            traces = _sequence(candidate.get("topology_trace"))
            if not traces and _clean_id(candidate.get("triple")):
                traces = [{
                    "hop": candidate.get("hop", 0),
                    "path_position": candidate.get(
                        "raw_path_position", candidate.get("path_position", order)
                    ),
                    "triple": candidate.get("triple", ""),
                }]
            for trace_order, trace in enumerate(traces):
                if isinstance(trace, dict):
                    triple = trace.get("triple", "")
                    hop = int(trace.get("hop", candidate.get("hop", 0)) or 0)
                    position = int(trace.get("path_position", trace_order) or trace_order)
                elif isinstance(trace, (list, tuple)) and len(trace) >= 3:
                    hop, position, triple = trace[:3]
                    hop, position = int(hop), int(position)
                else:
                    continue
                parsed = _parse_triple(triple)
                if parsed is None:
                    continue
                source, relation, target = parsed
                direct_key = (context_id, source, relation, target)
                reverse_key = (context_id, target, relation, source)
                direction = "forward"
                canonical_source, canonical_target = source, target
                aligned = [
                    value for value in self.graph_edges.get(direct_key, ())
                    if value in selected
                ]
                if not aligned:
                    aligned = [
                        value for value in self.graph_edges.get(reverse_key, ())
                        if value in selected
                    ]
                    if aligned:
                        direction = "reverse"
                        canonical_source, canonical_target = target, source
                if not aligned:
                    # The recorded traversal is not source-grounded in the
                    # final evidence window and therefore cannot inject trace.
                    continue
                supporting_chunk = min(
                    aligned, key=lambda value: (selected_rank[value], value)
                )
                edge = RuntimeEdge(
                    source=source, relation=relation, target=target,
                    chunk_id=supporting_chunk,
                    score=scores.get(supporting_chunk, candidate_score),
                    hop=hop, position=position,
                    canonical_source=canonical_source,
                    canonical_target=canonical_target,
                    direction=direction,
                )
                previous = edges.get(edge.signature)
                if previous is None or (edge.score, -edge.hop, -edge.position) > (
                    previous.score, -previous.hop, -previous.position
                ):
                    edges[edge.signature] = edge
        return list(edges.values()), {
            chunk_id: (0.0 if score == -math.inf else score)
            for chunk_id, score in scores.items()
        }

    def _chosen_edge_records(
        self,
        runtime_steps: Any,
        candidates: Sequence[Mapping[str, Any]],
        selected_ids: Sequence[str],
        context_id: str,
    ) -> tuple[list[RuntimeEdge], dict[str, float]]:
        """Align explicit choices plus the frontier actually executed next.

        Explicitly chosen edges have priority.  Executed-frontier edges are
        admitted only because the engine really traversed their targets; the
        full exposed candidate set is never reconstructed here.
        """

        scores: dict[str, float] = {}
        for candidate in candidates:
            chunk_id = _clean_id(candidate.get("chunk_id"))
            if chunk_id:
                scores[chunk_id] = max(
                    scores.get(chunk_id, -math.inf),
                    _score(candidate.get("score")),
                )
        selected = set(selected_ids)
        selected_rank = {
            chunk_id: index for index, chunk_id in enumerate(selected_ids)
        }
        edges: dict[tuple[str, str, str, str], RuntimeEdge] = {}
        for raw_step in _sequence(runtime_steps):
            if not isinstance(raw_step, dict):
                continue
            hop = int(raw_step.get("hop", 0) or 0)
            edge_streams = (
                (True, _sequence(raw_step.get("chosen_edges"))),
                (False, _sequence(raw_step.get("executed_edges"))),
            )
            for explicitly_chosen, raw_edges in edge_streams:
              for edge_order, raw_edge in enumerate(raw_edges, start=1):
                if not isinstance(raw_edge, dict):
                    continue
                source = _clean_id(raw_edge.get("source_entity"))
                relation = str(raw_edge.get("relation", "")).strip()
                target = _clean_id(raw_edge.get("target_entity"))
                if not source or not relation or not target:
                    continue
                recorded_chunks = set(_ids(raw_edge.get("chunk_ids")))
                direct_key = (context_id, source, relation, target)
                reverse_key = (context_id, target, relation, source)
                direction = "forward"
                canonical_source, canonical_target = source, target
                aligned = [
                    value for value in self.graph_edges.get(direct_key, ())
                    if value in selected and value in recorded_chunks
                ]
                if not aligned:
                    aligned = [
                        value for value in self.graph_edges.get(reverse_key, ())
                        if value in selected and value in recorded_chunks
                    ]
                    if aligned:
                        direction = "reverse"
                        canonical_source, canonical_target = target, source
                if not aligned:
                    continue
                supporting_chunk = min(
                    aligned, key=lambda value: (selected_rank[value], value)
                )
                position = int(
                    raw_edge.get("path_position", edge_order) or edge_order
                )
                edge = RuntimeEdge(
                    source=source,
                    relation=relation,
                    target=target,
                    chunk_id=supporting_chunk,
                    score=(
                        0.0 if scores.get(supporting_chunk, 0.0) == -math.inf
                        else scores.get(supporting_chunk, 0.0)
                    ),
                    hop=hop,
                    position=position,
                    canonical_source=canonical_source,
                    canonical_target=canonical_target,
                    direction=direction,
                    explicitly_chosen=explicitly_chosen,
                )
                previous = edges.get(edge.signature)
                if previous is None or (
                    edge.explicitly_chosen,
                    -edge.hop,
                    -edge.position,
                ) > (
                    previous.explicitly_chosen,
                    -previous.hop,
                    -previous.position,
                ):
                    edges[edge.signature] = edge
        for chunk_id in selected_ids:
            scores.setdefault(chunk_id, 0.0)
        return list(edges.values()), {
            chunk_id: (0.0 if score == -math.inf else score)
            for chunk_id, score in scores.items()
        }

    def _best_chain(
        self,
        edges: Sequence[RuntimeEdge],
        question: str,
        *,
        unique_chunks: bool = False,
    ) -> tuple[RuntimeEdge, ...]:
        adjacency: dict[str, list[RuntimeEdge]] = {}
        for edge in edges:
            adjacency.setdefault(edge.source, []).append(edge)
        paths: list[tuple[RuntimeEdge, ...]] = []

        def visit(path: tuple[RuntimeEdge, ...], used: frozenset[tuple[str, str, str, str]]):
            paths.append(path)
            if len(path) >= self.budget:
                return
            for edge in adjacency.get(path[-1].target, ()):
                if edge.signature not in used and (
                    not unique_chunks
                    or edge.chunk_id not in {item.chunk_id for item in path}
                ):
                    visit(path + (edge,), used | {edge.signature})

        for edge in edges:
            visit((edge,), frozenset({edge.signature}))
        if not paths:
            return ()

        def rank(path: tuple[RuntimeEdge, ...]) -> tuple[Any, ...]:
            anchor = int(_phrase_in_question(path[0].source, question))
            explicit = sum(edge.explicitly_chosen for edge in path)
            query_entities = sum(
                _phrase_in_question(entity, question)
                for edge in path for entity in (edge.source, edge.target)
            )
            unique_chunks = len(dict.fromkeys(edge.chunk_id for edge in path))
            mean_score = sum(edge.score for edge in path) / len(path)
            order = tuple((-edge.hop, -edge.position) for edge in path)
            signature = tuple(edge.signature for edge in path)
            return anchor, explicit, query_entities, unique_chunks, len(path), mean_score, order, signature

        return max(paths, key=rank)

    def _runtime_forest(
        self,
        edges: Sequence[RuntimeEdge],
        question: str,
        selected_ids: Sequence[str],
    ) -> tuple[tuple[RuntimeEdge, ...], ...]:
        """Partition the parallel runtime expansion into continuous branches.

        The online engine expands several active nodes in one hop.  Its final
        B evidence chunks therefore form a small path forest, not necessarily
        one serial entity chain.  Each source chunk contributes to at most one
        frozen branch, preventing repeated within-document facts from crowding
        out the other selected structural evidence.
        """

        remaining = list(edges)
        branches: list[tuple[RuntimeEdge, ...]] = []
        while remaining:
            chain = self._best_chain(
                remaining, question, unique_chunks=True,
            )
            if not chain:
                break
            branches.append(chain)
            used_chunks = {edge.chunk_id for edge in chain}
            remaining = [
                edge for edge in remaining if edge.chunk_id not in used_chunks
            ]
        selected_rank = {
            chunk_id: index for index, chunk_id in enumerate(selected_ids)
        }
        branches.sort(key=lambda chain: (
            min(edge.hop for edge in chain),
            min(edge.position for edge in chain),
            min(selected_rank.get(edge.chunk_id, len(selected_rank)) for edge in chain),
            tuple(edge.signature for edge in chain),
        ))
        return tuple(branches)

    def _comparative_branches(
        self,
        edges: Sequence[RuntimeEdge],
        question: str,
    ) -> tuple[tuple[RuntimeEdge, ...], bool]:
        ranked = sorted(
            edges,
            key=lambda edge: (
                _phrase_in_question(edge.source, question),
                edge.explicitly_chosen,
                edge.score,
                -edge.hop,
                -edge.position,
                edge.signature,
            ),
            reverse=True,
        )
        selected: list[RuntimeEdge] = []
        seen_anchors: set[str] = set()
        seen_edges: set[tuple[str, str, str, str]] = set()
        for edge in ranked:
            anchors = [
                entity for entity in (edge.source, edge.target)
                if _phrase_in_question(entity, question)
                and entity.casefold() not in seen_anchors
            ]
            if not anchors or edge.signature in seen_edges:
                continue
            selected.append(edge)
            seen_anchors.add(anchors[0].casefold())
            seen_edges.add(edge.signature)
            if len(selected) == 2:
                break
        return tuple(selected), len(selected) == 2

    def _fallback(
        self,
        *,
        row: Mapping[str, Any],
        candidate_ids: Sequence[str],
        selected_ids: Sequence[str],
        scores: Mapping[str, float],
        intent_type: str,
        reason: str,
    ) -> FoldManifest:
        score_order = tuple(sorted(
            selected_ids, key=lambda chunk_id: (-scores.get(chunk_id, 0.0), chunk_id)
        ))
        return FoldManifest(
            query_id=_clean_id(row.get("query_id")), dataset=self.dataset,
            context_id=_clean_id(row.get("context_id")),
            question=str(row.get("question", "")), intent_type=intent_type,
            intent_strategy=str(row.get("strategy", "")),
            path_policy="runtime_graph_naive_fallback", path_steps=(),
            candidate_chunk_ids=tuple(candidate_ids),
            selected_chunk_ids=tuple(score_order), core_chunk_ids=(),
            peripheral_chunk_ids=tuple(score_order),
            score_order_chunk_ids=tuple(score_order),
            path_order_chunk_ids=tuple(score_order),
            chunk_scores=tuple((chunk_id, scores.get(chunk_id, 0.0)) for chunk_id in candidate_ids),
            budget=self.budget, source_token_budget=self.source_token_budget,
            trace_token_budget=self.trace_token_budget,
            total_evidence_token_budget=self.total_evidence_token_budget,
            foldable=False, fallback_reason=reason,
            fallback_to_graph_naive=True, path_continuous=False,
            branch_complete=intent_type != "Comparative",
            implementation_version=self.implementation_version,
        ).validate()

    def build(self, row: Mapping[str, Any]) -> FoldManifest:
        # Deliberate whitelist: gold_answer, gold_evidence, and pred_answer are
        # not read anywhere in this method.
        context_id = _clean_id(row.get("context_id"))
        local_ids = self.context_chunks.get(context_id, ())
        local_set = set(local_ids)
        candidates = self._candidate_records(row.get("candidate_evidence"))
        selected_ids = [
            chunk_id for chunk_id in _ids(row.get("retrieved_evidence"))
            if chunk_id in local_set
        ][: self.budget]
        candidate_scores = {
            _clean_id(item.get("chunk_id")): _score(item.get("score"))
            for item in candidates if _clean_id(item.get("chunk_id"))
        }
        if not selected_ids:
            selected_ids = [
                chunk_id for chunk_id, _ in sorted(
                    candidate_scores.items(), key=lambda item: (-item[1], item[0])
                ) if chunk_id in local_set
            ][: self.budget]
        intent_type = _intent(row.get("strategy"))
        if not selected_ids:
            return self._fallback(
                row=row, candidate_ids=local_ids, selected_ids=(), scores={},
                intent_type=intent_type, reason="runtime_no_selected_evidence",
            )
        edges, scores = self._chosen_edge_records(
            row.get("runtime_path_steps"), candidates, selected_ids, context_id,
        )
        for chunk_id in selected_ids:
            scores.setdefault(chunk_id, candidate_scores.get(chunk_id, 0.0))

        branch_complete = True
        if intent_type == "Comparative":
            branch_edges, branch_complete = self._comparative_branches(
                edges, str(row.get("question", ""))
            )
            if not branch_complete:
                return self._fallback(
                    row=row, candidate_ids=local_ids, selected_ids=selected_ids,
                    scores=scores, intent_type=intent_type,
                    reason="runtime_comparative_branch_incomplete",
                )
            steps = tuple(
                PathStep(
                    step=1, branch=branch, source_entity=edge.source,
                    relation=edge.relation, target_entity=edge.target,
                    supporting_chunk_id=edge.chunk_id, edge_score=edge.score,
                    query_score=edge.score,
                    canonical_source_entity=edge.canonical_source,
                    canonical_target_entity=edge.canonical_target,
                    traversal_direction=edge.direction,
                )
                for branch, edge in zip(("A", "B"), branch_edges)
            )
            anchors = tuple(edge.source for edge in branch_edges)
            policy = "runtime_trace_comparative_branches"
        else:
            branches = self._runtime_forest(
                edges, str(row.get("question", "")), selected_ids,
            )
            if not branches:
                return self._fallback(
                    row=row, candidate_ids=local_ids, selected_ids=selected_ids,
                    scores=scores, intent_type=intent_type,
                    reason="runtime_no_continuous_trace",
                )
            multi_branch = len(branches) > 1
            steps = tuple(
                PathStep(
                    step=index,
                    branch=(chr(ord("A") + branch_index) if multi_branch else "main"),
                    source_entity=edge.source, relation=edge.relation,
                    target_entity=edge.target, supporting_chunk_id=edge.chunk_id,
                    edge_score=edge.score, query_score=edge.score,
                    canonical_source_entity=edge.canonical_source,
                    canonical_target_entity=edge.canonical_target,
                    traversal_direction=edge.direction,
                )
                for branch_index, chain in enumerate(branches)
                for index, edge in enumerate(chain, start=1)
            )
            anchors = tuple(dict.fromkeys(
                chain[0].source for chain in branches
            ))
            policy = "runtime_executed_frontier_branch_forest"

        core_ids = tuple(dict.fromkeys(step.supporting_chunk_id for step in steps))
        if not set(core_ids).issubset(selected_ids):
            return self._fallback(
                row=row, candidate_ids=local_ids, selected_ids=selected_ids,
                scores=scores, intent_type=intent_type,
                reason="runtime_trace_outside_final_evidence",
            )
        score_order = tuple(sorted(
            selected_ids, key=lambda chunk_id: (-scores.get(chunk_id, 0.0), chunk_id)
        ))
        peripheral_ids = tuple(
            chunk_id for chunk_id in score_order if chunk_id not in core_ids
        )
        # The controlled factorial requires a real ordering intervention.
        # Core sources follow their first occurrence in the frozen runtime
        # path/branch forest; non-core sources retain semantic-score order.
        # This changes only presentation order, never the selected B evidence
        # set. Older v1.3/v1.4 manifests remain score preserving and continue
        # to validate through the version-specific rule in path_manifest.py.
        path_order = core_ids + peripheral_ids
        confidence = sum(step.edge_score for step in steps) / len(steps)
        return FoldManifest(
            query_id=_clean_id(row.get("query_id")), dataset=self.dataset,
            context_id=context_id, question=str(row.get("question", "")),
            intent_type=intent_type, intent_strategy=str(row.get("strategy", "")),
            anchor_entities=anchors, path_policy=policy, path_steps=steps,
            candidate_chunk_ids=tuple(local_ids),
            selected_chunk_ids=tuple(score_order), core_chunk_ids=core_ids,
            peripheral_chunk_ids=peripheral_ids,
            score_order_chunk_ids=score_order, path_order_chunk_ids=path_order,
            chunk_scores=tuple((chunk_id, scores.get(chunk_id, 0.0)) for chunk_id in local_ids),
            budget=self.budget, source_token_budget=self.source_token_budget,
            trace_token_budget=self.trace_token_budget,
            total_evidence_token_budget=self.total_evidence_token_budget,
            foldable=True, fallback_reason="", fallback_to_graph_naive=False,
            path_confidence=confidence, path_continuous=True,
            branch_complete=branch_complete, anchor_source="runtime_trace",
            implementation_version=self.implementation_version,
        ).validate()

    def build_frame(self, reference: pd.DataFrame) -> list[FoldManifest]:
        required = {
            "query_id", "question", "context_id", "strategy",
            "candidate_evidence", "retrieved_evidence", "runtime_path_steps",
        }
        missing = required.difference(reference.columns)
        if missing:
            raise ValueError(
                f"runtime reference is missing columns: {sorted(missing)}"
            )
        if reference["query_id"].astype(str).duplicated().any():
            raise ValueError("runtime reference contains duplicate query_id values")
        return [self.build(row) for row in reference.to_dict(orient="records")]


__all__ = ["RUNTIME_IMPLEMENTATION_VERSION", "RuntimeManifestBuilder"]
