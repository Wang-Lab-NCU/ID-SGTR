"""Controlled P0 replay over the complete Reasoning-Setting local universe."""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
from itertools import combinations
import os
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .evidence import EvidenceVariant
from .telemetry import QueryTelemetry, TrackedChatModel
from .controlled import response_draft
from .path_manifest import FoldManifest, stable_sha256
from .folding_renderers import (
    FOLDING_RENDERER_VERSION,
    FoldingVariant,
    RenderBudgets,
    render_manifest,
)


def _parse_sequence(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple, set)) else []


def _stable_seed(query_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _unit(vector: object) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


def _normalize_short_answer(value: object) -> str:
    """Extract the first concise answer span from a model response."""
    answer = str(value or "").strip()
    if "Final Answer:" in answer:
        answer = answer.split("Final Answer:")[-1].strip()
    answer = answer.splitlines()[0].strip() if answer else ""
    answer = answer.replace("**", "").replace("__", "")
    answer = answer.strip().strip("`").strip('"').strip("'")
    if answer.endswith("."):
        answer = answer[:-1].strip()
    binary = re.match(r"^(yes|no)\b", answer, flags=re.IGNORECASE)
    return binary.group(1).lower() if binary else answer


def _marked_short_answer(value: object) -> str:
    """Accept only an explicit final-answer marker; never expose a rationale."""
    text = str(value or "").strip()
    if "Final Answer:" not in text:
        return ""
    return _normalize_short_answer(text)


def _parse_stage0_decision(value: object) -> tuple[bool, str]:
    """Parse the legacy-compatible Stage-0 ``Final Answer`` early exit."""
    text = str(value or "").strip()
    if "Final Answer:" not in text:
        return False, ""
    answer = _normalize_short_answer(text)
    negative_patterns = (
        "not found", "no information", "information is missing",
        "cannot answer", "unable to answer", "doesn't mention",
        "not provided", "n/a", "not specify", "not specified",
        "need more evidence",
    )
    if not answer or any(pattern in answer.lower() for pattern in negative_patterns):
        return False, ""
    return True, answer


def _shortest_path(adjacency: Mapping[str, set[str]], start: str, goal: str) -> list[str]:
    if start == goal:
        return [start]
    queue: deque[str] = deque([start])
    parent: dict[str, str | None] = {start: None}
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor == goal:
                path = [goal]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(neighbor)
    return []


def _connected(adjacency: Mapping[str, set[str]], chunk_ids: Sequence[str]) -> bool:
    """Return whether the induced chunk subgraph is connected."""
    selected = set(chunk_ids)
    if not selected:
        return False
    start = min(selected)
    seen = {start}
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, set()) & selected:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen == selected


@dataclass(frozen=True)
class LocalCandidate:
    chunk_id: str
    text: str
    title: str
    semantic_score: float
    title_score: float
    structural: bool
    support_count: int
    hop: int
    path_position: int
    topology_trace: tuple[tuple[int, int, str], ...]


class LocalUniverseIndex:
    """Index all local chunks and deterministic graph annotations by context."""

    def __init__(
        self,
        chunks: pd.DataFrame,
        embeddings: pd.DataFrame,
        graph: pd.DataFrame,
        *,
        budget: int = 3,
        seed: int = 42,
        loose_threshold: float | None = None,
        strict_threshold: float | None = None,
    ):
        required_chunks = {"context_id", "chunk_id", "title", "text"}
        required_embeddings = {"context_id", "chunk_id", "embedding", "title_embedding"}
        required_graph = {"context_id", "chunk_id", "node_1", "node_2", "edge"}
        for name, frame, required in (
            ("chunks", chunks, required_chunks),
            ("embeddings", embeddings, required_embeddings),
            ("graph", graph, required_graph),
        ):
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{name} table is missing columns: {sorted(missing)}")
        if budget < 1:
            raise ValueError("budget must be at least 1")

        self.budget = int(budget)
        self.seed = int(seed)
        self.loose_threshold = float(
            os.getenv("ID_SGTR_TAU_LOOSE", "0.25")
            if loose_threshold is None else loose_threshold
        )
        self.strict_threshold = float(
            os.getenv("ID_SGTR_TAU_STRICT", "0.45")
            if strict_threshold is None else strict_threshold
        )
        if self.strict_threshold <= self.loose_threshold:
            raise ValueError("strict threshold must be greater than loose threshold")

        chunk_copy = chunks.copy()
        embedding_copy = embeddings.copy()
        for frame in (chunk_copy, embedding_copy):
            frame["context_id"] = frame["context_id"].astype(str)
            frame["chunk_id"] = frame["chunk_id"].astype(str)
        vector_columns = embedding_copy[["context_id", "chunk_id", "embedding", "title_embedding"]]
        merged = chunk_copy.merge(
            vector_columns,
            on=["context_id", "chunk_id"],
            how="left",
            validate="one_to_one",
        )
        if merged["embedding"].isna().any() or merged["title_embedding"].isna().any():
            missing_count = int(merged["embedding"].isna().sum())
            raise ValueError(f"missing precomputed embeddings for {missing_count} chunks")
        merged["embedding_unit"] = merged["embedding"].map(_unit)
        merged["title_embedding_unit"] = merged["title_embedding"].map(_unit)
        self.context_chunks = {
            context_id: group.reset_index(drop=True)
            for context_id, group in merged.groupby("context_id", sort=False)
        }

        graph_copy = graph.copy()
        graph_copy["context_id"] = graph_copy["context_id"].astype(str)
        graph_copy["chunk_id"] = graph_copy["chunk_id"].astype(str)
        self.relations: dict[str, dict[str, list[tuple[str, str, str]]]] = {}
        self.nodes: dict[str, dict[str, set[str]]] = {}
        for context_id, group in graph_copy.groupby("context_id", sort=False):
            relations: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            nodes: dict[str, set[str]] = defaultdict(set)
            for row in group.itertuples(index=False):
                chunk_id = str(row.chunk_id)
                node_1, node_2, edge = str(row.node_1), str(row.node_2), str(row.edge)
                triple = (node_1, edge, node_2)
                if triple not in relations[chunk_id]:
                    relations[chunk_id].append(triple)
                nodes[chunk_id].update((node_1, node_2))
            self.relations[str(context_id)] = dict(relations)
            self.nodes[str(context_id)] = dict(nodes)

    @staticmethod
    def topology_policy(intent_type: str) -> str:
        return "multi_branch_comparison" if intent_type == "Comparative" else "connected_core"

    def _context_candidates(
        self,
        context_id: str,
        query_vector: Sequence[float],
        intent_type: str = "Reasoning",
    ) -> list[LocalCandidate]:
        context_id = str(context_id)
        if context_id not in self.context_chunks:
            raise ValueError(f"context_id {context_id!r} has no local chunks")
        frame = self.context_chunks[context_id]
        query_unit = _unit(query_vector)
        if not len(query_unit):
            raise ValueError("query embedding is empty")

        semantic = [float(np.dot(query_unit, vector)) for vector in frame["embedding_unit"]]
        title_scores = [float(np.dot(query_unit, vector)) for vector in frame["title_embedding_unit"]]
        chunk_ids = frame["chunk_id"].astype(str).tolist()
        score_by_id = dict(zip(chunk_ids, semantic))

        node_sets = self.nodes.get(context_id, {})
        entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        for chunk_id in chunk_ids:
            for node in node_sets.get(chunk_id, set()):
                entity_to_chunks[node].add(chunk_id)
        adjacency: dict[str, set[str]] = {chunk_id: set() for chunk_id in chunk_ids}
        for linked in entity_to_chunks.values():
            linked_ids = sorted(linked)
            for left in linked_ids:
                adjacency[left].update(right for right in linked_ids if right != left)

        # The core skeleton is the most query-relevant connected chunk subgraph
        # within the evidence budget.  Semantic top-k items are not structural
        # merely because they rank highly: they must have real graph backing.
        loose_ids = [cid for cid in chunk_ids if score_by_id[cid] >= self.loose_threshold]
        structural: set[str] = set()
        best_key: tuple[float, float, int, int, tuple[str, ...]] | None = None
        for size in range(2, min(self.budget, len(loose_ids)) + 1):
            for subset in combinations(sorted(loose_ids), size):
                if not _connected(adjacency, subset):
                    continue
                total_score = sum(score_by_id[cid] for cid in subset)
                internal_edges = sum(
                    1
                    for index, left in enumerate(subset)
                    for right in subset[index + 1:]
                    if right in adjacency.get(left, set())
                )
                # Average relevance prevents a longer, weak distractor component
                # from winning solely because it contains more positive scores.
                key = (
                    total_score / size,
                    total_score,
                    size,
                    internal_edges,
                    tuple(reversed(subset)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    structural = set(subset)

        comparison_branches: set[str] = set()
        if intent_type == "Comparative":
            comparison_branches = set(sorted(
                chunk_ids,
                key=lambda cid: (-score_by_id[cid], cid),
            )[: self.budget])
            structural = comparison_branches

        support: dict[str, int] = {
            cid: (
                1 if comparison_branches
                else len(adjacency.get(cid, set()) & structural)
            )
            for cid in structural
        }
        distance: dict[str, int] = {}
        if structural:
            if comparison_branches:
                distance = {cid: 0 for cid in structural}
            else:
                primary = max(structural, key=lambda cid: (score_by_id[cid], cid))
                distance = {primary: 0}
        queue: deque[str] = deque(distance)
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency.get(current, set()) & structural):
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    queue.append(neighbor)
        topology_order = sorted(
            chunk_ids,
            key=lambda cid: (
                0 if cid in structural else 1,
                distance.get(cid, 10_000),
                -support.get(cid, 0),
                -score_by_id[cid],
                cid,
            ),
        )
        position_by_id = {chunk_id: index for index, chunk_id in enumerate(topology_order, start=1)}

        candidates: list[LocalCandidate] = []
        for row, score, title_score in zip(frame.itertuples(index=False), semantic, title_scores):
            chunk_id = str(row.chunk_id)
            hop = distance.get(chunk_id, 0) + 1 if chunk_id in structural else 0
            position = position_by_id[chunk_id]
            trace = tuple(
                (hop, position, f"{node_1} --[{edge}]--> {node_2}")
                for node_1, edge, node_2 in self.relations.get(context_id, {}).get(chunk_id, [])[:5]
            )
            candidates.append(LocalCandidate(
                chunk_id=chunk_id,
                text=str(row.text),
                title=str(row.title),
                semantic_score=float(score),
                title_score=float(title_score),
                structural=chunk_id in structural,
                support_count=int(support.get(chunk_id, 0)),
                hop=int(hop),
                path_position=int(position),
                topology_trace=trace,
            ))
        return candidates

    def _topology_selection(self, candidates: list[LocalCandidate]) -> list[LocalCandidate]:
        structural = [
            candidate for candidate in candidates
            if candidate.structural and candidate.semantic_score >= self.loose_threshold
        ]
        peripheral = [
            candidate for candidate in candidates
            if not candidate.structural and candidate.semantic_score >= self.strict_threshold
        ]

        def rank(candidate: LocalCandidate) -> tuple[float, int, int, str]:
            # Topology controls eligibility through the loose/strict thresholds.
            # Within the eligible set, semantic relevance remains primary and
            # graph support is only a deterministic tie-breaker.  Adding an
            # unbounded support bonus here can let a weak but central distractor
            # displace a substantially more query-relevant source chunk.
            return (
                -candidate.semantic_score,
                -candidate.support_count,
                candidate.path_position,
                candidate.chunk_id,
            )

        # Preserve the connected core before admitting strictly filtered
        # peripheral chunks.  This is the dual-tier folding rule: the loose
        # threshold is useful only if core evidence cannot be displaced again by
        # isolated semantic distractors.
        selected = sorted(structural, key=rank)[: self.budget]
        if len(selected) < self.budget:
            selected.extend(sorted(peripheral, key=rank)[: self.budget - len(selected)])
        if len(selected) < self.budget:
            selected_ids = {candidate.chunk_id for candidate in selected}
            fillers = sorted(
                (candidate for candidate in candidates if candidate.chunk_id not in selected_ids),
                key=lambda candidate: (-candidate.semantic_score, candidate.chunk_id),
            )
            selected.extend(fillers[: self.budget - len(selected)])
        return sorted(selected, key=lambda candidate: (candidate.path_position, candidate.chunk_id))

    def select(
        self,
        context_id: str,
        query_vector: Sequence[float],
        variant: EvidenceVariant | str,
        *,
        query_id: str,
        intent_type: str = "Reasoning",
        gold_evidence: object = None,
    ) -> tuple[list[LocalCandidate], list[LocalCandidate]]:
        variant = EvidenceVariant(variant)
        candidates = self._context_candidates(context_id, query_vector, intent_type)
        by_id = {candidate.chunk_id: candidate for candidate in candidates}

        if variant is EvidenceVariant.ORACLE:
            selected = [
                by_id[str(chunk_id)]
                for chunk_id in _parse_sequence(gold_evidence)
                if str(chunk_id) in by_id
            ][: self.budget]
        elif variant is EvidenceVariant.ENTITY_TO_CHUNK:
            selected = sorted(candidates, key=lambda item: (-item.title_score, item.chunk_id))[: self.budget]
        elif variant in (
            EvidenceVariant.TRIPLE_ONLY,
            EvidenceVariant.GRAPH_NAIVE,
            EvidenceVariant.TOPOLOGY_FOLDING,
        ):
            selected = self._topology_selection(candidates)
            if variant is EvidenceVariant.GRAPH_NAIVE:
                selected = sorted(
                    selected,
                    key=lambda item: (-item.semantic_score, item.chunk_id),
                )
        else:
            selected = sorted(candidates, key=lambda item: (-item.semantic_score, item.chunk_id))[: self.budget]
            if variant is EvidenceVariant.SOURCE_RANDOM:
                random.Random(_stable_seed(query_id, self.seed)).shuffle(selected)
        return candidates, selected

    @staticmethod
    def render(selected: Sequence[LocalCandidate], variant: EvidenceVariant | str, char_limit: int = 1000) -> str:
        variant = EvidenceVariant(variant)
        rendered: list[str] = []
        for candidate in selected:
            trace_lines = [
                f"[Hop {hop} Path {path}] {triple}"
                for hop, path, triple in candidate.topology_trace
                if triple.strip()
            ]
            if variant is EvidenceVariant.TRIPLE_ONLY:
                rendered.append("\n".join(trace_lines) or "(source relation unavailable)")
            elif variant is EvidenceVariant.TOPOLOGY_FOLDING:
                relations = "\n".join(trace_lines) or "(source relation unavailable)"
                rendered.append(
                    f"{relations}\n[Source {candidate.chunk_id}] "
                    f"{' '.join(candidate.text.split())[:char_limit]}"
                )
            else:
                rendered.append(
                    f"[Ref {candidate.chunk_id}] {' '.join(candidate.text.split())[:char_limit]}"
                )
        return "\n".join(rendered) if rendered else "No evidence available."

    @staticmethod
    def manifest(candidates: Sequence[LocalCandidate]) -> list[dict[str, object]]:
        return [{
            "chunk_id": candidate.chunk_id,
            "score": candidate.semantic_score,
            "title_score": candidate.title_score,
            "hop": candidate.hop,
            "path_position": candidate.path_position,
            "topology_position": candidate.path_position,
            "triple": candidate.topology_trace[0][2] if candidate.topology_trace else "",
            "topology_trace": [
                {"hop": hop, "path_position": path, "triple": triple}
                for hop, path, triple in candidate.topology_trace
            ],
            "is_structural": candidate.structural,
            "support_count": candidate.support_count,
        } for candidate in candidates]


class LocalControlledRunner:
    """Run one local-universe variant with a fixed two-stage answer protocol."""

    def __init__(self, reasoner: Any, formatter: Any, index: LocalUniverseIndex):
        self.reasoner = reasoner
        self.formatter = formatter
        self.index = index

    def _answer(self, question: str, evidence: str, telemetry: QueryTelemetry) -> str:
        """Run the fixed reasoning-plus-formatting answer protocol."""
        reasoning_prompt = f"""Solve the multi-hop question using ONLY the fixed evidence below.

Follow every relation required by the question. For comparisons, resolve the
requested attribute for both alternatives. Think carefully and end with a
short Final Answer.

Question: {question}

Evidence:
{evidence}

End with: Final Answer: <shortest answer span>"""
        telemetry.reasoning_prompt_sha256 = hashlib.sha256(
            reasoning_prompt.encode("utf-8")
        ).hexdigest()
        reasoning_response = TrackedChatModel(
            self.reasoner, telemetry, "answer"
        ).invoke(reasoning_prompt)
        draft = response_draft(reasoning_response)
        if os.getenv("ID_SGTR_CODE_FINALIZE", "false").strip().lower() in {
            "1", "true", "yes",
        }:
            telemetry.finalization_policy = "strict_code_final_answer_marker"
            telemetry.formatter_prompt_sha256 = ""
            return _marked_short_answer(draft)
        formatter_prompt = f"""Normalize the answer using the question, evidence, and reasoning draft.

Question: {question}

Evidence:
{evidence}

Reasoning draft:
{draft or 'No reasoning draft was returned.'}

Final Answer: <shortest answer span>

Return exactly one line. The span must be only the entity, place, date,
number, yes, or no. Do not return reasoning, a sentence, an intermediate
entity, or a different attribute."""
        telemetry.formatter_prompt_sha256 = hashlib.sha256(
            formatter_prompt.encode("utf-8")
        ).hexdigest()
        response = TrackedChatModel(
            self.formatter, telemetry, "answer"
        ).invoke(formatter_prompt)
        telemetry.finalization_policy = "llm_formatter"
        return _normalize_short_answer(getattr(response, "content", response))

    def run_row(
        self,
        row: Mapping[str, object],
        query_vector: Sequence[float],
        variant: EvidenceVariant | str,
        intent_type: str = "Reasoning",
        intent_strategy: str = "",
    ) -> dict[str, object]:
        variant = EvidenceVariant(variant)
        query_id = str(row.get("query_id", ""))
        question = str(row.get("question", ""))
        context_id = str(row.get("context_id", ""))
        telemetry = QueryTelemetry(query_id=query_id)
        telemetry.start()

        candidates, selected = self.index.select(
            context_id,
            query_vector,
            variant,
            query_id=query_id,
            intent_type=intent_type,
            gold_evidence=row.get("gold_evidence"),
        )
        evidence = self.index.render(selected, variant)
        answer = self._answer(question, evidence, telemetry)
        telemetry.retrieved_evidence = [candidate.chunk_id for candidate in selected]
        telemetry.accessed_evidence = list(telemetry.retrieved_evidence)
        telemetry.candidate_evidence = self.index.manifest(candidates)
        telemetry.stop()

        result = dict(row)
        result.update({
            "gold_answer": row.get("gold_answer", row.get("answer", "")),
            "pred_answer": answer,
            "evidence_variant": variant.value,
            "evidence_budget": self.index.budget,
            "local_universe_size": len(candidates),
            "intent_type": intent_type,
            "intent_strategy": intent_strategy,
            "topology_policy": self.index.topology_policy(intent_type),
            "fixed_candidate_count": len(candidates),
            "selected_evidence_set": sorted(candidate.chunk_id for candidate in selected),
            "accessed_evidence": list(telemetry.retrieved_evidence),
            **telemetry.to_dict(),
        })
        return result


class LocalControlledV2Runner:
    """Replay a frozen :class:`FoldManifest` without retrieval or reselection.

    The runner deliberately accepts neither a query vector nor a graph.  Gold
    annotations may remain in the output row for evaluation, but they are not
    passed to the renderer and cannot affect evidence selection.
    """

    def __init__(
        self,
        reasoner: Any,
        formatter: Any,
        chunks: pd.DataFrame,
        *,
        budgets: RenderBudgets | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        required = {"context_id", "chunk_id", "text"}
        missing = required.difference(chunks.columns)
        if missing:
            raise ValueError(f"chunks table is missing columns: {sorted(missing)}")
        chunk_copy = chunks.copy()
        chunk_copy["context_id"] = chunk_copy["context_id"].astype(str)
        chunk_copy["chunk_id"] = chunk_copy["chunk_id"].astype(str)
        duplicate = chunk_copy.duplicated(["context_id", "chunk_id"])
        if duplicate.any():
            raise ValueError("chunks contains duplicate (context_id, chunk_id) rows")
        self.source_by_key = {
            (str(row.context_id), str(row.chunk_id)): str(row.text)
            for row in chunk_copy.itertuples(index=False)
        }
        self.reasoner = reasoner
        self.formatter = formatter
        self.budgets = budgets
        self.tokenizer = tokenizer

    def _answer(self, question: str, evidence: str, telemetry: QueryTelemetry) -> str:
        reasoning_prompt = f"""Solve the multi-hop question using ONLY the fixed evidence below.

Follow every relation required by the question. For comparisons, resolve the
requested attribute for both alternatives. Think carefully and end with a
short Final Answer.

Source passages are the authoritative evidence. An optional
[Advisory Topology] block is incomplete navigation metadata: use it only to
locate relations across sources, never as a substitute for the passages, and
ignore it if it conflicts with or omits information from a source passage.

Question: {question}

Evidence:
{evidence}

End with: Final Answer: <shortest answer span>"""
        telemetry.reasoning_prompt_sha256 = hashlib.sha256(
            reasoning_prompt.encode("utf-8")
        ).hexdigest()
        reasoning_response = TrackedChatModel(
            self.reasoner, telemetry, "answer"
        ).invoke(reasoning_prompt)
        draft = response_draft(reasoning_response)
        if os.getenv("ID_SGTR_CODE_FINALIZE", "false").strip().lower() in {
            "1", "true", "yes",
        }:
            telemetry.finalization_policy = "strict_code_final_answer_marker"
            telemetry.formatter_prompt_sha256 = ""
            return _marked_short_answer(draft)
        formatter_prompt = f"""Normalize the answer using the question, evidence, and reasoning draft.

Question: {question}

Evidence:
{evidence}

Reasoning draft:
{draft or 'No reasoning draft was returned.'}

Final Answer: <shortest answer span>

Return exactly one line. The span must be only the entity, place, date,
number, yes, or no. Do not return reasoning, a sentence, an intermediate
entity, or a different attribute."""
        telemetry.formatter_prompt_sha256 = hashlib.sha256(
            formatter_prompt.encode("utf-8")
        ).hexdigest()
        response = TrackedChatModel(
            self.formatter, telemetry, "answer"
        ).invoke(formatter_prompt)
        telemetry.finalization_policy = "llm_formatter"
        return _normalize_short_answer(getattr(response, "content", response))

    def _sources(self, manifest: FoldManifest) -> dict[str, str]:
        sources: dict[str, str] = {}
        for chunk_id in manifest.selected_chunk_ids:
            key = (manifest.context_id, str(chunk_id))
            if key not in self.source_by_key:
                raise ValueError(
                    f"manifest {manifest.query_id!r} references missing chunk {key!r}"
                )
            sources[str(chunk_id)] = self.source_by_key[key]
        return sources

    @staticmethod
    def _candidate_manifest(manifest: FoldManifest) -> list[dict[str, object]]:
        score_by_id = dict(manifest.chunk_scores)
        step_by_chunk: dict[str, list[dict[str, object]]] = defaultdict(list)
        for step in manifest.path_steps:
            step_by_chunk[step.supporting_chunk_id].append(step.to_dict())
        return [{
            "chunk_id": chunk_id,
            "score": float(score_by_id.get(chunk_id, 0.0)),
            "selected": chunk_id in set(manifest.selected_chunk_ids),
            "is_core": chunk_id in set(manifest.core_chunk_ids),
            "is_peripheral": chunk_id in set(manifest.peripheral_chunk_ids),
            "aligned_path_steps": step_by_chunk.get(chunk_id, []),
        } for chunk_id in manifest.candidate_chunk_ids]

    def run_row(
        self,
        row: Mapping[str, object],
        manifest: FoldManifest,
        variant: FoldingVariant | str,
    ) -> dict[str, object]:
        manifest.validate()
        if self.budgets is not None:
            frozen_budgets = RenderBudgets.from_manifest(manifest)
            if self.budgets != frozen_budgets:
                raise ValueError(
                    "renderer budgets must match the frozen manifest: "
                    f"requested={self.budgets}, manifest={frozen_budgets}"
                )
        query_id = str(row.get("query_id", ""))
        question = str(row.get("question", "")).strip()
        context_id = str(row.get("context_id", ""))
        if query_id != manifest.query_id:
            raise ValueError(
                f"row query_id {query_id!r} does not match manifest {manifest.query_id!r}"
            )
        if context_id != manifest.context_id:
            raise ValueError(
                f"row context_id {context_id!r} does not match manifest {manifest.context_id!r}"
            )
        if question != manifest.question:
            raise ValueError(f"question mismatch for manifest {manifest.query_id!r}")

        telemetry = QueryTelemetry(query_id=query_id)
        telemetry.start()
        with telemetry.phase("retrieval"):
            rendered = render_manifest(
                manifest,
                self._sources(manifest),
                variant,
                budgets=self.budgets,
                tokenizer=self.tokenizer,
            )
        answer = self._answer(question, rendered.text, telemetry)
        telemetry.retrieved_evidence = list(rendered.chunk_ids)
        telemetry.accessed_evidence = list(rendered.chunk_ids)
        telemetry.candidate_evidence = self._candidate_manifest(manifest)
        telemetry.manifest_version = manifest.manifest_version
        telemetry.implementation_version = manifest.implementation_version
        telemetry.manifest_sha256 = manifest.sha256
        telemetry.path_policy = manifest.path_policy
        telemetry.path_length = manifest.path_length
        telemetry.path_continuous = manifest.path_continuous
        telemetry.path_confidence = manifest.path_confidence
        telemetry.foldable = manifest.foldable
        telemetry.folding_fallback = rendered.folding_fallback
        telemetry.folding_fallback_reason = rendered.folding_fallback_reason
        telemetry.selected_count = manifest.selected_count
        telemetry.unused_budget = manifest.unused_budget
        telemetry.core_chunk_ids = list(manifest.core_chunk_ids)
        telemetry.peripheral_chunk_ids = list(manifest.peripheral_chunk_ids)
        telemetry.trace_count = rendered.trace_count
        telemetry.trace_tokens = rendered.trace_tokens
        telemetry.source_tokens = rendered.source_tokens
        telemetry.evidence_tokens = rendered.total_tokens
        telemetry.order_changed = rendered.order_changed
        telemetry.manifest_order_changed = (
            tuple(manifest.path_order_chunk_ids)
            != tuple(manifest.score_order_chunk_ids)
        )
        telemetry.branch_complete = manifest.branch_complete
        telemetry.anchor_margin = manifest.anchor_margin
        telemetry.anchor_source = manifest.anchor_source
        telemetry.llm_reranker_used = manifest.llm_reranker_used
        telemetry.stop()

        source_hashes = {
            chunk_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for chunk_id, text in rendered.source_fragments
        }
        result = dict(row)
        result.update({
            "gold_answer": row.get("gold_answer", row.get("answer", "")),
            "pred_answer": answer,
            "evidence_variant": FoldingVariant(variant).value,
            "evidence_budget": manifest.budget,
            "local_universe_size": len(manifest.candidate_chunk_ids),
            "intent_type": manifest.intent_type,
            "intent_strategy": manifest.intent_strategy,
            "topology_policy": manifest.path_policy,
            "fixed_candidate_count": len(manifest.candidate_chunk_ids),
            "selected_evidence_set": sorted(manifest.selected_chunk_ids),
            "rendered_evidence_order": list(rendered.chunk_ids),
            "source_fragment_sha256": source_hashes,
            "source_text_sha256": stable_sha256(source_hashes),
            "folding_renderer_version": FOLDING_RENDERER_VERSION,
            "rendered_evidence_sha256": hashlib.sha256(
                rendered.text.encode("utf-8")
            ).hexdigest(),
            "request_seed": int(os.getenv("ID_SGTR_SEED", "42")),
            "token_counter": rendered.token_counter_name,
            "accessed_evidence": list(rendered.chunk_ids),
            **telemetry.to_dict(),
        })
        return result


class LocalControlled333Runner(LocalControlledRunner):
    """Controlled B=(3,3,3) replay with a legacy-style Stage-0 early exit."""

    budget = 3

    def _stage0_prompt(
        self,
        question: str,
        selected: Sequence[LocalCandidate],
    ) -> str:
        definitions = "\n".join(
            f"- {candidate.title}" for candidate in selected
        ) or "None"
        context = self.index.render(selected, EvidenceVariant.SOURCE_SCORE)
        return f"""You are a Fact-Checking & Answer Extraction Agent. Answer the query immediately only if every relation needed by the question is supported by the evidence.

### User Query
\"{question}\"

### Entity Definitions
{definitions}

### Source Context
{context}

### Decision Logic (STRICT)
- If the complete answer can be derived, return exactly:
  Final Answer: <shortest answer span>
- If any required relation or comparison branch is missing, return exactly:
  Need More Evidence

Do not guess. Do not explain."""

    def run_row(
        self,
        row: Mapping[str, object],
        query_vector: Sequence[float],
        variant: EvidenceVariant | str = EvidenceVariant.TOPOLOGY_FOLDING,
        intent_type: str = "Reasoning",
        intent_strategy: str = "",
    ) -> dict[str, object]:
        variant = EvidenceVariant(variant)
        if variant is not EvidenceVariant.TOPOLOGY_FOLDING:
            raise ValueError("controlled-local-333 only supports topology_folding")

        query_id = str(row.get("query_id", ""))
        question = str(row.get("question", ""))
        context_id = str(row.get("context_id", ""))
        telemetry = QueryTelemetry(query_id=query_id)
        telemetry.start()

        candidates, stage0_selected = self.index.select(
            context_id,
            query_vector,
            EvidenceVariant.ENTITY_TO_CHUNK,
            query_id=query_id,
            intent_type=intent_type,
            gold_evidence=row.get("gold_evidence"),
        )
        stage0_selected = stage0_selected[: self.budget]
        stage0_ids = [candidate.chunk_id for candidate in stage0_selected]
        stage0_response = TrackedChatModel(
            self.reasoner, telemetry, "answer"
        ).invoke(self._stage0_prompt(question, stage0_selected))
        stage0_text = str(
            getattr(stage0_response, "content", stage0_response) or ""
        ).strip()
        stage0_is_final, stage0_answer = _parse_stage0_decision(stage0_text)

        if stage0_is_final:
            selected = stage0_selected
            answer = stage0_answer
            route = "Agent-Zero-Shot"
            telemetry.retrieval_rounds = 1
        else:
            candidates, selected = self.index.select(
                context_id,
                query_vector,
                EvidenceVariant.TOPOLOGY_FOLDING,
                query_id=query_id,
                intent_type=intent_type,
                gold_evidence=row.get("gold_evidence"),
            )
            evidence = self.index.render(selected, EvidenceVariant.TOPOLOGY_FOLDING)
            answer = self._answer(question, evidence, telemetry)
            route = "Topology-Final-Synthesis"
            telemetry.retrieval_rounds = 2

        selected_ids = [candidate.chunk_id for candidate in selected]
        accessed_ids = list(dict.fromkeys(stage0_ids + selected_ids))
        telemetry.retrieved_evidence = selected_ids
        telemetry.accessed_evidence = accessed_ids
        telemetry.candidate_evidence = self.index.manifest(candidates)
        telemetry.stop()

        strategy = f"{intent_strategy} -> {route}" if intent_strategy else route
        result = dict(row)
        result.update({
            "gold_answer": row.get("gold_answer", row.get("answer", "")),
            "pred_answer": answer,
            "strategy": strategy,
            "evidence_variant": EvidenceVariant.TOPOLOGY_FOLDING.value,
            "evidence_budget": self.budget,
            "stage0_budget": self.budget,
            "path_budget": self.budget,
            "final_budget": self.budget,
            "stage0_selector": EvidenceVariant.ENTITY_TO_CHUNK.value,
            "stage0_is_final": stage0_is_final,
            "stage0_answer": stage0_answer,
            "stage0_raw_response": stage0_text,
            "stage0_evidence": stage0_ids,
            "folded_evidence": selected_ids if not stage0_is_final else [],
            "local_universe_size": len(candidates),
            "intent_type": intent_type,
            "intent_strategy": intent_strategy,
            "topology_policy": self.index.topology_policy(intent_type),
            "fixed_candidate_count": len(candidates),
            "selected_evidence_set": sorted(selected_ids),
            "accessed_evidence": accessed_ids,
            **telemetry.to_dict(),
        })
        return result
