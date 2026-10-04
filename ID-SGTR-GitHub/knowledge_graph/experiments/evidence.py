"""Evidence assembly variants used by the controlled Topology Folding ablation.

All variants consume the same candidate items and budget.  Only their textual
representation or ordering changes, which prevents retrieval-path changes from
confounding the folding ablation.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class EvidenceVariant(str, Enum):
    TRIPLE_ONLY = "triple_only"
    SOURCE_SCORE = "source_score"
    SOURCE_RANDOM = "source_random"
    GRAPH_NAIVE = "graph_naive"
    PATH_ORDER_SOURCE = "path_order_source"
    TRACE_SCORE_SOURCE = "trace_score_source"
    TOPOLOGY_FOLDING_V2 = "topology_folding_v2"
    LEGACY_TOPOLOGY_FOLDING = "legacy_topology_folding"
    TOPOLOGY_FOLDING = "topology_folding"
    ENTITY_TO_CHUNK = "entity_to_chunk"
    ORACLE = "oracle"


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    text: str
    score: float = 0.0
    path_position: int | None = None
    triple: str = ""
    is_gold: bool = False
    # Legacy controlled replay records these fields in candidate manifests.
    # They remain optional here; Topology Folding v2 uses PathStep instead.
    hop: int = 0
    topology_trace: tuple[tuple[int, int, str], ...] = ()


def _stable_seed(query_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class EvidenceAssembler:
    """Deduplicate, order, and truncate evidence under a shared chunk budget."""

    def __init__(
        self,
        budget: int = 3,
        random_seed: int = 42,
        topology_support_weight: float | None = None,
    ):
        if budget < 1:
            raise ValueError("budget must be at least 1")
        self.budget = budget
        self.random_seed = random_seed
        self.topology_support_weight = float(
            os.getenv("ID_SGTR_TOPOLOGY_SUPPORT_WEIGHT", "0")
            if topology_support_weight is None else topology_support_weight
        )
        if self.topology_support_weight < 0:
            raise ValueError("topology_support_weight must be non-negative")

    @staticmethod
    def _trace(item: EvidenceItem) -> tuple[tuple[int, int, str], ...]:
        trace: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int, str]] = set()
        for raw in item.topology_trace:
            value = (int(raw[0]), int(raw[1]), str(raw[2]))
            if value[2].strip() and value not in seen:
                trace.append(value)
                seen.add(value)
        if item.triple.strip():
            value = (
                int(item.hop), int(item.path_position or 0), str(item.triple),
            )
            if value not in seen:
                trace.append(value)
        return tuple(trace)

    @staticmethod
    def _deduplicate(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
        best: dict[str, EvidenceItem] = {}
        order: list[str] = []
        for raw in items:
            item = EvidenceItem(
                chunk_id=str(raw.chunk_id), text=str(raw.text), score=float(raw.score),
                path_position=raw.path_position, triple=str(raw.triple), is_gold=bool(raw.is_gold),
                hop=int(raw.hop), topology_trace=EvidenceAssembler._trace(raw),
            )
            if item.chunk_id not in best:
                order.append(item.chunk_id)
                best[item.chunk_id] = item
            else:
                current = best[item.chunk_id]
                primary = item if item.score > current.score else current
                merged_trace: list[tuple[int, int, str]] = []
                seen_trace: set[tuple[int, int, str]] = set()
                for trace in current.topology_trace + item.topology_trace:
                    if trace not in seen_trace:
                        merged_trace.append(trace)
                        seen_trace.add(trace)
                positive_hops = [
                    hop for hop, _, _ in merged_trace if int(hop) > 0
                ]
                best[item.chunk_id] = EvidenceItem(
                    chunk_id=primary.chunk_id,
                    text=primary.text,
                    score=max(current.score, item.score),
                    path_position=primary.path_position,
                    triple=primary.triple,
                    is_gold=current.is_gold or item.is_gold,
                    hop=min(positive_hops) if positive_hops else primary.hop,
                    topology_trace=tuple(merged_trace),
                )
        return [best[chunk_id] for chunk_id in order]

    @staticmethod
    def _support_count(item: EvidenceItem) -> int:
        hops = {
            int(hop) for hop, _, _ in item.topology_trace if int(hop) > 0
        }
        if item.hop > 0:
            hops.add(int(item.hop))
        return len(hops)

    @staticmethod
    def _trace_entities(item: EvidenceItem) -> set[str]:
        entities: set[str] = set()
        for _, _, triple in EvidenceAssembler._trace(item):
            match = re.match(r"\s*(.*?)\s*--\[.*?\]-->\s*(.*?)\s*$", triple)
            if match:
                entities.update(
                    value.strip().casefold()
                    for value in match.groups()
                    if value.strip()
                )
        return entities

    @staticmethod
    def _is_comparative(query: str) -> bool:
        return bool(re.search(
            r"\b(same|different|both|which|more|less|older|younger|earlier|later)\b",
            str(query).casefold(),
        ))

    def _path_constrained_budget(
        self,
        items: list[EvidenceItem],
        *,
        query: str,
    ) -> list[EvidenceItem]:
        """Greedily combine relevance with path/branch coverage under B.

        The Cross-Encoder score remains the dominant signal.  Connectivity is
        a bounded tie-breaking bonus, while comparative questions additionally
        reward a new disconnected branch so both compared subjects can survive
        a three-chunk budget.
        """
        if not items:
            return []
        path_bonus = max(
            0.0, float(os.getenv("ID_SGTR_RERANK_PATH_BONUS", "0.12"))
        )
        branch_bonus = max(
            0.0, float(os.getenv("ID_SGTR_RERANK_BRANCH_BONUS", "0.10"))
        )
        structural_bonus = max(
            0.0, float(os.getenv("ID_SGTR_RERANK_STRUCTURAL_BONUS", "0.04"))
        )
        comparative = self._is_comparative(query)
        remaining = list(items)
        selected: list[EvidenceItem] = []
        selected_entities: set[str] = set()

        while remaining and len(selected) < self.budget:
            scored: list[tuple[float, float, int, str, EvidenceItem]] = []
            for item in remaining:
                entities = self._trace_entities(item)
                connected = bool(entities and selected_entities.intersection(entities))
                new_branch = bool(
                    comparative and selected_entities and entities
                    and not selected_entities.intersection(entities)
                )
                adjusted = (
                    item.score
                    + self.topology_support_weight * self._support_count(item)
                    + (structural_bonus if entities else 0.0)
                    + (path_bonus if connected else 0.0)
                    + (branch_bonus if new_branch else 0.0)
                )
                scored.append((
                    adjusted,
                    item.score,
                    self._support_count(item),
                    item.chunk_id,
                    item,
                ))
            scored.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
            chosen = scored[0][-1]
            selected.append(chosen)
            selected_entities.update(self._trace_entities(chosen))
            remaining = [
                item for item in remaining if item.chunk_id != chosen.chunk_id
            ]
        return selected

    def _freeze_budget(
        self,
        items: list[EvidenceItem],
        *,
        query: str = "",
    ) -> list[EvidenceItem]:
        """Choose one common evidence set before any representation ordering."""

        from .pcef_v2 import enabled, select_set
        if enabled():
            return select_set(items, query, self.budget)

        if os.getenv(
            "ID_SGTR_PATH_CONSTRAINED_SELECTION", ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            return self._path_constrained_budget(items, query=query)

        ranked = sorted(
            items,
            key=lambda item: (
                -(
                    item.score
                    + self.topology_support_weight * self._support_count(item)
                ),
                -item.score,
                -self._support_count(item),
                item.chunk_id,
            ),
        )
        return ranked[: self.budget]

    def select(
        self,
        items: Iterable[EvidenceItem],
        variant: EvidenceVariant | str,
        *,
        query_id: str = "",
    ) -> list[EvidenceItem]:
        variant = EvidenceVariant(variant)
        if variant in {
            EvidenceVariant.PATH_ORDER_SOURCE,
            EvidenceVariant.TRACE_SCORE_SOURCE,
            EvidenceVariant.TOPOLOGY_FOLDING_V2,
            EvidenceVariant.LEGACY_TOPOLOGY_FOLDING,
        }:
            raise ValueError(
                f"{variant.value} must be rendered by folding_renderers.py"
            )
        from .pcef_v2 import enabled, latest_scores
        if enabled():
            items = latest_scores(list(items))
        selected = self._deduplicate(items)

        if variant is EvidenceVariant.ORACLE:
            selected = [item for item in selected if item.is_gold]
            selected.sort(key=lambda x: (x.path_position is None, x.path_position or 0, -x.score, x.chunk_id))
            return selected[: self.budget]
        selected = self._freeze_budget(selected, query=query_id)
        if variant is EvidenceVariant.SOURCE_RANDOM:
            random.Random(_stable_seed(query_id, self.random_seed)).shuffle(selected)
        elif variant in (
            EvidenceVariant.SOURCE_SCORE,
            EvidenceVariant.GRAPH_NAIVE,
            EvidenceVariant.ENTITY_TO_CHUNK,
        ):
            selected.sort(key=lambda x: (-x.score, x.chunk_id))
        else:
            # Triple-only and Topology Folding preserve the graph-path order.
            selected.sort(key=lambda x: (
                x.hop <= 0,
                x.hop if x.hop > 0 else 0,
                x.path_position is None,
                x.path_position or 0,
                -x.score,
                x.chunk_id,
            ))
        return selected

    def render(
        self,
        items: Iterable[EvidenceItem],
        variant: EvidenceVariant | str,
        *,
        query_id: str = "",
        char_limit: int = 1000,
    ) -> tuple[list[str], list[str]]:
        variant = EvidenceVariant(variant)
        selected = self.select(items, variant, query_id=query_id)
        rendered: list[str] = []
        for item in selected:
            trace_lines = [
                (
                    f"[Hop {hop} Path {path}] {triple}"
                    if hop > 0 else f"[Path {path}] {triple}"
                )
                for hop, path, triple in item.topology_trace
                if triple.strip()
            ]
            if variant is EvidenceVariant.TRIPLE_ONLY:
                rendered.append(
                    "\n".join(trace_lines) or "(source relation unavailable)"
                )
            elif variant is EvidenceVariant.TOPOLOGY_FOLDING:
                relation = "\n".join(trace_lines) or "(source relation unavailable)"
                text = " ".join(item.text.split())[:char_limit]
                rendered.append(
                    f"{relation}\n"
                    f"[Source {item.chunk_id}] {text}"
                )
            else:
                text = " ".join(item.text.split())[:char_limit]
                rendered.append(f"[Ref {item.chunk_id}] {text}")
        return rendered, [item.chunk_id for item in selected]


def make_items(
    chunk_ids: Sequence[str],
    texts: Sequence[str],
    scores: Sequence[float] | None = None,
) -> list[EvidenceItem]:
    """Convenience adapter for legacy retrieval code."""
    if len(chunk_ids) != len(texts):
        raise ValueError("chunk_ids and texts must have equal length")
    scores = scores if scores is not None else [0.0] * len(chunk_ids)
    if len(scores) != len(chunk_ids):
        raise ValueError("scores and chunk_ids must have equal length")
    return [
        EvidenceItem(str(cid), str(text), float(score), position)
        for position, (cid, text, score) in enumerate(zip(chunk_ids, texts, scores), start=1)
    ]
