"""Query-conditioned path construction for controlled Topology Folding v2.

This module deliberately separates evidence *selection* from evidence
*rendering*.  It builds one immutable :class:`FoldManifest` per query from the
complete Reasoning-Setting context.  Gold answers and gold supporting chunks
are not accepted by the public API.

The legacy P0 implementation connected chunks whenever they shared any
entity.  In contrast, this implementation searches the directed
entity-relation graph, maps only the traversed relations back to their source
chunks, and applies the loose/strict thresholds after the path skeleton has
been constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .path_manifest import FoldManifest, PathStep


# Unicode-aware words are essential for 2Wiki entities such as Ælfgar,
# Adèle, Małgorzata, and Żuławski.  The former ASCII-only tokenizer silently
# damaged exact anchors and produced avoidable no_reliable_query_anchor cases.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "to",
    "was", "were", "what", "when", "where", "which", "who", "whose",
    "with", "same", "both", "than", "into", "film", "name",
})
_GENERIC_ENTITIES = frozenset({
    "actor", "actress", "album", "author", "band", "book", "city",
    "company", "country", "director", "film", "government", "group",
    "location", "man", "movie", "music", "person", "place", "school",
    "song", "state", "university", "woman", "writer", "year",
})

# Query relation slots are deliberately small and interpretable.  They do not
# encode answers or dataset IDs; they only connect common question predicates
# to paraphrases found in extracted graph relations.  Beam search rewards a
# slot once and penalizes repeatedly following an already satisfied slot.
_QUERY_CUE_ALIASES: dict[str, tuple[str, ...]] = {
    "nationality": ("nationality", "national", "citizenship"),
    "mother": ("mother", "maternal"),
    "father": ("father", "paternal"),
    "spouse": ("spouse", "wife", "husband", "married", "partner"),
    "director": ("director", "directed"),
    "performer": (
        "performer", "performed", "singer", "musician", "artist",
        "actor", "actress", "star", "stars", "starred", "plays", "played",
        "featured", "features",
    ),
    "author": ("author", "wrote", "writer", "written"),
    "composer": ("composer", "composed"),
    "location": ("located", "location", "where", "place"),
    "birth": ("born", "birth", "birthplace"),
    "death": ("died", "death"),
    "release": ("released", "release", "came out", "premiered"),
    "language": ("language", "libretto"),
    "profession": ("profession", "occupation", "job"),
    "name": ("middle name", "full name", "birth name"),
    "count": ("how many", "number of"),
    "award": ("award", "oscar", "tony", "grammy", "prize"),
    "publisher": ("publisher", "published", "publishes"),
    "waterway": ("river", "tributary"),
    "date": ("when", "date", "year", "earlier", "later", "older", "younger"),
}
_EDGE_CUE_ALIASES: dict[str, tuple[str, ...]] = {
    **_QUERY_CUE_ALIASES,
    "nationality": (
        "nationality", "national", "citizen", "american", "british",
        "english", "french", "german", "italian", "spanish", "polish",
        "russian", "canadian", "australian", "irish", "japanese",
        "chinese", "indian", "swedish", "norwegian", "danish", "dutch",
    ),
    # Extracted graphs frequently encode a parent relation from the child's
    # perspective ("X is the son of actress Y").  The gender-bearing noun is
    # part of the predicate text, so it is safe to use it as a relation cue;
    # no answer name or dataset-specific identifier is consulted.
    "mother": (
        "mother", "maternal", "daughter of actress", "son of actress",
        "daughter of her", "son of her",
    ),
    "father": (
        "father", "paternal", "daughter of actor", "son of actor",
        "daughter of his", "son of his",
    ),
    "spouse": ("spouse", "wife", "husband", "married", "partner"),
    "performer": (
        "performer", "performed", "singer", "vocalist", "musician",
        "guitarist", "artist", "actor", "actress", "star", "stars",
        "starred", "featured", "features",
    ),
    "death": ("died", "death", "date of death"),
    "release": (
        "released", "release", "came out", "premiered", "publication date",
    ),
    "language": ("language", "libretto"),
    "profession": (
        "profession", "occupation", "writer", "director", "producer",
        "actor", "actress",
    ),
    "name": ("middle name", "full name", "birth name", "name"),
    "count": ("number", "how many", "yard", "yards", "acts"),
    "award": ("award", "oscar", "tony", "grammy", "prize"),
    "publisher": ("publisher", "published", "publishes", "publication"),
    "waterway": ("river", "tributary", "stream", "flows", "watercourse"),
    "date": (
        "when", "date", "year", "born", "birth", "died", "death",
        "released", "release", "premiered",
    ),
}


def _tokens(value: object, *, drop_stopwords: bool = True) -> tuple[str, ...]:
    values = tuple(token.casefold() for token in _TOKEN_RE.findall(str(value or "")))
    if drop_stopwords:
        values = tuple(token for token in values if token not in _STOPWORDS)
    return values


def _normalized_phrase(value: object) -> str:
    return " ".join(_tokens(value, drop_stopwords=False))


def _unit(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _contains_alias(text: str, aliases: Sequence[str]) -> bool:
    normalized = f" {_normalized_phrase(text)} "
    return any(f" {_normalized_phrase(alias)} " in normalized for alias in aliases)


def _query_relation_cues(question: str) -> tuple[str, ...]:
    return tuple(
        cue for cue, aliases in _QUERY_CUE_ALIASES.items()
        if _contains_alias(question, aliases)
    )


def _relation_slot_groups(question: str) -> tuple[frozenset[str], ...]:
    """Return semantic relation slots after collapsing surface duplicates.

    Interrogative location/date words often describe the endpoint of a birth,
    death, or release relation.  Counting both surface cues as independent
    hops made questions such as "where was the director born" require three
    edges instead of two.  A group is satisfied when any of its aliases is
    matched by the same path; distinct semantic relations remain distinct.
    """

    remaining = set(_query_relation_cues(question))
    groups: list[frozenset[str]] = []
    for event, companions in (
        ("birth", {"birth", "date", "location"}),
        ("death", {"death", "date", "location"}),
        ("release", {"release", "date"}),
    ):
        if event in remaining:
            group = frozenset(remaining & companions)
            groups.append(group)
            remaining -= group
    groups.extend(frozenset({cue}) for cue in sorted(remaining))
    return tuple(groups)


def _matched_relation_cues(
    traversal: "_Traversal",
    query_cues: Sequence[str],
) -> frozenset[str]:
    edge_text = (
        f"{traversal.edge.relation} {traversal.edge.target} "
        f"{traversal.edge.source}"
    )
    return frozenset(
        cue for cue in query_cues
        if _contains_alias(edge_text, _EDGE_CUE_ALIASES[cue])
    )


def _target_compatibility(target: str, cues: Iterable[str]) -> float:
    """Coarse answer-type guard used only for relation-slot ranking."""

    cues = set(cues)
    if not cues:
        return 0.0
    target_tokens = _tokens(target, drop_stopwords=False)
    has_digit = any(any(character.isdigit() for character in token) for token in target_tokens)
    temporal = has_digit or any(
        token in {
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        }
        for token in target_tokens
    )
    if "nationality" in cues:
        return -1.0 if temporal else (1.0 if 1 <= len(target_tokens) <= 5 else 0.0)
    if cues & {
        "mother", "father", "spouse", "director", "performer", "author",
        "composer", "name",
    }:
        return -0.5 if temporal else (1.0 if 1 <= len(target_tokens) <= 7 else 0.0)
    if cues & {"date", "death", "release"}:
        return 1.0 if temporal else 0.0
    if "count" in cues:
        return 1.0 if has_digit else 0.0
    return 0.0


def _stable_edge_id(
    context_id: str,
    source: str,
    relation: str,
    target: str,
    chunk_id: str,
) -> str:
    value = "\x1f".join((context_id, source, relation, target, chunk_id))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class FolderConfig:
    """Shared, pre-registered path construction parameters."""

    budget: int = 3
    loose_threshold: float = 0.25
    strict_threshold: float = 0.45
    beam_width: int = 12
    max_path_steps: int = 3
    min_edge_score: float = 0.18
    path_confidence_threshold: float = 0.24
    anchor_threshold: float = 0.34
    reverse_penalty: float = 0.04
    source_token_budget: int = 1400
    trace_token_budget: int = 96
    total_evidence_token_budget: int = 1536

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be at least 1")
        if self.strict_threshold <= self.loose_threshold:
            raise ValueError("strict_threshold must be greater than loose_threshold")
        if self.beam_width < 1 or self.max_path_steps < 1:
            raise ValueError("beam_width and max_path_steps must be positive")
        if min(
            self.source_token_budget,
            self.trace_token_budget,
            self.total_evidence_token_budget,
        ) < 0:
            raise ValueError("token budgets must be non-negative")
        if self.source_token_budget > self.total_evidence_token_budget:
            raise ValueError("source token budget exceeds total evidence budget")
        if self.trace_token_budget > self.total_evidence_token_budget:
            raise ValueError("trace token budget exceeds total evidence budget")


@dataclass(frozen=True)
class DirectedEdge:
    context_id: str
    source: str
    relation: str
    target: str
    chunk_id: str
    edge_id: str


@dataclass(frozen=True)
class AnchorCandidate:
    entity: str
    score: float
    source: str
    specificity: float
    degree: int


@dataclass(frozen=True)
class _Traversal:
    edge: DirectedEdge
    source: str
    target: str
    direction: str


@dataclass(frozen=True)
class _ScoredTraversal:
    traversal: _Traversal
    score: float
    query_score: float
    relation_coverage: float
    matched_cues: frozenset[str] = frozenset()
    target_compatibility: float = 0.0


@dataclass(frozen=True)
class _PathState:
    current: str
    traversals: tuple[_ScoredTraversal, ...]
    visited_entities: frozenset[str]
    used_edge_ids: frozenset[str]

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.traversal.edge.chunk_id for item in self.traversals))


class TopologyFolderV2:
    """Build deterministic query-conditioned path manifests."""

    def __init__(
        self,
        chunks: pd.DataFrame,
        embeddings: pd.DataFrame,
        graph: pd.DataFrame,
        *,
        dataset: str = "",
        config: FolderConfig | None = None,
    ) -> None:
        self.dataset = str(dataset)
        self.config = config or FolderConfig()
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

        chunk_copy = chunks.copy()
        embedding_copy = embeddings.copy()
        for frame in (chunk_copy, embedding_copy):
            frame["context_id"] = frame["context_id"].astype(str)
            frame["chunk_id"] = frame["chunk_id"].astype(str)
        merged = chunk_copy.merge(
            embedding_copy[[
                "context_id", "chunk_id", "embedding", "title_embedding",
            ]],
            on=["context_id", "chunk_id"],
            how="left",
            validate="one_to_one",
        )
        if merged["embedding"].isna().any() or merged["title_embedding"].isna().any():
            raise ValueError("every local chunk must have precomputed text and title embeddings")
        merged["embedding_unit"] = merged["embedding"].map(_unit)
        merged["title_embedding_unit"] = merged["title_embedding"].map(_unit)
        self.context_chunks: dict[str, pd.DataFrame] = {
            str(context_id): group.reset_index(drop=True)
            for context_id, group in merged.groupby("context_id", sort=False)
        }

        graph_copy = graph.copy()
        graph_copy["context_id"] = graph_copy["context_id"].astype(str)
        graph_copy["chunk_id"] = graph_copy["chunk_id"].astype(str)
        self.context_edges: dict[str, tuple[DirectedEdge, ...]] = {}
        self.context_entity_chunks: dict[str, dict[str, frozenset[str]]] = {}
        for context_id, group in graph_copy.groupby("context_id", sort=False):
            seen: set[tuple[str, str, str, str]] = set()
            edges: list[DirectedEdge] = []
            entity_chunks: dict[str, set[str]] = defaultdict(set)
            for row in group.itertuples(index=False):
                source = str(row.node_1).strip()
                target = str(row.node_2).strip()
                relation = str(row.edge).strip()
                chunk_id = str(row.chunk_id)
                if not source or not target or not relation:
                    continue
                key = (source, relation, target, chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(DirectedEdge(
                    context_id=str(context_id),
                    source=source,
                    relation=relation,
                    target=target,
                    chunk_id=chunk_id,
                    edge_id=_stable_edge_id(
                        str(context_id), source, relation, target, chunk_id,
                    ),
                ))
                entity_chunks[source].add(chunk_id)
                entity_chunks[target].add(chunk_id)
            self.context_edges[str(context_id)] = tuple(edges)
            self.context_entity_chunks[str(context_id)] = {
                entity: frozenset(ids) for entity, ids in entity_chunks.items()
            }

    def _score_context(
        self,
        context_id: str,
        query_vector: Sequence[float],
    ) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
        context_id = str(context_id)
        frame = self.context_chunks.get(context_id)
        if frame is None:
            raise ValueError(f"context_id {context_id!r} has no local chunks")
        query_unit = _unit(query_vector)
        if not len(query_unit):
            raise ValueError("query embedding is empty")
        dimension = len(query_unit)
        if any(len(vector) != dimension for vector in frame["embedding_unit"]):
            raise ValueError("query and chunk embeddings have different dimensions")
        semantic = {
            str(row.chunk_id): float(np.dot(query_unit, row.embedding_unit))
            for row in frame.itertuples(index=False)
        }
        title = {
            str(row.chunk_id): float(np.dot(query_unit, row.title_embedding_unit))
            for row in frame.itertuples(index=False)
        }
        return frame, semantic, title

    def _anchor_candidates(
        self,
        context_id: str,
        question: str,
        semantic_scores: Mapping[str, float],
    ) -> list[AnchorCandidate]:
        query_phrase = _normalized_phrase(question)
        query_tokens = set(_tokens(question))
        entity_chunks = self.context_entity_chunks.get(str(context_id), {})
        edges = self.context_edges.get(str(context_id), ())
        degree: dict[str, int] = defaultdict(int)
        for edge in edges:
            degree[edge.source] += 1
            degree[edge.target] += 1

        anchors: list[AnchorCandidate] = []
        for entity, chunks in entity_chunks.items():
            entity_phrase = _normalized_phrase(entity)
            entity_tokens = set(_tokens(entity))
            if not entity_phrase or not entity_tokens:
                continue
            exact = (
                f" {entity_phrase} " in f" {query_phrase} "
                or entity_phrase == query_phrase
            )
            overlap = _jaccard(entity_tokens, query_tokens)
            if not exact and overlap <= 0:
                continue
            specificity = min(
                1.0,
                0.25 * len(entity_tokens) + min(len(entity_phrase), 48) / 96.0,
            )
            generic = entity_phrase in _GENERIC_ENTITIES
            hub_penalty = min(0.18, math.log1p(degree.get(entity, 0)) / 40.0)
            chunk_signal = max((semantic_scores.get(cid, -1.0) for cid in chunks), default=-1.0)
            score = (
                (0.72 if exact else 0.0)
                + 0.38 * overlap
                + 0.16 * max(0.0, chunk_signal)
                + 0.08 * specificity
                - hub_penalty
                - (0.35 if generic else 0.0)
            )
            anchors.append(AnchorCandidate(
                entity=entity,
                score=float(score),
                source="query_exact" if exact else "query_lexical",
                specificity=float(specificity),
                degree=int(degree.get(entity, 0)),
            ))
        return sorted(
            anchors,
            key=lambda item: (
                0 if item.source == "query_exact" else 1,
                -item.score,
                -item.specificity,
                item.entity.casefold(),
            ),
        )

    def _adjacency(self, context_id: str) -> dict[str, list[_Traversal]]:
        adjacency: dict[str, list[_Traversal]] = defaultdict(list)
        for edge in self.context_edges.get(str(context_id), ()):
            adjacency[edge.source].append(_Traversal(
                edge=edge,
                source=edge.source,
                target=edge.target,
                direction="forward",
            ))
            # Extraction direction is not always the question traversal
            # direction.  Reverse traversal is explicit, audited, and mildly
            # penalized; it never invents a new canonical graph relation.
            adjacency[edge.target].append(_Traversal(
                edge=edge,
                source=edge.target,
                target=edge.source,
                direction="reverse",
            ))
        for values in adjacency.values():
            values.sort(key=lambda item: (
                item.direction == "reverse",
                item.edge.chunk_id,
                item.edge.edge_id,
            ))
        return dict(adjacency)

    def _score_traversal(
        self,
        traversal: _Traversal,
        question_tokens: set[str],
        semantic_scores: Mapping[str, float],
        anchor: AnchorCandidate,
        query_cues: Sequence[str],
        used_cues: frozenset[str],
        *,
        first_step: bool,
        target_degree: int,
    ) -> _ScoredTraversal:
        endpoint_tokens = set(_tokens(
            f"{traversal.edge.source} {traversal.edge.target}"
        ))
        relation_tokens = set(_tokens(traversal.edge.relation)) - endpoint_tokens
        coverage = _jaccard(relation_tokens, question_tokens)
        matched_cues = _matched_relation_cues(traversal, query_cues)
        new_cues = matched_cues - used_cues
        repeated_cues = matched_cues & used_cues
        slot_reward = 1.0 if new_cues else 0.0
        repeat_penalty = 1.0 if repeated_cues and not new_cues else 0.0
        compatibility = _target_compatibility(traversal.target, new_cues)
        query_score = float(semantic_scores.get(traversal.edge.chunk_id, -1.0))
        target_specificity = min(1.0, len(_tokens(traversal.target)) / 3.0)
        hub_penalty = min(0.10, math.log1p(target_degree) / 55.0)
        score = (
            0.38 * query_score
            + 0.16 * coverage
            + 0.42 * slot_reward
            + 0.18 * compatibility
            + (0.07 * anchor.score if first_step else 0.03)
            + 0.04 * target_specificity
            - 0.20 * repeat_penalty
            - hub_penalty
            - (self.config.reverse_penalty if traversal.direction == "reverse" else 0.0)
        )
        return _ScoredTraversal(
            traversal=traversal,
            score=float(score),
            query_score=query_score,
            relation_coverage=float(max(coverage, slot_reward)),
            matched_cues=matched_cues,
            target_compatibility=float(compatibility),
        )

    @staticmethod
    def _state_rank(state: _PathState) -> tuple[float, float, int, float, str]:
        scores = [item.score for item in state.traversals]
        coverages = [item.relation_coverage for item in state.traversals]
        mean_score = sum(scores) / len(scores)
        mean_coverage = sum(coverages) / len(coverages)
        unique_chunks = len(state.chunk_ids)
        reverse_count = sum(
            item.traversal.direction == "reverse" for item in state.traversals
        )
        signature = "|".join(item.traversal.edge.edge_id for item in state.traversals)
        utility = mean_score + 0.08 * min(unique_chunks, 3) + 0.06 * mean_coverage
        return utility, mean_score, unique_chunks, -reverse_count, signature

    def _search_path(
        self,
        context_id: str,
        question: str,
        semantic_scores: Mapping[str, float],
        anchor: AnchorCandidate,
        *,
        max_steps: int | None = None,
        target_steps: int = 2,
    ) -> tuple[_ScoredTraversal, ...]:
        """Run a bounded relation-level beam search from one query anchor."""

        adjacency = self._adjacency(context_id)
        if anchor.entity not in adjacency:
            return ()
        degrees = {entity: len(values) for entity, values in adjacency.items()}
        question_tokens = set(_tokens(question))
        query_cues = _query_relation_cues(question)
        # A semantically unspecified walk is not Topology Folding.  Returning
        # no path here activates the explicit Graph-Naive fallback instead of
        # presenting an arbitrary connected walk as query-conditioned proof.
        if not query_cues:
            return ()
        limit = max_steps or self.config.max_path_steps
        beam = [_PathState(
            current=anchor.entity,
            traversals=(),
            visited_entities=frozenset({anchor.entity}),
            used_edge_ids=frozenset(),
        )]
        completed: list[_PathState] = []
        for _ in range(limit):
            expanded: list[_PathState] = []
            for state in beam:
                for traversal in adjacency.get(state.current, ()):
                    edge = traversal.edge
                    if edge.edge_id in state.used_edge_ids:
                        continue
                    if traversal.target in state.visited_entities:
                        continue
                    if (
                        edge.chunk_id not in state.chunk_ids
                        and len(state.chunk_ids) >= self.config.budget
                    ):
                        continue
                    scored = self._score_traversal(
                        traversal,
                        question_tokens,
                        semantic_scores,
                        anchor,
                        query_cues,
                        frozenset(
                            cue
                            for previous in state.traversals
                            for cue in previous.matched_cues
                        ),
                        first_step=not state.traversals,
                        target_degree=degrees.get(traversal.target, 0),
                    )
                    # A relation phrase can mention the requested type while
                    # pointing at an incompatible endpoint, e.g. "American
                    # film ... released in 1994".  Such an edge must not
                    # satisfy a nationality slot merely because the adjective
                    # appears somewhere in the sentence.
                    new_cues = scored.matched_cues - frozenset(
                        cue
                        for previous in state.traversals
                        for cue in previous.matched_cues
                    )
                    if new_cues and scored.target_compatibility < 0:
                        continue
                    if scored.score < self.config.min_edge_score:
                        continue
                    expanded.append(_PathState(
                        current=traversal.target,
                        traversals=state.traversals + (scored,),
                        visited_entities=state.visited_entities | {traversal.target},
                        used_edge_ids=state.used_edge_ids | {edge.edge_id},
                    ))
            if not expanded:
                break
            expanded.sort(key=self._state_rank, reverse=True)
            beam = expanded[: self.config.beam_width]
            completed.extend(beam)
        if not completed:
            return ()
        required_slots = _relation_slot_groups(question)
        eligible = [
            state
            for state in completed
            if len(state.traversals) == target_steps
            and all(
                slot & frozenset(
                    cue
                    for traversal in state.traversals
                    for cue in traversal.matched_cues
                )
                for slot in required_slots
            )
        ]
        # A shorter path does not satisfy the pre-declared reasoning depth.
        # Returning it as foldable would silently turn a stress-test query
        # into a shallower problem, so fail closed to Graph-Naive instead.
        if not eligible:
            return ()
        best = max(eligible, key=self._state_rank)
        return best.traversals

    @staticmethod
    def _infer_target_steps(question: str, intent_type: str, budget: int) -> int:
        lowered = question.casefold()
        nested = lowered.count(" of ") + lowered.count(" whose ")
        relation_depth = len(_relation_slot_groups(question))
        if intent_type == "Comparative":
            return min(budget, max(1, relation_depth))
        if intent_type == "Reasoning":
            return min(budget, max(2, nested, relation_depth))
        return min(budget, max(1, relation_depth))

    @staticmethod
    def _path_steps(
        traversals: Sequence[_ScoredTraversal],
        *,
        branch: str,
    ) -> tuple[PathStep, ...]:
        return tuple(
            PathStep(
                step=index,
                branch=branch,
                source_entity=item.traversal.source,
                relation=item.traversal.edge.relation,
                target_entity=item.traversal.target,
                supporting_chunk_id=item.traversal.edge.chunk_id,
                edge_score=item.score,
                query_score=item.query_score,
                canonical_source_entity=item.traversal.edge.source,
                canonical_target_entity=item.traversal.edge.target,
                traversal_direction=item.traversal.direction,
            )
            for index, item in enumerate(traversals, start=1)
        )

    @staticmethod
    def _deduplicate_path_steps(
        branches: Sequence[tuple[str, Sequence[_ScoredTraversal]]],
        budget: int,
    ) -> tuple[PathStep, ...]:
        """Keep branch prefixes while respecting the unique source budget."""

        accepted: list[PathStep] = []
        selected_chunks: set[str] = set()
        # Round-robin gives comparative branches equal access to the budget.
        depth = 0
        while True:
            progressed = False
            for branch, traversals in branches:
                if depth >= len(traversals):
                    continue
                item = traversals[depth]
                chunk_id = item.traversal.edge.chunk_id
                if chunk_id not in selected_chunks and len(selected_chunks) >= budget:
                    continue
                selected_chunks.add(chunk_id)
                branch_step = 1 + sum(step.branch == branch for step in accepted)
                accepted.append(PathStep(
                    step=branch_step,
                    branch=branch,
                    source_entity=item.traversal.source,
                    relation=item.traversal.edge.relation,
                    target_entity=item.traversal.target,
                    supporting_chunk_id=chunk_id,
                    edge_score=item.score,
                    query_score=item.query_score,
                    canonical_source_entity=item.traversal.edge.source,
                    canonical_target_entity=item.traversal.edge.target,
                    traversal_direction=item.traversal.direction,
                ))
                progressed = True
            depth += 1
            if not progressed or all(depth >= len(items) for _, items in branches):
                break
        return tuple(accepted)

    def _fallback_manifest(
        self,
        *,
        query_id: str,
        context_id: str,
        question: str,
        intent_type: str,
        intent_strategy: str,
        candidate_ids: Sequence[str],
        semantic_scores: Mapping[str, float],
        anchors: Sequence[AnchorCandidate],
        reason: str,
        anchor_margin: float,
    ) -> FoldManifest:
        score_order = tuple(sorted(
            candidate_ids,
            key=lambda chunk_id: (-semantic_scores.get(chunk_id, -1.0), chunk_id),
        )[: self.config.budget])
        return FoldManifest(
            query_id=query_id,
            dataset=self.dataset,
            context_id=str(context_id),
            question=question,
            intent_type=intent_type,
            intent_strategy=intent_strategy,
            anchor_entities=tuple(anchor.entity for anchor in anchors),
            path_policy="graph_naive_fallback",
            path_steps=(),
            candidate_chunk_ids=tuple(candidate_ids),
            selected_chunk_ids=score_order,
            score_order_chunk_ids=score_order,
            path_order_chunk_ids=score_order,
            chunk_scores=tuple(
                (chunk_id, float(semantic_scores.get(chunk_id, -1.0)))
                for chunk_id in candidate_ids
            ),
            core_chunk_ids=(),
            peripheral_chunk_ids=(),
            budget=self.config.budget,
            token_budget=self.config.total_evidence_token_budget,
            source_token_budget=self.config.source_token_budget,
            trace_token_budget=self.config.trace_token_budget,
            total_evidence_token_budget=self.config.total_evidence_token_budget,
            foldable=False,
            fallback_to_graph_naive=True,
            fallback_reason=reason,
            path_confidence=0.0,
            path_continuous=False,
            anchor_margin=anchor_margin,
            anchor_source=(anchors[0].source if anchors else "none"),
            llm_reranker_used=False,
            branch_complete=False,
        ).validate()

    def build_manifest(
        self,
        query_id: str,
        context_id: str,
        question: str,
        query_vector: Sequence[float],
        intent_type: str,
        *,
        intent_strategy: str = "",
    ) -> FoldManifest:
        """Build one manifest without consulting any gold annotation."""

        frame, semantic_scores, _ = self._score_context(context_id, query_vector)
        candidate_ids = tuple(frame["chunk_id"].astype(str).tolist())
        anchors = self._anchor_candidates(context_id, question, semantic_scores)
        qualified = [
            anchor for anchor in anchors
            if anchor.score >= self.config.anchor_threshold
        ]
        anchor_margin = (
            qualified[0].score - qualified[1].score
            if len(qualified) > 1 else (qualified[0].score if qualified else 0.0)
        )
        if not qualified:
            return self._fallback_manifest(
                query_id=query_id,
                context_id=str(context_id),
                question=question,
                intent_type=intent_type,
                intent_strategy=intent_strategy,
                candidate_ids=candidate_ids,
                semantic_scores=semantic_scores,
                anchors=(),
                reason="no_reliable_query_anchor",
                anchor_margin=0.0,
            )

        target_steps = self._infer_target_steps(
            question, intent_type, self.config.budget,
        )
        branch_complete = True
        if intent_type == "Comparative":
            exact = [anchor for anchor in qualified if anchor.source == "query_exact"]
            branch_anchors = exact[:2]
            if len(branch_anchors) < 2:
                return self._fallback_manifest(
                    query_id=query_id,
                    context_id=str(context_id),
                    question=question,
                    intent_type=intent_type,
                    intent_strategy=intent_strategy,
                    candidate_ids=candidate_ids,
                    semantic_scores=semantic_scores,
                    anchors=qualified[:2],
                    reason="comparative_branch_incomplete",
                    anchor_margin=anchor_margin,
                )
            # Never truncate one comparative branch merely to fit B.  A
            # comparison is foldable only when both relation chains fit in the
            # registered slot budget (shared source chunks may deduplicate
            # later, but we cannot assume that before constructing the paths).
            if 2 * target_steps > self.config.budget:
                return self._fallback_manifest(
                    query_id=query_id,
                    context_id=str(context_id),
                    question=question,
                    intent_type=intent_type,
                    intent_strategy=intent_strategy,
                    candidate_ids=candidate_ids,
                    semantic_scores=semantic_scores,
                    anchors=branch_anchors,
                    reason="comparative_branch_incomplete",
                    anchor_margin=anchor_margin,
                )
            branch_paths = [
                (
                    branch,
                    self._search_path(
                        str(context_id), question, semantic_scores, anchor,
                        max_steps=target_steps,
                        target_steps=target_steps,
                    ),
                )
                for branch, anchor in zip(("A", "B"), branch_anchors)
            ]
            branch_complete = all(path for _, path in branch_paths)
            if not branch_complete:
                return self._fallback_manifest(
                    query_id=query_id,
                    context_id=str(context_id),
                    question=question,
                    intent_type=intent_type,
                    intent_strategy=intent_strategy,
                    candidate_ids=candidate_ids,
                    semantic_scores=semantic_scores,
                    anchors=branch_anchors,
                    reason="comparative_branch_incomplete",
                    anchor_margin=anchor_margin,
                )
            steps = self._deduplicate_path_steps(branch_paths, self.config.budget)
            accepted_branches = {step.branch for step in steps}
            branch_complete = (
                accepted_branches == {"A", "B"}
                and all(
                    sum(step.branch == branch for step in steps) == target_steps
                    for branch in ("A", "B")
                )
            )
            if not branch_complete:
                return self._fallback_manifest(
                    query_id=query_id,
                    context_id=str(context_id),
                    question=question,
                    intent_type=intent_type,
                    intent_strategy=intent_strategy,
                    candidate_ids=candidate_ids,
                    semantic_scores=semantic_scores,
                    anchors=branch_anchors,
                    reason="comparative_branch_incomplete",
                    anchor_margin=anchor_margin,
                )
            used_anchors = branch_anchors
            path_policy = "directed_comparative_branches"
        else:
            anchor = qualified[0]
            traversal_path = self._search_path(
                str(context_id), question, semantic_scores, anchor,
                target_steps=target_steps,
            )
            if not traversal_path:
                return self._fallback_manifest(
                    query_id=query_id,
                    context_id=str(context_id),
                    question=question,
                    intent_type=intent_type,
                    intent_strategy=intent_strategy,
                    candidate_ids=candidate_ids,
                    semantic_scores=semantic_scores,
                    anchors=(anchor,),
                    reason="no_continuous_relation_path",
                    anchor_margin=anchor_margin,
                )
            steps = self._path_steps(traversal_path, branch="main")
            used_anchors = (anchor,)
            path_policy = "directed_relation_path"

        path_confidence = float(
            sum(step.edge_score for step in steps) / len(steps)
        ) if steps else 0.0
        if path_confidence < self.config.path_confidence_threshold:
            return self._fallback_manifest(
                query_id=query_id,
                context_id=str(context_id),
                question=question,
                intent_type=intent_type,
                intent_strategy=intent_strategy,
                candidate_ids=candidate_ids,
                semantic_scores=semantic_scores,
                anchors=used_anchors,
                reason="path_confidence_below_threshold",
                anchor_margin=anchor_margin,
            )

        core_ids = tuple(dict.fromkeys(step.supporting_chunk_id for step in steps))
        if any(
            semantic_scores.get(chunk_id, -1.0) < self.config.loose_threshold
            for chunk_id in core_ids
        ):
            return self._fallback_manifest(
                query_id=query_id,
                context_id=str(context_id),
                question=question,
                intent_type=intent_type,
                intent_strategy=intent_strategy,
                candidate_ids=candidate_ids,
                semantic_scores=semantic_scores,
                anchors=used_anchors,
                reason="core_below_loose_threshold",
                anchor_margin=anchor_margin,
            )

        skeleton_entities = {
            entity
            for step in steps
            for entity in (step.source_entity, step.target_entity)
        }
        entity_chunks = self.context_entity_chunks.get(str(context_id), {})
        adjacent_ids = {
            chunk_id
            for entity in skeleton_entities
            for chunk_id in entity_chunks.get(entity, ())
        }
        peripheral_pool = [
            chunk_id for chunk_id in candidate_ids
            if chunk_id not in core_ids
            and chunk_id in adjacent_ids
            and semantic_scores.get(chunk_id, -1.0) >= self.config.strict_threshold
        ]
        peripheral_ids = tuple(sorted(
            peripheral_pool,
            key=lambda chunk_id: (-semantic_scores[chunk_id], chunk_id),
        )[: max(0, self.config.budget - len(core_ids))])
        path_order = tuple(core_ids + peripheral_ids)
        score_order = tuple(sorted(
            path_order,
            key=lambda chunk_id: (-semantic_scores[chunk_id], chunk_id),
        ))
        manifest = FoldManifest(
            query_id=query_id,
            dataset=self.dataset,
            context_id=str(context_id),
            question=question,
            intent_type=intent_type,
            intent_strategy=intent_strategy,
            anchor_entities=tuple(anchor.entity for anchor in used_anchors),
            path_policy=path_policy,
            path_steps=steps,
            candidate_chunk_ids=candidate_ids,
            selected_chunk_ids=path_order,
            score_order_chunk_ids=score_order,
            path_order_chunk_ids=path_order,
            chunk_scores=tuple(
                (chunk_id, float(semantic_scores[chunk_id]))
                for chunk_id in candidate_ids
            ),
            core_chunk_ids=core_ids,
            peripheral_chunk_ids=peripheral_ids,
            budget=self.config.budget,
            token_budget=self.config.total_evidence_token_budget,
            source_token_budget=self.config.source_token_budget,
            trace_token_budget=self.config.trace_token_budget,
            total_evidence_token_budget=self.config.total_evidence_token_budget,
            foldable=True,
            fallback_to_graph_naive=False,
            fallback_reason="",
            path_confidence=path_confidence,
            path_continuous=True,
            anchor_margin=anchor_margin,
            anchor_source=used_anchors[0].source,
            llm_reranker_used=False,
            branch_complete=branch_complete,
        )
        return manifest.validate()


__all__ = [
    "AnchorCandidate",
    "DirectedEdge",
    "FolderConfig",
    "TopologyFolderV2",
]
