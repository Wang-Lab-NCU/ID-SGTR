"""Stable path manifests for controlled Topology Folding experiments.

The manifest is the boundary between *selection* and *representation*.  A
path builder writes it once; every controlled renderer subsequently consumes
the same candidate and selected chunk IDs.  Consequently, no renderer is able
to silently change retrieval results.

The module deliberately has no dependency on pandas, torch, an embedding
model, or gold evidence.  It can therefore be used by command-line validation
and unit tests without loading the experiment stack.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_VERSION = "2.0"
IMPLEMENTATION_VERSION = "topology-folding-v2.4-unicode-slot-groups"
SUPPORTED_IMPLEMENTATION_VERSIONS = frozenset({
    IMPLEMENTATION_VERSION,
    "topology-folding-runtime-trace-v1",
    "topology-folding-runtime-trace-v1.1-graph-aligned",
    "topology-folding-runtime-trace-v1.2-branch-forest",
    "topology-folding-runtime-trace-v1.3-chosen-edges",
    "topology-folding-runtime-trace-v1.4-executed-frontier",
    "topology-folding-runtime-trace-v1.5-factorial-path-order",
    "topology-folding-runtime-trace-v1.6-question-aligned-selection",
    "topology-folding-runtime-trace-v1.7-conservative-question-aligned",
    "topology-folding-runtime-trace-v1.8-factorial-clean-question-aligned",
})
_DIRECTIONS = frozenset({"forward", "reverse"})


class ManifestValidationError(ValueError):
    """Raised when a manifest cannot be safely replayed."""


def _clean_id(value: Any) -> str:
    return str(value).strip()


def _clean_ids(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(_clean_id(value) for value in values)


def _clean_chunk_scores(values: Any) -> tuple[tuple[str, float], ...]:
    """Normalize score mappings/pairs to a stable, ID-sorted tuple."""

    if isinstance(values, Mapping):
        pairs = values.items()
    else:
        pairs = values or ()
    normalized: list[tuple[str, float]] = []
    for item in pairs:
        if isinstance(item, Mapping):
            chunk_id = item.get("chunk_id", "")
            score = item.get("score", 0.0)
        else:
            try:
                chunk_id, score = item
            except (TypeError, ValueError) as exc:
                raise TypeError("chunk_scores entries must be (chunk_id, score) pairs") from exc
        normalized.append((_clean_id(chunk_id), float(score)))
    return tuple(sorted(normalized, key=lambda pair: pair[0]))


def _json_ready(value: Any) -> Any:
    """Return a recursively JSON-compatible value with deterministic types."""

    if isinstance(value, FoldManifest):
        return value.to_dict()
    if isinstance(value, PathStep):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__!r} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to the canonical representation used for hashing."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest of a manifest or JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PathStep:
    """One aligned relation in a query-directed path.

    ``source_entity`` and ``target_entity`` describe the traversal direction
    used by the reasoning path.  ``canonical_*`` preserve the direction in the
    stored knowledge graph.  They differ exactly when
    ``traversal_direction == \"reverse\"``.

    ``step`` is one-based *within a branch*.  The ``branch`` field allows a
    comparative query to store independent left/right paths without pretending
    that the end of one branch connects to the start of the other.
    """

    step: int
    branch: str
    source_entity: str
    relation: str
    target_entity: str
    supporting_chunk_id: str
    edge_score: float = 0.0
    query_score: float = 0.0
    canonical_source_entity: str = ""
    canonical_target_entity: str = ""
    traversal_direction: str = "forward"

    def __post_init__(self) -> None:
        direction = str(self.traversal_direction).strip().lower()
        source = _clean_id(self.source_entity)
        target = _clean_id(self.target_entity)
        canonical_source = _clean_id(self.canonical_source_entity)
        canonical_target = _clean_id(self.canonical_target_entity)

        # Older builders only supplied traversal entities.  Infer canonical
        # endpoints so their output remains valid and fully explicit on disk.
        if not canonical_source and not canonical_target:
            if direction == "reverse":
                canonical_source, canonical_target = target, source
            else:
                canonical_source, canonical_target = source, target

        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "branch", _clean_id(self.branch))
        object.__setattr__(self, "source_entity", source)
        object.__setattr__(self, "relation", str(self.relation).strip())
        object.__setattr__(self, "target_entity", target)
        object.__setattr__(self, "supporting_chunk_id", _clean_id(self.supporting_chunk_id))
        object.__setattr__(self, "edge_score", float(self.edge_score))
        object.__setattr__(self, "query_score", float(self.query_score))
        object.__setattr__(self, "canonical_source_entity", canonical_source)
        object.__setattr__(self, "canonical_target_entity", canonical_target)
        object.__setattr__(self, "traversal_direction", direction)

    @property
    def traversal_source_entity(self) -> str:
        """Explicit alias used by renderers and audit output."""

        return self.source_entity

    @property
    def traversal_target_entity(self) -> str:
        """Explicit alias used by renderers and audit output."""

        return self.target_entity

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.step < 1:
            errors.append("step must be one-based")
        for name in (
            "branch",
            "source_entity",
            "relation",
            "target_entity",
            "supporting_chunk_id",
            "canonical_source_entity",
            "canonical_target_entity",
        ):
            if not getattr(self, name):
                errors.append(f"{name} must not be empty")
        if self.traversal_direction not in _DIRECTIONS:
            errors.append("traversal_direction must be 'forward' or 'reverse'")
        if not math.isfinite(self.edge_score):
            errors.append("edge_score must be finite")
        if not math.isfinite(self.query_score):
            errors.append("query_score must be finite")

        if self.traversal_direction == "forward":
            expected = (self.canonical_source_entity, self.canonical_target_entity)
        elif self.traversal_direction == "reverse":
            expected = (self.canonical_target_entity, self.canonical_source_entity)
        else:
            expected = None
        if expected is not None and expected != (self.source_entity, self.target_entity):
            errors.append(
                "canonical endpoints do not match the declared traversal direction"
            )
        return errors

    def validate(self) -> "PathStep":
        errors = self.validation_errors()
        if errors:
            raise ManifestValidationError("invalid path step: " + "; ".join(errors))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "branch": self.branch,
            "source_entity": self.source_entity,
            "relation": self.relation,
            "target_entity": self.target_entity,
            "supporting_chunk_id": self.supporting_chunk_id,
            "edge_score": self.edge_score,
            "query_score": self.query_score,
            "canonical_source_entity": self.canonical_source_entity,
            "canonical_target_entity": self.canonical_target_entity,
            "traversal_direction": self.traversal_direction,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PathStep":
        data = dict(value)
        # Accept explicit traversal aliases in externally produced manifests.
        source = data.get("source_entity", data.get("traversal_source_entity", ""))
        target = data.get("target_entity", data.get("traversal_target_entity", ""))
        return cls(
            step=data.get("step", 0),
            branch=data.get("branch", "main"),
            source_entity=source,
            relation=data.get("relation", ""),
            target_entity=target,
            supporting_chunk_id=data.get("supporting_chunk_id", ""),
            edge_score=data.get("edge_score", 0.0),
            query_score=data.get("query_score", 0.0),
            canonical_source_entity=data.get("canonical_source_entity", ""),
            canonical_target_entity=data.get("canonical_target_entity", ""),
            traversal_direction=data.get("traversal_direction", "forward"),
        )


@dataclass(frozen=True)
class FoldManifest:
    """Immutable selection/path record shared by all controlled renderers."""

    query_id: str
    dataset: str = ""
    context_id: str = ""
    question: str = ""
    intent_type: str = ""
    intent_strategy: str = ""
    anchor_entities: tuple[str, ...] = field(default_factory=tuple)
    path_policy: str = ""
    path_steps: tuple[PathStep, ...] = field(default_factory=tuple)
    candidate_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    selected_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    core_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    peripheral_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    score_order_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    path_order_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    chunk_scores: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    budget: int = 3
    token_budget: int = 0
    source_token_budget: int = 1400
    trace_token_budget: int = 96
    total_evidence_token_budget: int = 1536
    foldable: bool = False
    fallback_reason: str = ""
    fallback_to_graph_naive: bool = False
    path_confidence: float = 0.0
    path_continuous: bool = False
    branch_complete: bool = True
    anchor_margin: float = 0.0
    anchor_source: str = ""
    llm_reranker_used: bool = False
    manifest_version: str = MANIFEST_VERSION
    implementation_version: str = IMPLEMENTATION_VERSION

    def __post_init__(self) -> None:
        steps = tuple(
            step if isinstance(step, PathStep) else PathStep.from_dict(step)
            for step in self.path_steps
        )
        object.__setattr__(self, "query_id", _clean_id(self.query_id))
        object.__setattr__(self, "dataset", _clean_id(self.dataset))
        object.__setattr__(self, "context_id", _clean_id(self.context_id))
        object.__setattr__(self, "question", str(self.question).strip())
        object.__setattr__(self, "intent_type", _clean_id(self.intent_type))
        object.__setattr__(self, "intent_strategy", str(self.intent_strategy).strip())
        object.__setattr__(self, "anchor_entities", _clean_ids(self.anchor_entities))
        object.__setattr__(self, "path_policy", _clean_id(self.path_policy))
        object.__setattr__(self, "path_steps", steps)
        object.__setattr__(self, "candidate_chunk_ids", _clean_ids(self.candidate_chunk_ids))
        object.__setattr__(self, "selected_chunk_ids", _clean_ids(self.selected_chunk_ids))
        object.__setattr__(self, "core_chunk_ids", _clean_ids(self.core_chunk_ids))
        object.__setattr__(self, "peripheral_chunk_ids", _clean_ids(self.peripheral_chunk_ids))
        score_order = _clean_ids(self.score_order_chunk_ids)
        path_order = _clean_ids(self.path_order_chunk_ids)
        # Backward compatibility: manifests created before the factorial
        # protocol used selected_chunk_ids as both observable orders.
        if not score_order and self.selected_chunk_ids:
            score_order = tuple(self.selected_chunk_ids)
        if not path_order and self.selected_chunk_ids:
            path_order = tuple(self.selected_chunk_ids)
        chunk_scores = _clean_chunk_scores(self.chunk_scores)
        if not chunk_scores and self.selected_chunk_ids:
            chunk_scores = tuple(sorted(
                ((chunk_id, 0.0) for chunk_id in self.selected_chunk_ids),
                key=lambda pair: pair[0],
            ))
        object.__setattr__(self, "score_order_chunk_ids", score_order)
        object.__setattr__(self, "path_order_chunk_ids", path_order)
        object.__setattr__(self, "chunk_scores", chunk_scores)
        object.__setattr__(self, "budget", int(self.budget))
        object.__setattr__(self, "token_budget", int(self.token_budget))
        object.__setattr__(self, "source_token_budget", int(self.source_token_budget))
        object.__setattr__(self, "trace_token_budget", int(self.trace_token_budget))
        object.__setattr__(
            self, "total_evidence_token_budget", int(self.total_evidence_token_budget)
        )
        object.__setattr__(self, "foldable", bool(self.foldable))
        object.__setattr__(self, "fallback_reason", str(self.fallback_reason).strip())
        fallback = bool(self.fallback_to_graph_naive)
        if not self.foldable and self.fallback_reason:
            fallback = True
        object.__setattr__(self, "fallback_to_graph_naive", fallback)
        object.__setattr__(self, "path_confidence", float(self.path_confidence))
        object.__setattr__(self, "path_continuous", bool(self.path_continuous))
        object.__setattr__(self, "branch_complete", bool(self.branch_complete))
        object.__setattr__(self, "anchor_margin", float(self.anchor_margin))
        object.__setattr__(self, "anchor_source", _clean_id(self.anchor_source))
        object.__setattr__(self, "llm_reranker_used", bool(self.llm_reranker_used))
        object.__setattr__(self, "manifest_version", _clean_id(self.manifest_version))
        object.__setattr__(self, "implementation_version", _clean_id(self.implementation_version))

    @property
    def selected_count(self) -> int:
        return len(self.selected_chunk_ids)

    @property
    def unused_budget(self) -> int:
        return self.budget - self.selected_count

    @property
    def path_length(self) -> int:
        return len(self.path_steps)

    @property
    def trace_count(self) -> int:
        return len(self.path_steps)

    @property
    def sha256(self) -> str:
        return stable_sha256(self)

    @property
    def manifest_sha256(self) -> str:
        """Telemetry-friendly alias for :attr:`sha256`."""

        return self.sha256

    def canonical_json(self) -> str:
        return canonical_json(self)

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.query_id:
            errors.append("query_id must not be empty")
        if not self.manifest_version:
            errors.append("manifest_version must not be empty")
        elif self.manifest_version != MANIFEST_VERSION:
            errors.append(
                f"manifest_version must be {MANIFEST_VERSION!r}, "
                f"got {self.manifest_version!r}"
            )
        if not self.implementation_version:
            errors.append("implementation_version must not be empty")
        elif self.implementation_version not in SUPPORTED_IMPLEMENTATION_VERSIONS:
            errors.append(
                "implementation_version must be one of "
                f"{sorted(SUPPORTED_IMPLEMENTATION_VERSIONS)!r}, "
                f"got {self.implementation_version!r}"
            )
        if self.budget < 1:
            errors.append("budget must be at least 1")
        if self.token_budget < 0:
            errors.append("token_budget must be non-negative")
        if self.source_token_budget < 0:
            errors.append("source_token_budget must be non-negative")
        if self.trace_token_budget < 0:
            errors.append("trace_token_budget must be non-negative")
        if self.total_evidence_token_budget < 1:
            errors.append("total_evidence_token_budget must be positive")
        if self.source_token_budget + self.trace_token_budget > self.total_evidence_token_budget:
            errors.append(
                "source_token_budget + trace_token_budget exceeds "
                "total_evidence_token_budget"
            )
        if not math.isfinite(self.path_confidence):
            errors.append("path_confidence must be finite")
        if not math.isfinite(self.anchor_margin):
            errors.append("anchor_margin must be finite")
        if self.anchor_margin < 0:
            errors.append("anchor_margin must be non-negative")

        named_ids: tuple[tuple[str, Sequence[str]], ...] = (
            ("anchor_entities", self.anchor_entities),
            ("candidate_chunk_ids", self.candidate_chunk_ids),
            ("selected_chunk_ids", self.selected_chunk_ids),
            ("core_chunk_ids", self.core_chunk_ids),
            ("peripheral_chunk_ids", self.peripheral_chunk_ids),
            ("score_order_chunk_ids", self.score_order_chunk_ids),
            ("path_order_chunk_ids", self.path_order_chunk_ids),
        )
        for name, values in named_ids:
            if any(not value for value in values):
                errors.append(f"{name} contains an empty ID")
            if len(set(values)) != len(values):
                errors.append(f"{name} contains duplicate IDs")

        candidates = set(self.candidate_chunk_ids)
        selected = set(self.selected_chunk_ids)
        core = set(self.core_chunk_ids)
        peripheral = set(self.peripheral_chunk_ids)
        if len(self.selected_chunk_ids) > self.budget:
            errors.append("selected_chunk_ids exceeds budget")
        if not selected.issubset(candidates):
            errors.append("selected_chunk_ids must be a subset of candidate_chunk_ids")
        if not core.issubset(selected):
            errors.append("core_chunk_ids must be a subset of selected_chunk_ids")
        if not peripheral.issubset(selected):
            errors.append("peripheral_chunk_ids must be a subset of selected_chunk_ids")
        if core & peripheral:
            errors.append("core_chunk_ids and peripheral_chunk_ids must be disjoint")
        if (
            len(self.score_order_chunk_ids) != len(self.selected_chunk_ids)
            or set(self.score_order_chunk_ids) != selected
        ):
            errors.append("score_order_chunk_ids must be a permutation of selected_chunk_ids")
        if (
            len(self.path_order_chunk_ids) != len(self.selected_chunk_ids)
            or set(self.path_order_chunk_ids) != selected
        ):
            errors.append("path_order_chunk_ids must be a permutation of selected_chunk_ids")

        score_ids = [chunk_id for chunk_id, _ in self.chunk_scores]
        if any(not chunk_id for chunk_id in score_ids):
            errors.append("chunk_scores contains an empty chunk ID")
        if len(set(score_ids)) != len(score_ids):
            errors.append("chunk_scores contains duplicate chunk IDs")
        if not set(score_ids).issubset(candidates):
            errors.append("chunk_scores IDs must be a subset of candidate_chunk_ids")
        if not selected.issubset(set(score_ids)):
            errors.append("chunk_scores must cover every selected chunk")
        if any(not math.isfinite(score) for _, score in self.chunk_scores):
            errors.append("chunk_scores values must be finite")
        score_by_id = dict(self.chunk_scores)
        if selected.issubset(score_by_id):
            expected_score_order = tuple(sorted(
                selected,
                key=lambda chunk_id: (-score_by_id[chunk_id], chunk_id),
            ))
            if tuple(self.score_order_chunk_ids) != expected_score_order:
                errors.append(
                    "score_order_chunk_ids must be descending by frozen "
                    "chunk score with chunk-ID tie breaking"
                )

        branches: dict[str, list[PathStep]] = {}
        seen_positions: set[tuple[str, int]] = set()
        supporting_ids: set[str] = set()
        for index, step in enumerate(self.path_steps):
            for error in step.validation_errors():
                errors.append(f"path_steps[{index}]: {error}")
            key = (step.branch, step.step)
            if key in seen_positions:
                errors.append(f"duplicate path step {step.branch}:{step.step}")
            seen_positions.add(key)
            branches.setdefault(step.branch, []).append(step)
            supporting_ids.add(step.supporting_chunk_id)

        for branch, steps in branches.items():
            ordered = sorted(steps, key=lambda item: item.step)
            expected_steps = list(range(1, len(ordered) + 1))
            actual_steps = [item.step for item in ordered]
            if actual_steps != expected_steps:
                errors.append(
                    f"branch {branch!r} steps must be contiguous and one-based: "
                    f"got {actual_steps}"
                )
            for left, right in zip(ordered, ordered[1:]):
                if left.target_entity != right.source_entity:
                    errors.append(
                        f"branch {branch!r} is discontinuous between steps "
                        f"{left.step} and {right.step}"
                    )

        if not supporting_ids.issubset(candidates):
            errors.append("every path step must align to a candidate chunk")
        if not supporting_ids.issubset(core):
            errors.append("every path step must align to a selected core chunk")
        expected_core_order = tuple(dict.fromkeys(
            step.supporting_chunk_id for step in self.path_steps
        ))
        if self.foldable and tuple(self.core_chunk_ids) != expected_core_order:
            errors.append(
                "core_chunk_ids must follow first occurrence in path_steps"
            )
        score_preserving_runtime = self.implementation_version in {
            "topology-folding-runtime-trace-v1.3-chosen-edges",
            "topology-folding-runtime-trace-v1.4-executed-frontier",
        }
        if score_preserving_runtime:
            if tuple(self.path_order_chunk_ids) != tuple(self.score_order_chunk_ids):
                errors.append(
                    "score-preserving runtime manifests must preserve semantic source order"
                )
        elif self.foldable and tuple(self.path_order_chunk_ids) != (
            tuple(self.core_chunk_ids) + tuple(self.peripheral_chunk_ids)
        ):
            errors.append(
                "path_order_chunk_ids must place path core before peripheral chunks"
            )
        if self.foldable and self.path_steps:
            expected_confidence = sum(
                step.edge_score for step in self.path_steps
            ) / len(self.path_steps)
            if not math.isclose(
                self.path_confidence, expected_confidence,
                rel_tol=1e-9, abs_tol=1e-9,
            ):
                errors.append(
                    "path_confidence must equal the mean frozen path edge score"
                )
        if self.foldable and self.intent_type == "Comparative" and len(branches) != 2:
            errors.append(
                "a foldable Comparative manifest must contain exactly two branches"
            )
        if self.path_continuous and any("discontinuous" in error for error in errors):
            errors.append("path_continuous is true but branch continuity failed")
        if self.foldable:
            if not self.path_steps:
                errors.append("a foldable manifest must contain at least one path step")
            if not self.path_continuous:
                errors.append("a foldable manifest must have path_continuous=true")
            if not self.core_chunk_ids:
                errors.append("a foldable manifest must contain core_chunk_ids")
            if not self.branch_complete:
                errors.append("a foldable manifest must have branch_complete=true")
            if self.fallback_to_graph_naive:
                errors.append("a foldable manifest cannot fall back to graph_naive")
            if self.fallback_reason:
                errors.append("a foldable manifest cannot record a fallback_reason")
        elif not self.fallback_reason:
            errors.append("a non-foldable manifest must record fallback_reason")
        elif not self.fallback_to_graph_naive:
            errors.append("a non-foldable manifest must fall back to graph_naive")
        return errors

    def validate(self) -> "FoldManifest":
        errors = self.validation_errors()
        if errors:
            raise ManifestValidationError(
                f"invalid manifest {self.query_id!r}: " + "; ".join(errors)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "implementation_version": self.implementation_version,
            "query_id": self.query_id,
            "dataset": self.dataset,
            "context_id": self.context_id,
            "question": self.question,
            "intent_type": self.intent_type,
            "intent_strategy": self.intent_strategy,
            "anchor_entities": list(self.anchor_entities),
            "path_policy": self.path_policy,
            "path_steps": [step.to_dict() for step in self.path_steps],
            "candidate_chunk_ids": list(self.candidate_chunk_ids),
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "core_chunk_ids": list(self.core_chunk_ids),
            "peripheral_chunk_ids": list(self.peripheral_chunk_ids),
            "score_order_chunk_ids": list(self.score_order_chunk_ids),
            "path_order_chunk_ids": list(self.path_order_chunk_ids),
            "chunk_scores": [[chunk_id, score] for chunk_id, score in self.chunk_scores],
            "budget": self.budget,
            "token_budget": self.token_budget,
            "source_token_budget": self.source_token_budget,
            "trace_token_budget": self.trace_token_budget,
            "total_evidence_token_budget": self.total_evidence_token_budget,
            "foldable": self.foldable,
            "fallback_reason": self.fallback_reason,
            "fallback_to_graph_naive": self.fallback_to_graph_naive,
            "path_confidence": self.path_confidence,
            "path_continuous": self.path_continuous,
            "branch_complete": self.branch_complete,
            "anchor_margin": self.anchor_margin,
            "anchor_source": self.anchor_source,
            "llm_reranker_used": self.llm_reranker_used,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FoldManifest":
        data = dict(value)
        return cls(
            query_id=data.get("query_id", ""),
            dataset=data.get("dataset", ""),
            context_id=data.get("context_id", ""),
            question=data.get("question", ""),
            intent_type=data.get("intent_type", ""),
            intent_strategy=data.get("intent_strategy", ""),
            anchor_entities=tuple(data.get("anchor_entities") or ()),
            path_policy=data.get("path_policy", ""),
            path_steps=tuple(PathStep.from_dict(item) for item in (data.get("path_steps") or ())),
            candidate_chunk_ids=tuple(data.get("candidate_chunk_ids") or ()),
            selected_chunk_ids=tuple(data.get("selected_chunk_ids") or ()),
            core_chunk_ids=tuple(data.get("core_chunk_ids") or ()),
            peripheral_chunk_ids=tuple(data.get("peripheral_chunk_ids") or ()),
            score_order_chunk_ids=tuple(data.get("score_order_chunk_ids") or ()),
            path_order_chunk_ids=tuple(data.get("path_order_chunk_ids") or ()),
            chunk_scores=data.get("chunk_scores") or (),
            budget=data.get("budget", 3),
            token_budget=data.get("token_budget", 0),
            source_token_budget=data.get("source_token_budget", 1400),
            trace_token_budget=data.get("trace_token_budget", 96),
            total_evidence_token_budget=data.get("total_evidence_token_budget", 1536),
            foldable=data.get("foldable", False),
            fallback_reason=data.get("fallback_reason", ""),
            fallback_to_graph_naive=data.get("fallback_to_graph_naive", False),
            path_confidence=data.get("path_confidence", 0.0),
            path_continuous=data.get("path_continuous", False),
            branch_complete=data.get("branch_complete", True),
            anchor_margin=data.get("anchor_margin", 0.0),
            anchor_source=data.get("anchor_source", ""),
            llm_reranker_used=data.get("llm_reranker_used", False),
            manifest_version=data.get("manifest_version", MANIFEST_VERSION),
            implementation_version=data.get(
                "implementation_version", IMPLEMENTATION_VERSION
            ),
        )


def save_jsonl(
    path: str | Path,
    manifests: Iterable[FoldManifest],
    *,
    validate: bool = True,
) -> Path:
    """Write manifests as canonical JSONL and return the resolved input path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    seen_query_ids: set[str] = set()
    for raw in manifests:
        manifest = raw if isinstance(raw, FoldManifest) else FoldManifest.from_dict(raw)
        if validate:
            manifest.validate()
        if manifest.query_id in seen_query_ids:
            raise ManifestValidationError(
                f"duplicate query_id in manifest file: {manifest.query_id!r}"
            )
        seen_query_ids.add(manifest.query_id)
        lines.append(manifest.canonical_json())
    payload = "".join(f"{line}\n" for line in lines)
    destination.write_text(payload, encoding="utf-8", newline="\n")
    return destination


def load_jsonl(
    path: str | Path,
    *,
    validate: bool = True,
) -> list[FoldManifest]:
    """Read a canonical or ordinary JSONL manifest file."""

    source = Path(path)
    manifests: list[FoldManifest] = []
    seen_query_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                manifest = FoldManifest.from_dict(value)
                if validate:
                    manifest.validate()
            except (TypeError, ValueError, ManifestValidationError) as exc:
                raise ManifestValidationError(
                    f"invalid manifest at {source}:{line_number}: {exc}"
                ) from exc
            if manifest.query_id in seen_query_ids:
                raise ManifestValidationError(
                    f"duplicate query_id at {source}:{line_number}: {manifest.query_id!r}"
                )
            seen_query_ids.add(manifest.query_id)
            manifests.append(manifest)
    return manifests


# Explicit names are useful at call sites and retain a concise save/load API.
save_manifest_jsonl = save_jsonl
load_manifest_jsonl = load_jsonl


__all__ = [
    "FoldManifest",
    "IMPLEMENTATION_VERSION",
    "SUPPORTED_IMPLEMENTATION_VERSIONS",
    "MANIFEST_VERSION",
    "ManifestValidationError",
    "PathStep",
    "canonical_json",
    "load_jsonl",
    "load_manifest_jsonl",
    "save_jsonl",
    "save_manifest_jsonl",
    "stable_sha256",
]
