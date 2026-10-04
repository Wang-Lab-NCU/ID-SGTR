"""Question-aligned evidence selection over frozen runtime candidates.

The v1.5 runtime protocol freezes the final evidence window produced by the
online engine.  This module defines a separate v1.6 selector that is allowed
to choose a different B-sized subset, but only from the pre-answer runtime
candidate pool.  Selection is gold-independent and rewards paths that cover
the semantic relation slots expressed by the question.

The resulting manifest remains the immutable boundary for the controlled
representation factorial: every renderer receives exactly the same selected
chunk IDs.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

from .path_manifest import FoldManifest, PathStep
from .runtime_manifest import (
    RuntimeEdge,
    RuntimeManifestBuilder,
    _clean_id,
    _ids,
    _intent,
    _phrase_in_question,
    _score,
)
from .topology_folding_v2 import (
    TopologyFolderV2,
    _EDGE_CUE_ALIASES,
    _contains_alias,
    _relation_slot_groups,
    _target_compatibility,
    _tokens,
)


QUESTION_ALIGNED_IMPLEMENTATION_VERSION = (
    "topology-folding-runtime-trace-v1.6-question-aligned-selection"
)
CONSERVATIVE_QUESTION_ALIGNED_IMPLEMENTATION_VERSION = (
    "topology-folding-runtime-trace-v1.7-conservative-question-aligned"
)
FACTORIAL_CLEAN_IMPLEMENTATION_VERSION = (
    "topology-folding-runtime-trace-v1.8-factorial-clean-question-aligned"
)


def _lexical_overlap(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class QuestionAlignedRuntimeManifestBuilder(RuntimeManifestBuilder):
    """Select a B-sized question-aligned proof path from runtime candidates."""

    def __init__(
        self,
        *args: Any,
        beam_width: int = 24,
        conservative_gate: bool = False,
        factorial_clean: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        self.beam_width = int(beam_width)
        self.conservative_gate = bool(conservative_gate)
        self.factorial_clean = bool(factorial_clean)
        self.implementation_version = (
            FACTORIAL_CLEAN_IMPLEMENTATION_VERSION
            if self.factorial_clean
            else CONSERVATIVE_QUESTION_ALIGNED_IMPLEMENTATION_VERSION
            if self.conservative_gate
            else QUESTION_ALIGNED_IMPLEMENTATION_VERSION
        )

    @staticmethod
    def _slot_matches(
        edge: RuntimeEdge,
        slots: Sequence[frozenset[str]],
    ) -> frozenset[int]:
        edge_text = f"{edge.source} {edge.relation} {edge.target}"
        return frozenset(
            index
            for index, slot in enumerate(slots)
            if any(
                _contains_alias(edge_text, _EDGE_CUE_ALIASES[cue])
                for cue in slot
            )
        )

    @staticmethod
    def _normalized_scores(
        candidate_ids: Sequence[str],
        scores: Mapping[str, float],
    ) -> dict[str, float]:
        values = [float(scores.get(chunk_id, 0.0)) for chunk_id in candidate_ids]
        if not values:
            return {}
        low, high = min(values), max(values)
        if math.isclose(low, high):
            return {chunk_id: 0.5 for chunk_id in candidate_ids}
        return {
            chunk_id: (float(scores.get(chunk_id, low)) - low) / (high - low)
            for chunk_id in candidate_ids
        }

    def _edge_value(
        self,
        edge: RuntimeEdge,
        *,
        question: str,
        slots: Sequence[frozenset[str]],
        covered: frozenset[int],
        semantic_scores: Mapping[str, float],
        first_step: bool,
        previous_hop: int,
    ) -> tuple[float, frozenset[int]]:
        matched = self._slot_matches(edge, slots)
        new_slots = matched - covered
        flattened_cues = frozenset(
            cue for index in new_slots for cue in slots[index]
        )
        compatibility = _target_compatibility(edge.target, flattened_cues)
        if new_slots and compatibility < 0:
            return -math.inf, matched

        semantic = float(semantic_scores.get(edge.chunk_id, 0.0))
        slot_gain = len(new_slots) / max(1, len(slots))
        lexical = _lexical_overlap(
            f"{edge.source} {edge.relation} {edge.target}",
            question,
        )
        anchor = float(
            first_step
            and (
                _phrase_in_question(edge.source, question)
                or _phrase_in_question(edge.target, question)
            )
        )
        hop_consistency = float(previous_hop <= 0 or edge.hop >= previous_hop)
        endpoint = max(0.0, compatibility)
        explicit = float(edge.explicitly_chosen)

        value = (
            0.30 * semantic
            + 0.25 * slot_gain
            + 0.15 * lexical
            + 0.10 * anchor
            + 0.08 * endpoint
            + 0.07 * explicit
            + 0.05 * hop_consistency
        )
        if matched and not new_slots:
            value -= 0.08
        return value, matched

    @staticmethod
    def _merge_edges(
        runtime_edges: Sequence[RuntimeEdge],
        trace_edges: Sequence[RuntimeEdge],
    ) -> tuple[RuntimeEdge, ...]:
        merged: dict[tuple[str, str, str, str], RuntimeEdge] = {}
        for edge in tuple(trace_edges) + tuple(runtime_edges):
            previous = merged.get(edge.signature)
            if previous is None or (
                edge.explicitly_chosen,
                edge.score,
                -edge.hop,
                -edge.position,
            ) > (
                previous.explicitly_chosen,
                previous.score,
                -previous.hop,
                -previous.position,
            ):
                merged[edge.signature] = edge
        return tuple(merged.values())

    def _best_question_chain(
        self,
        edges: Sequence[RuntimeEdge],
        *,
        question: str,
        intent_type: str,
        semantic_scores: Mapping[str, float],
    ) -> tuple[RuntimeEdge, ...]:
        slots = _relation_slot_groups(question)
        target_steps = TopologyFolderV2._infer_target_steps(
            question, intent_type, self.budget,
        )
        adjacency: dict[str, list[RuntimeEdge]] = {}
        for edge in edges:
            adjacency.setdefault(edge.source, []).append(edge)

        # state = path, covered slots, cumulative score
        beam: list[tuple[tuple[RuntimeEdge, ...], frozenset[int], float]] = []
        for edge in edges:
            value, matched = self._edge_value(
                edge,
                question=question,
                slots=slots,
                covered=frozenset(),
                semantic_scores=semantic_scores,
                first_step=True,
                previous_hop=0,
            )
            if math.isfinite(value):
                beam.append(((replace(edge, score=value),), matched, value))
        beam.sort(
            key=lambda state: (
                len(state[1]),
                _phrase_in_question(state[0][0].source, question),
                state[2],
            ),
            reverse=True,
        )
        beam = beam[: self.beam_width]
        completed: list[
            tuple[tuple[RuntimeEdge, ...], frozenset[int], float]
        ] = []

        for _ in range(1, target_steps):
            expanded: list[
                tuple[tuple[RuntimeEdge, ...], frozenset[int], float]
            ] = []
            for path, covered, total in beam:
                used_signatures = {edge.signature for edge in path}
                used_chunks = {edge.chunk_id for edge in path}
                visited = {path[0].source} | {edge.target for edge in path}
                for edge in adjacency.get(path[-1].target, ()):
                    if edge.signature in used_signatures or edge.target in visited:
                        continue
                    if edge.chunk_id not in used_chunks and len(used_chunks) >= self.budget:
                        continue
                    value, matched = self._edge_value(
                        edge,
                        question=question,
                        slots=slots,
                        covered=covered,
                        semantic_scores=semantic_scores,
                        first_step=False,
                        previous_hop=path[-1].hop,
                    )
                    if not math.isfinite(value):
                        continue
                    expanded.append((
                        path + (replace(edge, score=value),),
                        covered | matched,
                        total + value,
                    ))
            if not expanded:
                beam = []
                break
            expanded.sort(
                key=lambda state: (
                    len(state[1]),
                    len({edge.chunk_id for edge in state[0]}),
                    state[2],
                    tuple(edge.signature for edge in state[0]),
                ),
                reverse=True,
            )
            beam = expanded[: self.beam_width]
        completed.extend(beam)

        required = frozenset(range(len(slots)))
        eligible = [
            state for state in completed
            if len(state[0]) == target_steps
            and required.issubset(state[1])
            and (
                _phrase_in_question(state[0][0].source, question)
                or _phrase_in_question(state[0][0].target, question)
            )
        ]
        if not eligible:
            return ()
        best = max(
            eligible,
            key=lambda state: (
                len(state[1]),
                state[2],
                len({edge.chunk_id for edge in state[0]}),
                tuple(edge.signature for edge in state[0]),
            ),
        )
        return best[0]

    def _question_aligned_comparative(
        self,
        edges: Sequence[RuntimeEdge],
        *,
        question: str,
        semantic_scores: Mapping[str, float],
    ) -> tuple[RuntimeEdge, ...]:
        slots = _relation_slot_groups(question)
        adjusted: list[RuntimeEdge] = []
        for edge in edges:
            value, matched = self._edge_value(
                edge,
                question=question,
                slots=slots,
                covered=frozenset(),
                semantic_scores=semantic_scores,
                first_step=True,
                previous_hop=0,
            )
            if not math.isfinite(value):
                continue
            if slots and not matched:
                continue
            if not (
                _phrase_in_question(edge.source, question)
                or _phrase_in_question(edge.target, question)
            ):
                continue
            adjusted.append(replace(edge, score=value))
        selected, complete = self._comparative_branches(adjusted, question)
        return selected if complete else ()

    def _safe_fallback(
        self,
        *,
        row: Mapping[str, Any],
        candidate_ids: Sequence[str],
        original_selected_ids: Sequence[str],
        semantic_scores: Mapping[str, float],
        intent_type: str,
        reason: str,
    ) -> FoldManifest:
        """Fall back without changing the online engine's evidence set."""
        selected_ids = tuple(original_selected_ids[: self.budget])
        if not selected_ids:
            selected_ids = tuple(sorted(
                candidate_ids,
                key=lambda value: (-semantic_scores.get(value, 0.0), value),
            )[: self.budget])
        # _fallback retains exactly this set and deterministically renders the
        # Graph-Naive score order required by the manifest contract.
        return self._fallback(
            row=row,
            candidate_ids=candidate_ids,
            selected_ids=selected_ids,
            scores=semantic_scores,
            intent_type=intent_type,
            reason=reason,
        )

    def build(self, row: Mapping[str, Any]) -> FoldManifest:
        # Gold answer/evidence and the previous prediction are deliberately
        # absent from this method.
        context_id = _clean_id(row.get("context_id"))
        local_ids = self.context_chunks.get(context_id, ())
        local_set = set(local_ids)
        candidates = self._candidate_records(row.get("candidate_evidence"))
        original_selected_ids = list(dict.fromkeys(
            chunk_id for chunk_id in _ids(row.get("retrieved_evidence"))
            if chunk_id in local_set
        ))[: self.budget]

        candidate_ids = list(dict.fromkeys(
            _clean_id(candidate.get("chunk_id"))
            for candidate in candidates
            if _clean_id(candidate.get("chunk_id")) in local_set
        ))
        # The captured final evidence must always remain inside the frozen
        # candidate universe so every failed v1.6 selection can revert safely.
        candidate_ids.extend(
            chunk_id for chunk_id in original_selected_ids
            if chunk_id not in candidate_ids
        )
        intent_type = _intent(row.get("strategy"))
        raw_scores: dict[str, float] = {}
        for candidate in candidates:
            chunk_id = _clean_id(candidate.get("chunk_id"))
            if chunk_id in candidate_ids:
                raw_scores[chunk_id] = max(
                    raw_scores.get(chunk_id, -math.inf),
                    _score(candidate.get("score")),
                )
        for chunk_id in candidate_ids:
            raw_scores.setdefault(chunk_id, 0.0)
        semantic_scores = self._normalized_scores(candidate_ids, raw_scores)

        if not candidate_ids:
            return self._safe_fallback(
                row=row,
                candidate_ids=local_ids,
                original_selected_ids=original_selected_ids,
                semantic_scores={},
                intent_type=intent_type,
                reason="question_aligned_no_candidates",
            )

        runtime_edges, _ = self._chosen_edge_records(
            row.get("runtime_path_steps"),
            candidates,
            candidate_ids,
            context_id,
        )
        trace_edges, _ = self._edge_records(
            candidates, candidate_ids, context_id,
        )
        edges = self._merge_edges(runtime_edges, trace_edges)
        question = str(row.get("question", "")).strip()

        if intent_type == "Comparative":
            selected_edges = self._question_aligned_comparative(
                edges,
                question=question,
                semantic_scores=semantic_scores,
            )
            if not selected_edges:
                return self._safe_fallback(
                    row=row,
                    candidate_ids=candidate_ids,
                    original_selected_ids=original_selected_ids,
                    semantic_scores=semantic_scores,
                    intent_type=intent_type,
                    reason="question_aligned_comparative_incomplete",
                )
            steps = tuple(
                PathStep(
                    step=1,
                    branch=branch,
                    source_entity=edge.source,
                    relation=edge.relation,
                    target_entity=edge.target,
                    supporting_chunk_id=edge.chunk_id,
                    edge_score=edge.score,
                    query_score=semantic_scores.get(edge.chunk_id, 0.0),
                    canonical_source_entity=edge.canonical_source,
                    canonical_target_entity=edge.canonical_target,
                    traversal_direction=edge.direction,
                )
                for branch, edge in zip(("A", "B"), selected_edges)
            )
            anchors = tuple(edge.source for edge in selected_edges)
            path_policy = "question_aligned_comparative_slots"
        else:
            selected_edges = self._best_question_chain(
                edges,
                question=question,
                intent_type=intent_type,
                semantic_scores=semantic_scores,
            )
            if not selected_edges:
                return self._safe_fallback(
                    row=row,
                    candidate_ids=candidate_ids,
                    original_selected_ids=original_selected_ids,
                    semantic_scores=semantic_scores,
                    intent_type=intent_type,
                    reason="question_aligned_no_complete_path",
                )
            steps = tuple(
                PathStep(
                    step=index,
                    branch="main",
                    source_entity=edge.source,
                    relation=edge.relation,
                    target_entity=edge.target,
                    supporting_chunk_id=edge.chunk_id,
                    edge_score=edge.score,
                    query_score=semantic_scores.get(edge.chunk_id, 0.0),
                    canonical_source_entity=edge.canonical_source,
                    canonical_target_entity=edge.canonical_target,
                    traversal_direction=edge.direction,
                )
                for index, edge in enumerate(selected_edges, start=1)
            )
            anchors = (selected_edges[0].source,)
            path_policy = "question_aligned_relation_slots"

        core_ids = tuple(dict.fromkeys(
            step.supporting_chunk_id for step in steps
        ))
        adjusted_scores = dict(semantic_scores)
        for step in steps:
            adjusted_scores[step.supporting_chunk_id] = max(
                adjusted_scores.get(step.supporting_chunk_id, 0.0),
                step.edge_score,
            )
        fillers = [
            chunk_id for chunk_id in sorted(
                candidate_ids,
                key=lambda value: (-adjusted_scores[value], value),
            )
            if chunk_id not in core_ids
        ]
        selected_ids = tuple(
            (list(core_ids) + fillers)[: self.budget]
        )
        if self.conservative_gate:
            baseline = RuntimeManifestBuilder.build(self, row)
            baseline_selected = set(baseline.selected_chunk_ids)
            proposed_selected = set(selected_ids)
            added = proposed_selected - baseline_selected
            removed = baseline_selected - proposed_selected

            rejection_reason = ""
            if intent_type == "Retrieval":
                rejection_reason = "conservative_retrieval_intent"
            elif len(steps) < 2:
                rejection_reason = "conservative_path_too_short"
            elif len(added) > 1:
                rejection_reason = "conservative_multiple_replacements"
            elif removed & set(baseline.core_chunk_ids):
                rejection_reason = "conservative_old_core_eviction"

            if rejection_reason:
                return self._safe_fallback(
                    row=row,
                    candidate_ids=candidate_ids,
                    original_selected_ids=original_selected_ids,
                    semantic_scores=semantic_scores,
                    intent_type=intent_type,
                    reason=rejection_reason,
                )

        frozen_scores = (
            semantic_scores if self.factorial_clean else adjusted_scores
        )
        score_order = tuple(sorted(
            selected_ids,
            key=lambda value: (-frozen_scores[value], value),
        ))
        peripheral_ids = tuple(
            chunk_id for chunk_id in score_order if chunk_id not in core_ids
        )
        path_order = core_ids + peripheral_ids
        confidence = sum(step.edge_score for step in steps) / len(steps)

        return FoldManifest(
            query_id=_clean_id(row.get("query_id")),
            dataset=self.dataset,
            context_id=context_id,
            question=question,
            intent_type=intent_type,
            intent_strategy=str(row.get("strategy", "")),
            anchor_entities=anchors,
            path_policy=path_policy,
            path_steps=steps,
            candidate_chunk_ids=tuple(candidate_ids),
            selected_chunk_ids=score_order,
            core_chunk_ids=core_ids,
            peripheral_chunk_ids=peripheral_ids,
            score_order_chunk_ids=score_order,
            path_order_chunk_ids=path_order,
            chunk_scores=tuple(
                (chunk_id, frozen_scores.get(chunk_id, 0.0))
                for chunk_id in candidate_ids
            ),
            budget=self.budget,
            source_token_budget=self.source_token_budget,
            trace_token_budget=self.trace_token_budget,
            total_evidence_token_budget=self.total_evidence_token_budget,
            foldable=True,
            fallback_reason="",
            fallback_to_graph_naive=False,
            path_confidence=confidence,
            path_continuous=True,
            branch_complete=True,
            anchor_source="question_aligned_runtime_candidates",
            implementation_version=self.implementation_version,
        ).validate()


__all__ = [
    "CONSERVATIVE_QUESTION_ALIGNED_IMPLEMENTATION_VERSION",
    "FACTORIAL_CLEAN_IMPLEMENTATION_VERSION",
    "QUESTION_ALIGNED_IMPLEMENTATION_VERSION",
    "QuestionAlignedRuntimeManifestBuilder",
]
