"""Evidence-sufficiency checks for Stage-0 early answers."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "there", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "with",
}


def _normalise(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _question_terms(question: str) -> set[str]:
    return {
        token
        for token in _normalise(question).split()
        if len(token) >= 3 and token not in _STOP_WORDS
    }


def _chunk_id_set(raw_ids: Iterable[Any]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in raw_ids:
        value = str(raw).strip()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def parse_supporting_refs(text: Any) -> list[str]:
    """Parse both ``[1, 4]`` and common model variants like ``[Ref 1, Ref 4]``."""
    match = re.search(
        r"Supporting\s+Refs?\s*:\s*\[(.*?)\]",
        str(text or ""),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    values: list[str] = []
    for raw in re.split(r"[,;\s]+", match.group(1)):
        value = raw.strip().strip("'\"`[]()")
        if not value or value.casefold() in {"ref", "refs", "reference", "source"}:
            continue
        value = re.sub(r"^(?:ref(?:erence)?|source)[:#-]*", "", value, flags=re.I)
        value = value.strip().strip("'\"`[]()#:")
        if value:
            values.append(value)
    return _chunk_id_set(values)


def parse_stage0_confidence(text: Any) -> str:
    """Parse the model's explicit Stage-0 answer confidence."""
    value = str(text or "")
    if re.search(r"^\s*DEFER\s*$", value, re.IGNORECASE | re.MULTILINE):
        return "defer"
    match = re.search(
        r"(?:Answer\s+)?Confidence\s*:\s*(HIGH|LOW)",
        value,
        re.IGNORECASE,
    )
    return match.group(1).casefold() if match else ""


def _edge_chunk_ids(data: Mapping[str, Any]) -> set[str]:
    return {str(value) for value in data.get("chunk_ids", [])}


def _cited_topology_connected(graph: Any, cited_ids: Sequence[str]) -> tuple[bool, list[str]]:
    """Return whether all cited chunks participate in one connected edge set."""
    if len(cited_ids) <= 1:
        return True, []
    if graph is None:
        return False, list(cited_ids)

    cited = set(cited_ids)
    represented: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    for source, target, data in graph.edges(data=True):
        overlap = cited.intersection(_edge_chunk_ids(data))
        if not overlap:
            continue
        represented.update(overlap)
        source = str(source)
        target = str(target)
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    missing = sorted(cited - represented)
    if missing or not adjacency:
        return False, missing

    start = next(iter(adjacency))
    visited = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for neighbour in adjacency.get(current, ()):
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return len(visited) == len(adjacency), []


@dataclass(frozen=True)
class Stage0GateResult:
    accepted: bool
    policy: str
    cited_refs: list[str] = field(default_factory=list)
    refs_inferred: bool = False
    answer_grounded: bool = False
    query_coverage: float = 0.0
    path_connected: bool = False
    rejection_reason: str = ""
    model_confidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "policy": self.policy,
            "cited_refs": list(self.cited_refs),
            "refs_inferred": self.refs_inferred,
            "answer_grounded": self.answer_grounded,
            "query_coverage": self.query_coverage,
            "path_connected": self.path_connected,
            "rejection_reason": self.rejection_reason,
            "model_confidence": self.model_confidence,
        }


def _infer_supporting_refs(
    query: str,
    answer: str,
    items: Mapping[str, Any],
    graph: Any,
    *,
    relational_min_refs: int,
) -> list[str]:
    """Infer the smallest well-grounded Stage-0 evidence combination.

    Stage 0 contains at most three chunks, so exhaustive subset enumeration is
    deterministic and negligible compared with a model call.
    """
    answer_norm = _normalise(answer)
    is_boolean = answer_norm in {"yes", "no"}
    question_terms = _question_terms(query)
    candidates: list[tuple[float, int, tuple[str, ...]]] = []
    item_ids = list(items)
    max_size = min(3, len(item_ids))

    for size in range(1, max_size + 1):
        if is_boolean and size < relational_min_refs:
            continue
        for subset in combinations(item_ids, size):
            combined = _normalise(" ".join(str(items[ref].text) for ref in subset))
            if not is_boolean and answer_norm not in combined:
                continue
            combined_terms = set(combined.split())
            coverage = (
                len(question_terms.intersection(combined_terms)) / len(question_terms)
                if question_terms else 1.0
            )
            connected, _ = _cited_topology_connected(graph, subset)
            if size > 1 and not connected:
                continue
            # Prefer stronger query coverage, then the smallest sufficient set.
            candidates.append((coverage, -size, subset))

    if not candidates:
        return []
    candidates.sort(key=lambda value: (-value[0], -value[1], value[2]))
    return list(candidates[0][2])


def verify_stage0_answer(
    *,
    query: str,
    answer: str,
    cited_refs: Iterable[Any],
    stage0_items: Iterable[Any],
    graph: Any = None,
    policy: str | None = None,
    confidence: str | None = None,
) -> Stage0GateResult:
    """Verify a Stage-0 answer without another model call.

    Policies:
    - ``legacy``/``always``: reproduce the original unconditional early exit.
    - ``never``: disable Stage-0 early exit.
    - ``evidence_verified``: require valid citations and deterministic support.
    - ``confidence_grounded``: require explicit HIGH confidence, valid citations,
      and an answer grounded in the cited Stage-0 text.
    - ``confidence_coverage``: additionally require sufficient question-term
      coverage, a configurable minimum number of supporting references, and a
      connected cited path.  This is intended for backbones whose self-reported
      confidence is not calibrated well enough to be used as the sole exit gate.
    """
    selected_policy = (policy or os.getenv("ID_SGTR_STAGE0_POLICY", "legacy")).strip().lower()
    model_confidence = str(confidence or "").strip().casefold()
    refs = _chunk_id_set(cited_refs)
    if selected_policy in {"legacy", "always"}:
        return Stage0GateResult(
            accepted=True, policy=selected_policy, cited_refs=refs,
            answer_grounded=True, query_coverage=1.0, path_connected=True,
        )
    if selected_policy == "never":
        return Stage0GateResult(False, selected_policy, refs, rejection_reason="policy_never")
    confidence_policy = selected_policy in {
        "confidence_grounded", "confidence_coverage",
    }
    if selected_policy not in {
        "evidence_verified", "confidence_grounded", "confidence_coverage",
    }:
        raise ValueError(f"unsupported ID_SGTR_STAGE0_POLICY: {selected_policy}")

    items = {str(item.chunk_id): item for item in stage0_items}
    relational_min_refs = max(
        1, int(os.getenv("ID_SGTR_STAGE0_RELATIONAL_MIN_REFS", "2"))
    )
    refs_inferred = False
    if confidence_policy and model_confidence != "high":
        return Stage0GateResult(
            False, selected_policy, refs,
            rejection_reason="model_not_high_confidence",
            model_confidence=model_confidence,
        )
    if not refs:
        if (
            confidence_policy
            and _normalise(answer) in {"yes", "no"}
        ):
            # Boolean answers cannot be matched as literal spans. HIGH confidence
            # binds the answer to the complete evidence window that the model saw.
            refs = list(items)
        else:
            refs = _infer_supporting_refs(
                query, answer, items, graph,
                relational_min_refs=(
                    relational_min_refs
                    if selected_policy == "evidence_verified"
                    else 1
                ),
            )
        refs_inferred = bool(refs)
    if not refs:
        return Stage0GateResult(
            False, selected_policy, refs, refs_inferred=False,
            rejection_reason="no_grounded_evidence_set",
            model_confidence=model_confidence,
        )
    invalid = [ref for ref in refs if ref not in items]
    if invalid:
        return Stage0GateResult(
            False, selected_policy, refs, refs_inferred=False,
            rejection_reason="invalid_citations",
            model_confidence=model_confidence,
        )

    cited_text = " ".join(str(items[ref].text) for ref in refs)
    normalised_answer = _normalise(answer)
    normalised_text = _normalise(cited_text)
    is_boolean = normalised_answer in {"yes", "no"}
    answer_grounded = bool(normalised_answer) and (
        is_boolean or normalised_answer in normalised_text
    )

    terms = _question_terms(query)
    text_terms = set(normalised_text.split())
    query_coverage = len(terms.intersection(text_terms)) / len(terms) if terms else 1.0
    min_coverage = float(os.getenv("ID_SGTR_STAGE0_MIN_QUERY_COVERAGE", "0.35"))
    path_connected, _ = _cited_topology_connected(graph, refs)
    if not answer_grounded:
        reason = "boolean_requires_multiple_refs" if is_boolean else "answer_not_grounded"
        return Stage0GateResult(
            False, selected_policy, refs, refs_inferred, False,
            query_coverage, path_connected, reason, model_confidence,
        )
    if selected_policy == "confidence_grounded":
        return Stage0GateResult(
            True, selected_policy, refs, refs_inferred, True,
            query_coverage, path_connected, "", model_confidence,
        )
    if selected_policy == "confidence_coverage":
        confidence_min_refs = max(
            1, int(os.getenv("ID_SGTR_STAGE0_CONFIDENCE_MIN_REFS", "2"))
        )
        if len(refs) < confidence_min_refs:
            return Stage0GateResult(
                False, selected_policy, refs, refs_inferred, True,
                query_coverage, path_connected,
                "insufficient_supporting_refs", model_confidence,
            )
        if query_coverage < min_coverage:
            return Stage0GateResult(
                False, selected_policy, refs, refs_inferred, True,
                query_coverage, path_connected,
                "insufficient_query_coverage", model_confidence,
            )
        if len(refs) > 1 and not path_connected:
            return Stage0GateResult(
                False, selected_policy, refs, refs_inferred, True,
                query_coverage, False,
                "disconnected_support", model_confidence,
            )
        return Stage0GateResult(
            True, selected_policy, refs, refs_inferred, True,
            query_coverage, path_connected, "", model_confidence,
        )
    if is_boolean and len(refs) < relational_min_refs:
        return Stage0GateResult(
            False, selected_policy, refs, refs_inferred, True,
            query_coverage, path_connected,
            "boolean_requires_multiple_refs",
        )
    if query_coverage < min_coverage:
        return Stage0GateResult(
            False, selected_policy, refs, refs_inferred, True,
            query_coverage, path_connected,
            "insufficient_query_coverage",
        )
    if len(refs) > 1 and not path_connected:
        return Stage0GateResult(
            False, selected_policy, refs, refs_inferred, True,
            query_coverage, False,
            "disconnected_support",
        )
    return Stage0GateResult(
        True, selected_policy, refs, refs_inferred, True,
        query_coverage, path_connected, "", model_confidence,
    )
