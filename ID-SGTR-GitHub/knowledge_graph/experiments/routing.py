"""Deterministic helpers for context-first semantic routing."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any


_GENERIC_ENTITY_NAMES = {
    "actor", "actress", "album", "author", "book", "city", "company",
    "country", "director", "film", "government", "language", "music",
    "person", "place", "song", "writer", "year",
}


def _normalized_phrase(value: Any) -> str:
    return " ".join(re.findall(r"\w+", str(value).casefold(), flags=re.UNICODE))


@dataclass(frozen=True)
class SeedCandidate:
    """Auditable entity candidate used by the adaptive seed selector."""

    name: str
    combined_score: float
    retrieval_score: float
    specificity: float
    degree: int
    exact_query_match: bool
    generic_hub: bool
    original_order: int


def rank_seed_candidates(
    frame: Any,
    graph: Any,
    query: str = "",
    limit: int = 20,
    generic_degree_threshold: int = 50,
) -> list[SeedCandidate]:
    """Rank and deduplicate local entities using confidence and specificity."""
    ranked: list[SeedCandidate] = []
    seen: set[str] = set()
    normalized_query = _normalized_phrase(query)
    for order, (_, row) in enumerate(frame.iterrows()):
        name = str(row.get("Standard_Entity", "")).strip()
        normalized_name = _normalized_phrase(name)
        if not name or not normalized_name or normalized_name in seen:
            continue
        seen.add(normalized_name)
        retrieval_score = float(row.get("Score", 0.0) or 0.0)
        degree = int(graph.degree(name)) if name in graph else 0
        degree_specificity = 1.0 / (1.0 + math.log1p(degree))
        token_count = len(re.findall(r"\w+", name, flags=re.UNICODE))
        name_specificity = min(token_count, 4) / 4.0
        specificity = 0.7 * degree_specificity + 0.3 * name_specificity
        combined_score = 0.9 * retrieval_score + 0.1 * specificity
        exact_query_match = bool(
            normalized_name
            and re.search(
                rf"(?<!\w){re.escape(normalized_name)}(?!\w)",
                normalized_query,
                flags=re.UNICODE,
            )
        )
        lexical_generic = (
            token_count <= 2 and normalized_name in _GENERIC_ENTITY_NAMES
        )
        generic_hub = lexical_generic or (
            token_count <= 2 and degree >= max(1, generic_degree_threshold)
        )
        ranked.append(SeedCandidate(
            name=name,
            combined_score=combined_score,
            retrieval_score=retrieval_score,
            specificity=specificity,
            degree=degree,
            exact_query_match=exact_query_match,
            generic_hub=generic_hub,
            original_order=order,
        ))
    ranked.sort(key=lambda item: (
        -item.combined_score,
        -item.retrieval_score,
        item.original_order,
        item.name.casefold(),
    ))
    return ranked[:max(1, int(limit))]


def rank_seed_entities(frame: Any, graph: Any, limit: int = 15) -> list[str]:
    """Backward-compatible name-only view of deterministic seed ranking."""
    return [
        candidate.name
        for candidate in rank_seed_candidates(frame, graph, limit=limit)
    ]


def seed_rerank_decision(
    candidates: list[SeedCandidate],
    *,
    seed_limit: int = 5,
    margin_threshold: float = 0.04,
) -> tuple[bool, str, float, bool]:
    """Decide whether deterministic Top-k is safe or needs an LLM reranker."""
    if not candidates:
        return False, "no_candidates", 0.0, False
    top = candidates[:max(1, int(seed_limit))]
    margin = (
        max(0.0, candidates[0].combined_score - candidates[1].combined_score)
        if len(candidates) > 1 else candidates[0].combined_score
    )
    generic_hub_risk = any(candidate.generic_hub for candidate in top)
    if len(candidates) <= seed_limit:
        return False, "candidate_count_within_budget", margin, generic_hub_risk
    if generic_hub_risk:
        return True, "generic_hub_risk", margin, True
    if not any(candidate.exact_query_match for candidate in top):
        return True, "no_exact_query_anchor", margin, False
    if margin < max(0.0, float(margin_threshold)):
        return True, "seed_margin_below_threshold", margin, False
    return False, "deterministic_high_confidence", margin, False


def seed_rerank_prompt(
    query: str,
    candidates: list[SeedCandidate],
    descriptions: dict[str, str] | None = None,
    *,
    seed_limit: int = 5,
) -> str:
    """Build the shared low-confidence seed selection prompt."""
    descriptions = descriptions or {}
    rows = []
    for index, candidate in enumerate(candidates):
        description = " ".join(
            str(descriptions.get(candidate.name, "")).split()
        )[:240]
        rows.append(
            f"ID {index}: {candidate.name} "
            f"(retrieval={candidate.retrieval_score:.4f}, "
            f"degree={candidate.degree}, "
            f"exact_query_match={str(candidate.exact_query_match).lower()}, "
            f"info={description or 'n/a'})"
        )
    rendered = "\n".join(rows)
    return f"""You are selecting seed entities for multi-hop graph reasoning.

Question: {query}

Candidate entities:
{rendered}

Select between 1 and {max(1, int(seed_limit))} entities that are the best
starting points for answering the complete question.

Rules:
1. Preserve entities explicitly mentioned in the question.
2. For comparison questions, preserve the anchors for both branches.
3. Prefer specific named entities over generic high-degree concepts.
4. Do not select an entity merely because it is topically related.
5. Return no more than {max(1, int(seed_limit))} IDs.

Return ONLY a JSON integer array, for example: [0, 3].
"""


def parse_selected_ids(response: Any, upper_bound: int, limit: int = 5) -> list[int]:
    """Parse only an explicit JSON-like integer array; ignore incidental numbers."""
    text = str(getattr(response, "content", response)).strip()
    match = re.search(r"\[[\d,\s]*\]", text)
    if not match:
        return []
    try:
        values = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return []
    selected = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, int) and 0 <= value < upper_bound and value not in selected:
            selected.append(value)
        if len(selected) >= limit:
            break
    return selected


def context_rerank_prompt(query: str, candidate_passages: list[list[str]]) -> str:
    """Build a compact prompt for choosing one complete evidence context."""
    sections = []
    for index, passages in enumerate(candidate_passages):
        rendered = "\n".join(
            f"  Passage {offset + 1}: {passage}"
            for offset, passage in enumerate(passages)
        )
        sections.append(f"Candidate {index}:\n{rendered}")
    candidates = "\n\n".join(sections)
    return f"""You are selecting one evidence package for multi-hop question answering.

Question: {query}

{candidates}

Choose the single candidate whose passages collectively provide the strongest path to answer the question. Prefer complete multi-hop coverage over superficial word overlap. Do not answer the question.

Return ONLY a JSON array containing exactly one candidate integer ID.
Example: [1]
"""
