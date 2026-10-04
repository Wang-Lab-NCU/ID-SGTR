"""Shared terminal recovery helpers for Retrieval and Reasoning settings."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


def _unique(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values or ():
        value = str(raw).strip()
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _normalized_text(value: Any) -> str:
    """Normalize exact-content duplicates without fuzzy semantic collapsing."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def deduplicate_chunk_ids_by_text(
    chunk_ids: Sequence[Any],
    text_getter: Callable[[str], Any],
    *,
    preferred_ids: Sequence[Any] = (),
) -> list[str]:
    """Remove cross-context copies while retaining a deterministic chunk ID."""
    candidates = _unique(chunk_ids)
    candidate_set = set(candidates)
    preferred = [
        value for value in _unique(preferred_ids) if value in candidate_set
    ]
    preferred_set = set(preferred)
    ordered = preferred + [
        value for value in candidates if value not in preferred_set
    ]

    output: list[str] = []
    seen_content: set[str] = set()
    for chunk_id in ordered:
        content = _normalized_text(text_getter(chunk_id))
        key = f"text:{content}" if content else f"id:{chunk_id}"
        if key in seen_content:
            continue
        seen_content.add(key)
        output.append(chunk_id)
    return output


def build_terminal_query(
    question: str,
    *,
    entities: Iterable[Any] = (),
    facts: Iterable[Any] = (),
) -> str:
    """Create a hop-aware reranking query without another LLM call."""
    entity_text = "; ".join(_unique(entities)[:8])
    fact_text = "; ".join(_unique(facts)[:8])
    parts = [f"Question: {str(question).strip()}"]
    if entity_text:
        parts.append(f"Discovered entities: {entity_text}")
    if fact_text:
        parts.append(f"Executed relation facts: {fact_text}")
    parts.append("Retrieve passages that complete the missing relation chain and directly support the final answer.")
    return "\n".join(parts)


def select_terminal_chunks(
    ranked_ids: Sequence[Any],
    prior_path_ids: Sequence[Any],
    *,
    budget: int = 5,
    text_getter: Callable[[str], Any] | None = None,
) -> list[str]:
    """Combine reranked passages and path core without duplicate content."""
    limit = max(1, int(budget))
    ranked = _unique(ranked_ids)
    prior = _unique(prior_path_ids)
    reserve = min(2, len(prior), max(0, limit - 1))

    selected: list[str] = []
    selected_keys: set[str] = set()

    def append_unique(chunk_id: str) -> None:
        if chunk_id in selected:
            return
        content = _normalized_text(text_getter(chunk_id)) if text_getter else ""
        key = f"text:{content}" if content else f"id:{chunk_id}"
        if key in selected_keys:
            return
        selected.append(chunk_id)
        selected_keys.add(key)

    for chunk_id in ranked[: max(1, limit - reserve)]:
        append_unique(chunk_id)
    for chunk_id in prior[:reserve]:
        append_unique(chunk_id)
    for chunk_id in ranked + prior:
        if len(selected) >= limit:
            break
        append_unique(chunk_id)
    return selected[:limit]


def build_terminal_prompt(question: str, evidence_lines: Sequence[str]) -> str:
    evidence = "\n".join(evidence_lines) if evidence_lines else "No source evidence was recovered."
    return f"""You are the final synthesis stage of a multi-hop QA system.

Question: {question}

Source evidence:
{evidence}

Infer the best-supported answer by combining the supplied passages. Resolve
multi-hop relations across passages; the answer does not need to occur verbatim
in one passage. For a comparison, resolve both branches before comparing them.
Do not abstain merely because an intermediate relation is implicit. Use only
the source evidence above and return exactly these three lines:
Final Answer: <single best-supported shortest answer>
Supporting Refs: <comma-separated Ref IDs>
Confidence: HIGH or LOW
"""


@dataclass(frozen=True)
class TerminalDraft:
    answer: str = ""
    confidence: str = ""
    supporting_refs: list[str] = field(default_factory=list)


def parse_terminal_response(value: Any) -> TerminalDraft:
    text = str(getattr(value, "content", value) or "").strip()
    answer_match = re.search(
        r"Final\s*Answer\s*:[ \t]*(.*?)(?=\r?\n[ \t]*(?:Supporting\s*Refs|Confidence)\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    confidence_match = re.search(
        r"Confidence\s*:\s*(HIGH|LOW)", text, flags=re.IGNORECASE
    )
    refs_match = re.search(
        r"Supporting\s*Refs\s*:[ \t]*([^\r\n]*)", text, flags=re.IGNORECASE
    )
    answer_block = answer_match.group(1).strip() if answer_match else ""
    answer_lines = answer_block.splitlines()
    answer = answer_lines[0].strip() if answer_lines else ""
    answer = answer.strip('`').strip('"').strip("'")
    refs = re.findall(r"(?:Ref\s*)?([A-Za-z0-9_.:-]+)", refs_match.group(1)) if refs_match else []
    refs = _unique(refs)
    confidence = confidence_match.group(1).lower() if confidence_match else ""
    return TerminalDraft(answer=answer, confidence=confidence, supporting_refs=refs)
