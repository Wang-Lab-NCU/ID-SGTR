"""Provider-independent cleanup for concise QA answers without extra LLM calls."""

from __future__ import annotations

import re
from typing import Any


_NEGATIVE_ANSWERS = (
    "not found",
    "no information",
    "information is missing",
    "cannot answer",
    "unable to answer",
    "doesn't mention",
    "not mentioned",
    "not provided",
    "not explicitly stated",
    "not explicitly mentioned",
    "need more evidence",
    "n/a",
)


def normalize_short_answer(value: Any) -> str:
    """Return a concise visible answer span using deterministic text rules.

    The cleanup deliberately avoids semantic guessing. It only removes model
    boilerplate, Markdown wrappers, and explicit non-answer messages.
    """
    text = str(getattr(value, "content", value) or "").strip()
    # Qwen's OpenAI-compatible reasoning parser can return an empty visible
    # ``content`` field while retaining the model's explicit answer marker in
    # provider reasoning metadata.  This happens most often on the legacy
    # graph fallback path.  Recover only an explicitly marked final answer;
    # never treat arbitrary chain-of-thought as an answer.
    if not text and not isinstance(value, (str, bytes)):
        extra = getattr(value, "additional_kwargs", {}) or {}
        metadata = getattr(value, "response_metadata", {}) or {}
        reasoning = str(
            extra.get("reasoning") or extra.get("reasoning_content")
            or metadata.get("reasoning") or metadata.get("reasoning_content")
            or getattr(value, "reasoning", "") or ""
        ).strip()
        if "Final Answer:" in reasoning:
            text = reasoning.rsplit("Final Answer:", 1)[-1].strip()
    if not text:
        return ""

    if "Final Answer:" in text:
        text = text.rsplit("Final Answer:", 1)[-1].strip()

    # Llama commonly emits "The final answer is X" even when an exact span is
    # requested. Use the last occurrence so constructions such as
    # "A is not the answer, but the answer is B" resolve to B.
    matches = list(re.finditer(
        r"\b(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*",
        text,
        flags=re.IGNORECASE,
    ))
    if matches:
        text = text[matches[-1].end():].strip()

    text = text.splitlines()[0].strip()
    text = text.replace("**", "").replace("__", "")
    text = text.strip().strip("`").strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    if text.endswith("."):
        text = text[:-1].strip()

    lowered = text.lower()
    if not text or any(pattern in lowered for pattern in _NEGATIVE_ANSWERS):
        return ""

    binary = re.match(r"^(yes|no)\b", text, flags=re.IGNORECASE)
    if binary:
        return binary.group(1).lower()
    return text
