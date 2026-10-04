"""Pre-registered secondary strata for Topology Folding v2 analysis.

The categories below are fixed in code so a result file cannot silently define
its own favourable slices.  Empty categories are retained with ``count=0``.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


_METRICS = (
    "em",
    "f1",
    "precision",
    "recall",
    "evidence_recall",
    "evidence_precision",
    "complete_evidence_set",
    "answer_calls",
    "input_tokens",
    "output_tokens",
    "total_time_s",
    "evidence_tokens",
)
_BOOL_VALUES = ("true", "false", "unknown")
_INTENT_VALUES = (
    "Comparative", "Reasoning", "Retrieval", "Default", "unknown",
)
_PATH_VALUES = ("0", "1", "2", "3", "4+", "unknown")
_SELECTED_COUNT_VALUES = ("0", "1", "2", "3", "4+", "unknown")
_COMPLETENESS_VALUES = ("complete", "incomplete", "unknown")
_FALLBACK_REASON_VALUES = (
    "none",
    "no_reliable_query_anchor",
    "comparative_branch_incomplete",
    "no_continuous_relation_path",
    "path_confidence_below_threshold",
    "core_below_loose_threshold",
    "other",
    "unknown",
)


def _bool_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes"}:
        return "true"
    if text in {"0", "false", "no"}:
        return "false"
    return "unknown"


def _bounded_integer_label(value: object, *, upper: int) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    if number < 0:
        return "unknown"
    return f"{upper}+" if number >= upper else str(number)


def _intent_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text in _INTENT_VALUES[:-1] else "unknown"


def _completeness_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    return "complete" if number > 0 else "incomplete"


def _fallback_reason_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if not text:
        return "none"
    return text if text in _FALLBACK_REASON_VALUES[1:-2] else "other"


def _summarize(
    frame: pd.DataFrame,
    *,
    stratum: str,
    value: str,
    total_count: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "stratum": stratum,
        "value": value,
        "count": int(len(frame)),
        "coverage": float(len(frame) / total_count) if total_count else 0.0,
    }
    for metric in _METRICS:
        if metric not in frame:
            continue
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        result[metric] = (
            float(numeric.mean()) if numeric.notna().any() else float("nan")
        )
    return result


def _fixed_groups(
    frame: pd.DataFrame,
    *,
    column: str,
    categories: tuple[str, ...],
    normalizer: Callable[[object], str],
) -> list[dict[str, object]]:
    if column not in frame:
        return []
    labels = frame[column].map(normalizer)
    return [
        _summarize(
            frame.loc[labels.eq(category)],
            stratum=column,
            value=category,
            total_count=len(frame),
        )
        for category in categories
    ]


def stratify_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Return overall and fixed secondary strata without adaptive slicing."""

    if frame.empty:
        raise ValueError("results must not be empty")
    if "query_id" not in frame:
        raise ValueError("results must contain query_id")
    if frame["query_id"].astype(str).duplicated().any():
        raise ValueError("results must contain unique query_id values")

    rows = [_summarize(
        frame, stratum="overall", value="all", total_count=len(frame),
    )]
    for column in (
        "foldable", "folding_fallback", "path_continuous",
        "branch_complete", "manifest_order_changed",
    ):
        rows.extend(_fixed_groups(
            frame, column=column, categories=_BOOL_VALUES,
            normalizer=_bool_label,
        ))
    rows.extend(_fixed_groups(
        frame, column="intent_type", categories=_INTENT_VALUES,
        normalizer=_intent_label,
    ))
    rows.extend(_fixed_groups(
        frame, column="path_length", categories=_PATH_VALUES,
        normalizer=lambda value: _bounded_integer_label(value, upper=4),
    ))
    rows.extend(_fixed_groups(
        frame, column="selected_count", categories=_SELECTED_COUNT_VALUES,
        normalizer=lambda value: _bounded_integer_label(value, upper=4),
    ))
    rows.extend(_fixed_groups(
        frame, column="complete_evidence_set",
        categories=_COMPLETENESS_VALUES,
        normalizer=_completeness_label,
    ))
    rows.extend(_fixed_groups(
        frame, column="folding_fallback_reason",
        categories=_FALLBACK_REASON_VALUES,
        normalizer=_fallback_reason_label,
    ))
    return pd.DataFrame(rows)


__all__ = ["stratify_results"]
