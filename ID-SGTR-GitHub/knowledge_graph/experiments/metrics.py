"""Standard QA/evidence metrics and paired bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def normalize_answer(value: object) -> str:
    """HotpotQA-style normalization applied identically to EM and token F1."""
    text = "" if value is None else str(value).lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: object, gold: object) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: object, gold: object) -> tuple[float, float, float]:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        score = float(prediction_tokens == gold_tokens)
        return score, score, score
    if (prediction_tokens[0] in {"yes", "no", "noanswer"} or gold_tokens[0] in {"yes", "no", "noanswer"}) \
            and prediction_tokens != gold_tokens:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall), precision, recall


def evidence_scores(retrieved: Iterable[object], gold: Iterable[object]) -> dict[str, float]:
    retrieved_set = {str(item) for item in retrieved}
    gold_set = {str(item) for item in gold}
    overlap = len(retrieved_set & gold_set)
    recall = overlap / len(gold_set) if gold_set else math.nan
    precision = overlap / len(retrieved_set) if retrieved_set else (0.0 if gold_set else math.nan)
    complete = float(bool(gold_set) and gold_set.issubset(retrieved_set))
    return {"evidence_recall": recall, "evidence_precision": precision, "complete_evidence_set": complete}


def evaluate_predictions(rows: Iterable[Mapping[str, object]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    per_query: list[dict[str, float]] = []
    for row in rows:
        f1, precision, recall = token_f1(row.get("prediction", ""), row.get("gold", ""))
        scores = {
            "em": exact_match(row.get("prediction", ""), row.get("gold", "")),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        if "retrieved_evidence" in row and "gold_evidence" in row:
            scores.update(evidence_scores(row["retrieved_evidence"], row["gold_evidence"]))
        per_query.append(scores)
    if not per_query:
        raise ValueError("cannot evaluate an empty prediction set")
    summary: dict[str, float] = {}
    for key in per_query[0]:
        values = [row[key] for row in per_query if not math.isnan(row[key])]
        summary[key] = sum(values) / len(values) if values else math.nan
    summary["count"] = float(len(per_query))
    # This invariant catches swapped/corrupt aggregate columns such as the reported MuSiQue row.
    if summary["f1"] + 1e-12 < summary["em"]:
        raise AssertionError("invalid QA metrics: mean token F1 cannot be below exact match")
    return summary, per_query


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    count: int
    mean_a: float
    mean_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    metric: str = "f1",
    samples: int = 10_000,
    seed: int = 42,
) -> BootstrapResult:
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("paired score arrays must be non-empty and have equal length")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    rng = random.Random(seed)
    count = len(scores_a)
    differences: list[float] = []
    for _ in range(samples):
        indexes = [rng.randrange(count) for _ in range(count)]
        differences.append(sum(scores_a[i] - scores_b[i] for i in indexes) / count)
    observed = sum(a - b for a, b in zip(scores_a, scores_b)) / count
    probability_le_zero = sum(value <= 0 for value in differences) / samples
    probability_ge_zero = sum(value >= 0 for value in differences) / samples
    return BootstrapResult(
        metric=metric,
        count=count,
        mean_a=sum(scores_a) / count,
        mean_b=sum(scores_b) / count,
        difference=observed,
        ci_low=_quantile(differences, 0.025),
        ci_high=_quantile(differences, 0.975),
        p_value=min(1.0, 2 * min(probability_le_zero, probability_ge_zero)),
    )
