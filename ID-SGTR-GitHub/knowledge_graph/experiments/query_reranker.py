"""Shared query-aware Cross-Encoder reranking for both evaluation settings.

Retrieval Setting uses the reranker twice:
1. rerank a small list of globally recalled contexts and lock one context;
2. rerank every chunk inside that context.

Reasoning Setting skips (1) and applies (2) to the benchmark-provided local
context.  The module is intentionally independent from gold annotations.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import requests


_CURRENT_QUERY: ContextVar[str] = ContextVar("id_sgtr_rerank_query", default="")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if not math.isfinite(low) or not math.isfinite(high) or high - low <= 1e-12:
        return [0.5 for _ in values]
    return [(float(value) - low) / (high - low) for value in values]


_BRIDGE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "was", "were", "with",
}


def blend_hop_bridge_scores(
    original_scores: dict[str, float],
    documents: dict[str, str],
    *,
    active_nodes: Iterable[object],
    relevant_entities: Iterable[object],
    candidate_paths: Sequence[dict[str, Any]],
    history_facts: Iterable[object] = (),
    bridge_weight: float | None = None,
) -> dict[str, float]:
    """Blend cached question relevance with deterministic hop bridge coverage.

    This head does not call an LLM or Cross-Encoder.  It rewards passages that
    mention the current frontier and cover relation vocabulary exposed by the
    actually expanded graph edges.  Stage0 scores remain the dominant signal,
    and a zero weight reproduces the frozen static-cache behavior exactly.
    """
    weight = (
        float(os.getenv("ID_SGTR_HOP_BRIDGE_WEIGHT", "0"))
        if bridge_weight is None else float(bridge_weight)
    )
    weight = min(1.0, max(0.0, weight))
    if weight <= 0.0 or not documents:
        return {str(key): float(value) for key, value in original_scores.items()}

    def phrases(values: Iterable[object]) -> list[str]:
        return list(dict.fromkeys(
            text
            for value in values
            if len(text := str(value).strip().casefold()) >= 3
        ))

    frontier = phrases(active_nodes)
    known = phrases(relevant_entities)
    endpoints = phrases(
        value
        for path in candidate_paths[:12]
        for value in (path.get("u", ""), path.get("v", ""))
    )
    relation_text = " ".join(
        str(path.get("rel", "")) for path in candidate_paths[:12]
    )
    relation_text += " " + " ".join(map(str, history_facts))
    relation_tokens = {
        token
        for token in re.findall(r"\w+", relation_text.casefold(), flags=re.UNICODE)
        if len(token) >= 3 and token not in _BRIDGE_STOPWORDS
    }

    ids = [str(item_id) for item_id in documents]
    bridge_raw: list[float] = []
    original_raw: list[float] = []
    for item_id in ids:
        text = str(documents[item_id]).casefold()
        tokens = set(re.findall(r"\w+", text, flags=re.UNICODE))
        frontier_hits = sum(phrase in text for phrase in frontier)
        known_hits = sum(phrase in text for phrase in known)
        endpoint_hits = sum(phrase in text for phrase in endpoints)
        relation_coverage = (
            len(tokens.intersection(relation_tokens)) / len(relation_tokens)
            if relation_tokens else 0.0
        )
        bridge_raw.append(
            2.0 * frontier_hits
            + 0.75 * known_hits
            + 0.75 * endpoint_hits
            + 2.0 * relation_coverage
        )
        original_raw.append(float(original_scores.get(item_id, 0.0)))

    original_norm = _minmax(original_raw)
    bridge_norm = _minmax(bridge_raw)
    return {
        item_id: (1.0 - weight) * original + weight * bridge
        for item_id, original, bridge in zip(ids, original_norm, bridge_norm)
    }


@contextmanager
def bind_rerank_query(query: str) -> Iterator[None]:
    """Bind the query per worker thread without mutating a shared engine."""
    token = _CURRENT_QUERY.set(str(query))
    try:
        yield
    finally:
        _CURRENT_QUERY.reset(token)


def current_rerank_query() -> str:
    return _CURRENT_QUERY.get()


@dataclass(frozen=True)
class RankedItem:
    item_id: str
    score: float
    cross_score: float
    base_score: float


class QueryCrossEncoder:
    """Lazy, thread-safe Cross-Encoder client with deterministic score fusion."""

    def __init__(self) -> None:
        self.enabled = _env_bool("ID_SGTR_RERANK_ENABLED", False)
        self.backend = os.getenv("ID_SGTR_RERANK_BACKEND", "local").strip().lower()
        self.model = os.getenv(
            "ID_SGTR_RERANK_MODEL",
            "BAAI/bge-reranker-v2-m3",
        ).strip()
        self.base_url = os.getenv(
            "ID_SGTR_RERANK_BASE_URL", "http://127.0.0.1:30001"
        ).rstrip("/")
        self.device = os.getenv("ID_SGTR_RERANK_DEVICE", "cpu").strip()
        self.batch_size = max(1, int(os.getenv("ID_SGTR_RERANK_BATCH_SIZE", "16")))
        self.max_length = max(64, int(os.getenv("ID_SGTR_RERANK_MAX_LENGTH", "512")))
        self.query_max_tokens = max(
            16,
            min(
                self.max_length // 2,
                int(os.getenv("ID_SGTR_RERANK_QUERY_MAX_TOKENS", "128")),
            ),
        )
        self.document_max_tokens = max(
            32,
            int(
                os.getenv(
                    "ID_SGTR_RERANK_DOCUMENT_MAX_TOKENS",
                    str(self.max_length - self.query_max_tokens),
                )
            ),
        )
        self.cross_weight = min(
            1.0, max(0.0, float(os.getenv("ID_SGTR_RERANK_CROSS_WEIGHT", "0.50")))
        )
        self.context_cross_weight = min(
            1.0,
            max(
                0.0,
                float(os.getenv("ID_SGTR_CONTEXT_RERANK_CROSS_WEIGHT", "0.50")),
            ),
        )
        self.conservative_gate = _env_bool(
            "ID_SGTR_RERANK_CONSERVATIVE_GATE", False
        )
        self.evidence_budget = max(
            1, int(os.getenv("ID_SGTR_EVIDENCE_BUDGET", "3"))
        )
        self.protected_base_count = max(
            0,
            min(
                self.evidence_budget,
                int(os.getenv("ID_SGTR_RERANK_PROTECTED_BASE_COUNT", "2")),
            ),
        )
        self.max_replacements = max(
            0, int(os.getenv("ID_SGTR_RERANK_MAX_REPLACEMENTS", "1"))
        )
        self.replacement_margin = max(
            0.0, float(os.getenv("ID_SGTR_RERANK_REPLACEMENT_MARGIN", "0.05"))
        )
        self.timeout = max(5.0, float(os.getenv("ID_SGTR_RERANK_TIMEOUT", "120")))
        self._tokenizer: Any = None
        self._model: Any = None
        self._load_lock = threading.Lock()
        # Local transformer inference must be serialized.  HTTP/vLLM can batch
        # concurrent requests by itself, so it does not take this lock.
        self._inference_lock = threading.Lock()

    def _load_local(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model, local_files_only=os.path.isdir(self.model)
            )
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model,
                local_files_only=os.path.isdir(self.model),
                torch_dtype=dtype,
            )
            self._model.to(self.device)
            self._model.eval()

    def _score_local(self, query: str, documents: Sequence[str]) -> list[float]:
        import torch

        self._load_local()
        scores: list[float] = []
        with self._inference_lock, torch.no_grad():
            for start in range(0, len(documents), self.batch_size):
                batch = documents[start : start + self.batch_size]
                pairs = [[query, document] for document in batch]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self._model(**inputs, return_dict=True).logits
                scores.extend(float(value) for value in logits.view(-1).float().cpu())
        return scores

    def _score_http(self, query: str, documents: Sequence[str]) -> list[float]:
        response = requests.post(
            f"{self.base_url}/rerank",
            json={
                "model": self.model,
                "query": query,
                "documents": list(documents),
                # vLLM rejects over-length score pairs unless truncation is
                # requested explicitly.  Match the local tokenizer backend:
                # keep the complete short query and the beginning of the
                # evidence chunk within the frozen reranker token budget.
                "truncate_prompt_tokens": self.max_length,
                "truncation_side": "right",
                "max_tokens_per_query": self.query_max_tokens,
                "max_tokens_per_doc": self.document_max_tokens,
            },
            timeout=self.timeout,
        )
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} rerank error: {response.text[:1000]}",
                response=response,
            )
        payload = response.json()
        results = payload.get("results", payload.get("data", []))
        by_index: dict[int, float] = {}
        for position, item in enumerate(results):
            index = int(item.get("index", position))
            score = item.get("relevance_score", item.get("score"))
            if score is not None:
                by_index[index] = float(score)
        if len(by_index) != len(documents):
            raise ValueError(
                f"rerank endpoint returned {len(by_index)} scores "
                f"for {len(documents)} documents"
            )
        return [by_index[index] for index in range(len(documents))]

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        if not self.enabled or not query.strip():
            return [0.0 for _ in documents]
        if self.backend == "http":
            return self._score_http(query, documents)
        if self.backend == "local":
            return self._score_local(query, documents)
        raise ValueError(f"unsupported reranker backend: {self.backend}")

    def _record(self, scope: str, count: int, started: float) -> None:
        try:
            from .telemetry import record_rerank
        except ImportError:
            from experiments.telemetry import record_rerank
        record_rerank(
            scope, count, time.perf_counter() - started, backend=self.backend
        )

    def _apply_conservative_gate(
        self,
        item_ids: Sequence[object],
        combined_scores: Sequence[float],
        base_scores: Sequence[float],
    ) -> list[float]:
        """Protect the dense/graph core and permit only confident replacements.

        The query-aware score may replace at most ``max_replacements`` members
        of the original top-B set.  The first ``protected_base_count`` original
        items can never be evicted.  Non-admitted candidates are moved below
        the selected B-set so later path bonuses cannot silently undo the gate.
        """
        output = [float(value) for value in combined_scores]
        if (
            not self.conservative_gate
            or self.evidence_budget >= len(item_ids)
            or self.max_replacements <= 0
        ):
            return output

        ids = [str(value) for value in item_ids]
        base_order = sorted(
            range(len(ids)),
            key=lambda index: (-float(base_scores[index]), ids[index]),
        )
        fused_order = sorted(
            range(len(ids)),
            key=lambda index: (-output[index], ids[index]),
        )
        base_top = list(base_order[: self.evidence_budget])
        protected = set(base_top[: self.protected_base_count])
        admitted = list(base_top)
        replacements = 0

        for candidate in fused_order:
            if candidate in admitted:
                continue
            removable = [
                index for index in admitted if index not in protected
            ]
            if not removable or replacements >= self.max_replacements:
                break
            victim = min(
                removable,
                key=lambda index: (output[index], float(base_scores[index]), ids[index]),
            )
            if output[candidate] < output[victim] + self.replacement_margin:
                continue
            admitted.remove(victim)
            admitted.append(candidate)
            replacements += 1

        admitted_set = set(admitted)
        if not admitted_set:
            return output
        floor = min(output[index] for index in admitted_set) - 1.0
        for rank, index in enumerate(fused_order):
            if index not in admitted_set:
                output[index] = floor - rank * 1e-6
        return output

    def rank(
        self,
        query: str,
        item_ids: Sequence[object],
        documents: Sequence[str],
        *,
        base_scores: Sequence[float] | None = None,
        cross_weight: float | None = None,
    ) -> list[RankedItem]:
        if len(item_ids) != len(documents):
            raise ValueError("item_ids and documents must have equal length")
        if base_scores is None:
            base_scores = [0.0 for _ in item_ids]
        if len(base_scores) != len(item_ids):
            raise ValueError("base_scores and item_ids must have equal length")

        effective_cross_weight = (
            self.cross_weight
            if cross_weight is None
            else min(1.0, max(0.0, float(cross_weight)))
        )
        if self.enabled:
            base_norm = _minmax([float(value) for value in base_scores])
            if effective_cross_weight > 0.0:
                started = time.perf_counter()
                cross_raw = self.score(query, documents)
                self._record("local", len(documents), started)
                cross_norm = _minmax(cross_raw)
            else:
                # Static-cache ablation: reuse Stage0 scores without issuing
                # another identical Cross-Encoder request at every hop.
                cross_raw = [0.0 for _ in item_ids]
                cross_norm = [0.0 for _ in item_ids]
            combined = [
                effective_cross_weight * cross
                + (1.0 - effective_cross_weight) * base
                for cross, base in zip(cross_norm, base_norm)
            ]
            combined = self._apply_conservative_gate(
                item_ids, combined, base_scores
            )
        else:
            cross_raw = [0.0 for _ in item_ids]
            combined = [float(value) for value in base_scores]

        ranked = [
            RankedItem(str(item_id), float(score), float(cross), float(base))
            for item_id, score, cross, base in zip(
                item_ids, combined, cross_raw, base_scores
            )
        ]
        ranked.sort(key=lambda item: (-item.score, item.item_id))
        return ranked

    def rerank_contexts(
        self,
        query: str,
        context_ids: Sequence[object],
        passages: Sequence[Sequence[str]],
        *,
        base_scores: Sequence[float],
    ) -> list[RankedItem]:
        """Aggregate passage-level Cross-Encoder scores into context scores."""
        if not (
            len(context_ids) == len(passages) == len(base_scores)
        ):
            raise ValueError("context rerank inputs must have equal length")
        if not context_ids:
            return []

        flattened: list[str] = []
        owners: list[int] = []
        for context_index, group in enumerate(passages):
            for passage in group:
                flattened.append(str(passage))
                owners.append(context_index)

        if not flattened:
            return self.rank(
                query, context_ids, [str(value) for value in context_ids],
                base_scores=base_scores,
            )

        started = time.perf_counter()
        raw_scores = self.score(query, flattened)
        self._record("context", len(flattened), started)
        grouped: list[list[float]] = [[] for _ in context_ids]
        for owner, score in zip(owners, raw_scores):
            grouped[owner].append(float(score))
        context_cross = [
            (0.65 * max(values) + 0.35 * sum(values) / len(values))
            if values else -math.inf
            for values in grouped
        ]
        cross_norm = _minmax(context_cross)
        base_norm = _minmax([float(value) for value in base_scores])
        combined = [
            self.context_cross_weight * cross
            + (1.0 - self.context_cross_weight) * base
            for cross, base in zip(cross_norm, base_norm)
        ]
        output = [
            RankedItem(str(context_id), float(score), float(cross), float(base))
            for context_id, score, cross, base in zip(
                context_ids, combined, context_cross, base_scores
            )
        ]
        output.sort(key=lambda item: (-item.score, item.item_id))
        return output


_RERANKER: QueryCrossEncoder | None = None
_RERANKER_LOCK = threading.Lock()


def get_query_reranker() -> QueryCrossEncoder:
    """Return one lazy model/client per Python process."""
    global _RERANKER
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:
                _RERANKER = QueryCrossEncoder()
    return _RERANKER
