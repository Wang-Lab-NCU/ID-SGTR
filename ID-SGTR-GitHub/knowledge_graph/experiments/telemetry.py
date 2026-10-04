"""Per-query online efficiency telemetry for ID-SGTR and baselines."""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any, Iterator


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_message_text(item) for item in value)
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(getattr(value, "content", value))


def estimate_tokens(value: Any) -> int:
    """Provider-independent approximation used only when usage metadata is absent."""
    text = _message_text(value)
    return max(0, (len(text) + 3) // 4)


@dataclass
class QueryTelemetry:
    query_id: str = ""
    total_llm_calls: int = 0
    answer_calls: int = 0
    auxiliary_calls: int = 0
    retrieval_rounds: int = 0
    retrieval_time_s: float = 0.0
    generation_time_s: float = 0.0
    total_time_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_prompt_sha256: str = ""
    formatter_prompt_sha256: str = ""
    finalization_policy: str = ""
    token_count_estimated: bool = False
    fallback: bool = False
    routed_context: str = ""
    routing_initial_context: str = ""
    routing_score: float = 0.0
    routing_margin: float = 0.0
    routing_candidate_count: int = 0
    routing_reranked: bool = False
    routing_top_contexts: list[str] = field(default_factory=list)
    rerank_backend: str = ""
    context_rerank_calls: int = 0
    context_rerank_documents: int = 0
    local_rerank_calls: int = 0
    local_rerank_documents: int = 0
    rerank_time_s: float = 0.0
    seed_candidate_count: int = 0
    seed_selected_count: int = 0
    seed_selection_policy: str = ""
    seed_selection_reason: str = ""
    seed_margin: float = 0.0
    seed_generic_hub_risk: bool = False
    seed_reranker_used: bool = False
    stage0_policy: str = ""
    stage0_seed_entities: list[str] = field(default_factory=list)
    stage0_decision: dict[str, Any] = field(default_factory=dict)
    stage0_replay_applied: bool = False
    stage0_candidate_answer: str = ""
    stage0_supporting_refs: list[str] = field(default_factory=list)
    stage0_refs_inferred: bool = False
    stage0_answer_grounded: bool = False
    stage0_query_coverage: float = 0.0
    stage0_path_connected: bool = False
    stage0_model_confidence: str = ""
    stage0_early_exit: bool = False
    stage0_rejection_reason: str = ""
    # Snapshot taken immediately after the Stage-0 answer.  It permits an exact
    # counterfactual early-exit replay from a single force-continue execution.
    stage0_total_llm_calls: int = 0
    stage0_answer_calls: int = 0
    stage0_auxiliary_calls: int = 0
    stage0_input_tokens: int = 0
    stage0_output_tokens: int = 0
    stage0_elapsed_s: float = 0.0
    hop_gate_attempts: int = 0
    hop_gate_exits: int = 0
    hop_gate_rejections: int = 0
    hop_gate_last_hop: int = 0
    hop_gate_last_confidence: str = ""
    hop_gate_last_supporting_refs: list[str] = field(default_factory=list)
    hop_gate_last_rejection_reason: str = ""
    terminal_recovery_applied: bool = False
    terminal_recovery_contexts: list[str] = field(default_factory=list)
    terminal_recovery_raw_candidate_count: int = 0
    terminal_recovery_candidate_count: int = 0
    terminal_recovery_duplicate_count: int = 0
    terminal_recovery_selected_count: int = 0
    terminal_recovery_elapsed_s: float = 0.0
    terminal_recovery_grounded: bool = False
    terminal_recovery_confidence: str = ""
    terminal_recovery_rejection_reason: str = ""
    retrieved_evidence: list[str] = field(default_factory=list)
    accessed_evidence: list[str] = field(default_factory=list)
    candidate_evidence: list[dict[str, Any]] = field(default_factory=list)
    # Per-hop graph transitions explicitly selected by the reasoning agent.
    # Candidate expansion edges that were merely exposed to the model are not
    # included here.  This is the authoritative input for runtime manifests.
    runtime_path_steps: list[dict[str, Any]] = field(default_factory=list)
    # Frozen-manifest / Topology Folding v2 audit fields.  ``folding_fallback``
    # is intentionally distinct from the end-to-end system ``fallback`` flag.
    manifest_version: str = ""
    implementation_version: str = ""
    manifest_sha256: str = ""
    path_policy: str = ""
    path_length: int = 0
    path_continuous: bool = False
    path_confidence: float = 0.0
    foldable: bool = False
    folding_fallback: bool = False
    folding_fallback_reason: str = ""
    selected_count: int = 0
    unused_budget: int = 0
    core_chunk_ids: list[str] = field(default_factory=list)
    peripheral_chunk_ids: list[str] = field(default_factory=list)
    trace_count: int = 0
    trace_tokens: int = 0
    source_tokens: int = 0
    evidence_tokens: int = 0
    order_changed: bool = False
    # Manifest-level order contrast, identical across all four renderers.
    # ``order_changed`` above records whether the current renderer applied it.
    manifest_order_changed: bool = False
    branch_complete: bool = True
    anchor_margin: float = 0.0
    anchor_source: str = ""
    llm_reranker_used: bool = False
    call_latencies_s: list[float] = field(default_factory=list)
    _started_at: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        self._started_at = time.perf_counter()

    def stop(self) -> None:
        if self._started_at:
            self.total_time_s = time.perf_counter() - self._started_at
            # When explicit phases are not used, everything outside answer-model
            # invocations is online retrieval/routing time (including auxiliary LLMs).
            if self.retrieval_time_s == 0.0:
                self.retrieval_time_s = max(0.0, self.total_time_s - self.generation_time_s)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                if name == "retrieval":
                    self.retrieval_time_s += elapsed
                elif name == "generation":
                    self.generation_time_s += elapsed
                else:
                    raise ValueError(f"unknown telemetry phase: {name}")

    def record_call(self, role: str, latency_s: float, input_value: Any, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None) or {}
        response_meta = getattr(response, "response_metadata", None) or {}
        token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}
        input_tokens = usage.get("input_tokens", token_usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", token_usage.get("completion_tokens"))
        estimated = input_tokens is None or output_tokens is None
        if input_tokens is None:
            input_tokens = estimate_tokens(input_value)
        if output_tokens is None:
            output_tokens = estimate_tokens(response)
        with self._lock:
            self.total_llm_calls += 1
            if role == "answer":
                self.answer_calls += 1
            else:
                self.auxiliary_calls += 1
            self.input_tokens += int(input_tokens)
            self.output_tokens += int(output_tokens)
            self.token_count_estimated = self.token_count_estimated or estimated
            self.call_latencies_s.append(float(latency_s))
            if role == "answer":
                self.generation_time_s += float(latency_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }


class TrackedChatModel:
    """Transparent LangChain-style model proxy that counts each ``invoke`` call."""

    def __init__(self, model: Any, telemetry: QueryTelemetry, role: str):
        self._model = model
        self._telemetry = telemetry
        self._role = role

    def invoke(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        response = self._model.invoke(value, *args, **kwargs)
        latency = time.perf_counter() - started
        self._telemetry.record_call(self._role, latency, value, response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


_CURRENT_TELEMETRY: ContextVar[QueryTelemetry | None] = ContextVar("id_sgtr_telemetry", default=None)


@contextmanager
def bind_telemetry(telemetry: QueryTelemetry) -> Iterator[None]:
    """Bind telemetry to one thread/task without mutating the shared engine."""
    token = _CURRENT_TELEMETRY.set(telemetry)
    try:
        yield
    finally:
        _CURRENT_TELEMETRY.reset(token)


def record_evidence(chunk_ids: Any) -> None:
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    with telemetry._lock:
        current: list[str] = []
        current_seen: set[str] = set()
        for chunk_id in chunk_ids:
            value = str(chunk_id)
            if value not in current_seen:
                current.append(value)
                current_seen.add(value)
        # retrieved_evidence is the evidence in the latest/final prompt;
        # accessed_evidence is the union across all online stages.
        telemetry.retrieved_evidence = current
        seen = set(telemetry.accessed_evidence)
        for value in current:
            if value not in seen:
                telemetry.accessed_evidence.append(value)
                seen.add(value)


def record_candidates(items: Any, hop: int) -> None:
    """Persist a text-free runtime trajectory for fixed-path replay.

    The trace deliberately contains no source text, gold label, or generated
    answer.  It records only graph traversal metadata already produced before
    answer synthesis, so a later controlled experiment can freeze the exact
    online path without reconstructing it from gold annotations.
    """
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    with telemetry._lock:
        for order, item in enumerate(items):
            topology_trace = []
            for raw in getattr(item, "topology_trace", ()) or ():
                if isinstance(raw, dict):
                    trace_hop = raw.get("hop", hop)
                    trace_position = raw.get("path_position", order)
                    trace_triple = raw.get("triple", "")
                elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
                    trace_hop, trace_position, trace_triple = raw[:3]
                else:
                    continue
                topology_trace.append({
                    "hop": int(trace_hop),
                    "path_position": int(trace_position),
                    "triple": str(trace_triple),
                })
            telemetry.candidate_evidence.append({
                "chunk_id": str(item.chunk_id),
                "score": float(item.score),
                "hop": int(getattr(item, "hop", hop) or hop),
                "raw_path_position": int(
                    getattr(item, "path_position", order) or order
                ),
                "path_position": int(hop) * 1000 + int(item.path_position or order),
                "triple": str(item.triple),
                "topology_trace": topology_trace,
                "is_structural": bool(
                    topology_trace or str(getattr(item, "triple", "")).strip()
                ),
            })


def record_runtime_step(
    hop: int,
    active_nodes: Any,
    relevant_nodes: Any,
    next_nodes: Any,
    chosen_edges: Any,
) -> None:
    """Record only graph edges selected by the online reasoning decision.

    ``chosen_edges`` must already be restricted to candidate paths whose
    target occurs in the parsed ``Relevant Nodes`` or ``Next Hop`` fields.
    The function stores structural IDs only; no source text, gold label, model
    answer, or post-hoc path reconstruction is permitted.
    """

    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return

    def clean_nodes(values: Any) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values or ():
            item = str(value).strip()
            if item and item not in seen:
                output.append(item)
                seen.add(item)
        return output

    normalized_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for position, raw in enumerate(chosen_edges or (), start=1):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source_entity", raw.get("u", ""))).strip()
        relation = str(raw.get("relation", raw.get("rel", ""))).strip()
        target = str(raw.get("target_entity", raw.get("v", ""))).strip()
        chunk_ids = tuple(clean_nodes(raw.get("chunk_ids", ())))
        if not source or not relation or not target or not chunk_ids:
            continue
        signature = (source, relation, target, chunk_ids)
        if signature in seen_edges:
            continue
        seen_edges.add(signature)
        normalized_edges.append({
            "source_entity": source,
            "relation": relation,
            "target_entity": target,
            "chunk_ids": list(chunk_ids),
            "path_position": int(raw.get("path_position", position) or position),
        })

    with telemetry._lock:
        telemetry.runtime_path_steps.append({
            "hop": int(hop),
            "active_nodes": clean_nodes(active_nodes),
            "relevant_nodes": clean_nodes(relevant_nodes),
            "next_nodes": clean_nodes(next_nodes),
            "chosen_edges": normalized_edges,
            "executed_nodes": [],
            "executed_edges": [],
        })


def record_runtime_execution(
    hop: int,
    executed_nodes: Any,
    executed_edges: Any,
) -> None:
    """Attach the frontier that the engine actually executes next.

    This is deliberately narrower than candidate exposure: an edge is stored
    only when its target enters the next active frontier.  The manifest may
    use these edges when the model did not emit a parseable explicit choice.
    """

    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return

    def clean_nodes(values: Any) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values or ():
            item = str(value).strip()
            if item and item not in seen:
                output.append(item)
                seen.add(item)
        return output

    normalized_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for position, raw in enumerate(executed_edges or (), start=1):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source_entity", raw.get("u", ""))).strip()
        relation = str(raw.get("relation", raw.get("rel", ""))).strip()
        target = str(raw.get("target_entity", raw.get("v", ""))).strip()
        chunk_ids = tuple(clean_nodes(raw.get("chunk_ids", ())))
        if not source or not relation or not target or not chunk_ids:
            continue
        signature = (source, relation, target, chunk_ids)
        if signature in seen_edges:
            continue
        seen_edges.add(signature)
        normalized_edges.append({
            "source_entity": source,
            "relation": relation,
            "target_entity": target,
            "chunk_ids": list(chunk_ids),
            "path_position": int(raw.get("path_position", position) or position),
        })

    with telemetry._lock:
        target_step = next((
            step for step in reversed(telemetry.runtime_path_steps)
            if int(step.get("hop", -1)) == int(hop)
        ), None)
        if target_step is None:
            target_step = {
                "hop": int(hop),
                "active_nodes": [],
                "relevant_nodes": [],
                "next_nodes": [],
                "chosen_edges": [],
            }
            telemetry.runtime_path_steps.append(target_step)
        target_step["executed_nodes"] = clean_nodes(executed_nodes)
        target_step["executed_edges"] = normalized_edges


def record_routing(
    context_id: Any,
    score: float,
    margin: float,
    candidate_count: int,
    *,
    initial_context: Any | None = None,
    reranked: bool = False,
    top_contexts: Any = (),
) -> None:
    """Record the deterministic context-routing decision for one query."""
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    with telemetry._lock:
        telemetry.routed_context = str(context_id)
        telemetry.routing_initial_context = str(
            context_id if initial_context is None else initial_context
        )
        telemetry.routing_score = float(score)
        telemetry.routing_margin = float(margin)
        telemetry.routing_candidate_count = int(candidate_count)
        telemetry.routing_reranked = bool(reranked)
        telemetry.routing_top_contexts = [str(value) for value in top_contexts]


def record_rerank(
    scope: str,
    document_count: int,
    latency_s: float,
    *,
    backend: str,
) -> None:
    """Record Cross-Encoder work separately from answer/auxiliary LLM calls."""
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    with telemetry._lock:
        telemetry.rerank_backend = str(backend)
        telemetry.rerank_time_s += float(latency_s)
        if str(scope) == "context":
            telemetry.context_rerank_calls += 1
            telemetry.context_rerank_documents += int(document_count)
        else:
            telemetry.local_rerank_calls += 1
            telemetry.local_rerank_documents += int(document_count)


def record_stage0_gate(answer: Any, result: Any) -> None:
    """Persist the deterministic Stage-0 evidence-sufficiency decision."""
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    with telemetry._lock:
        telemetry.stage0_policy = str(result.policy)
        telemetry.stage0_candidate_answer = str(answer or "")
        telemetry.stage0_supporting_refs = [str(value) for value in result.cited_refs]
        telemetry.stage0_refs_inferred = bool(result.refs_inferred)
        telemetry.stage0_answer_grounded = bool(result.answer_grounded)
        telemetry.stage0_query_coverage = float(result.query_coverage)
        telemetry.stage0_path_connected = bool(result.path_connected)
        telemetry.stage0_model_confidence = str(
            getattr(result, "model_confidence", "") or ""
        )
        telemetry.stage0_early_exit = bool(result.accepted)
        telemetry.stage0_rejection_reason = str(result.rejection_reason)


def record_hop_gate(hop: int, result: Any) -> None:
    """Record a confidence-grounded answerability decision after graph expansion."""
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    with telemetry._lock:
        telemetry.hop_gate_attempts += 1
        telemetry.hop_gate_exits += int(bool(result.accepted))
        telemetry.hop_gate_rejections += int(not bool(result.accepted))
        telemetry.hop_gate_last_hop = int(hop)
        telemetry.hop_gate_last_confidence = str(
            getattr(result, "model_confidence", "") or ""
        )
        telemetry.hop_gate_last_supporting_refs = [
            str(value) for value in getattr(result, "cited_refs", ())
        ]
        telemetry.hop_gate_last_rejection_reason = str(
            getattr(result, "rejection_reason", "") or ""
        )


def current_routing_recovery_contexts(
    *, margin_threshold: float = 0.06, limit: int = 3
) -> list[str]:
    """Return one locked context, or Top-k contexts when routing is uncertain."""
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return []
    with telemetry._lock:
        values = list(telemetry.routing_top_contexts)
        if not values and telemetry.routed_context:
            values = [telemetry.routed_context]
        count = max(1, int(limit)) if telemetry.routing_margin < margin_threshold else 1
        return [str(value) for value in values[:count]]


def record_terminal_recovery(
    *,
    contexts: Any,
    raw_candidate_count: int,
    candidate_count: int,
    selected_count: int,
    elapsed_s: float,
    result: Any,
) -> None:
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    with telemetry._lock:
        telemetry.terminal_recovery_applied = True
        telemetry.terminal_recovery_contexts = [str(value) for value in contexts or ()]
        telemetry.terminal_recovery_raw_candidate_count = int(raw_candidate_count)
        telemetry.terminal_recovery_candidate_count = int(candidate_count)
        telemetry.terminal_recovery_duplicate_count = max(
            0, int(raw_candidate_count) - int(candidate_count)
        )
        telemetry.terminal_recovery_selected_count = int(selected_count)
        telemetry.terminal_recovery_elapsed_s = max(0.0, float(elapsed_s))
        telemetry.terminal_recovery_grounded = bool(result.accepted)
        telemetry.terminal_recovery_confidence = str(
            getattr(result, "model_confidence", "") or ""
        )
        telemetry.terminal_recovery_rejection_reason = str(
            getattr(result, "rejection_reason", "") or ""
        )


def record_stage0_policy(
    answer: Any,
    *,
    proposed_final: bool,
    allow_early_exit: bool,
    reasoning_stress: bool,
    decision: Any = None,
    replay_applied: bool = False,
) -> None:
    """Record the isolated Stage-0 early-exit treatment used by local runs."""
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    accepted = bool(proposed_final and allow_early_exit and not reasoning_stress)
    if accepted:
        rejection_reason = ""
    elif not proposed_final:
        rejection_reason = "model_requested_graph_expansion"
    elif reasoning_stress:
        rejection_reason = "reasoning_stress"
    else:
        rejection_reason = "stage0_exit_disabled"
    with telemetry._lock:
        telemetry.stage0_policy = (
            "early_exit_enabled" if allow_early_exit else "force_continue"
        )
        if isinstance(decision, dict):
            telemetry.stage0_decision = {
                "is_final": bool(decision.get("is_final", proposed_final)),
                "answer": str(decision.get("answer", answer or "")),
                "relevant_nodes": [
                    str(value) for value in decision.get("relevant_nodes", []) or []
                ],
                "next_nodes": [
                    str(value) for value in decision.get("next_nodes", []) or []
                ],
            }
        telemetry.stage0_replay_applied = bool(replay_applied)
        telemetry.stage0_candidate_answer = str(answer or "") if proposed_final else ""
        telemetry.stage0_early_exit = accepted
        telemetry.stage0_rejection_reason = rejection_reason
        telemetry.stage0_total_llm_calls = int(telemetry.total_llm_calls)
        telemetry.stage0_answer_calls = int(telemetry.answer_calls)
        telemetry.stage0_auxiliary_calls = int(telemetry.auxiliary_calls)
        telemetry.stage0_input_tokens = int(telemetry.input_tokens)
        telemetry.stage0_output_tokens = int(telemetry.output_tokens)
        telemetry.stage0_elapsed_s = max(
            0.0,
            time.perf_counter() - telemetry._started_at,
        ) if telemetry._started_at else 0.0


def record_stage0_seeds(seeds: Any) -> None:
    """Record the exact semantic anchors entering Stage0."""
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    output: list[str] = []
    seen: set[str] = set()
    for raw in seeds or ():
        value = str(raw).strip()
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    with telemetry._lock:
        telemetry.stage0_seed_entities = output


def record_seed_selection(
    *,
    candidate_count: int,
    selected_count: int,
    policy: str,
    reason: str,
    margin: float,
    generic_hub_risk: bool,
    reranker_used: bool,
) -> None:
    """Record the shared Retrieval/Reasoning adaptive seed policy."""
    telemetry = _CURRENT_TELEMETRY.get()
    if telemetry is None:
        return
    with telemetry._lock:
        telemetry.seed_candidate_count = int(candidate_count)
        telemetry.seed_selected_count = int(selected_count)
        telemetry.seed_selection_policy = str(policy)
        telemetry.seed_selection_reason = str(reason)
        telemetry.seed_margin = max(0.0, float(margin))
        telemetry.seed_generic_hub_risk = bool(generic_hub_risk)
        telemetry.seed_reranker_used = bool(reranker_used)


class ContextTrackedChatModel:
    """Shared model proxy that routes accounting to the current query context."""

    def __init__(self, model: Any, role: str):
        self._model = model
        self._role = role

    def invoke(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        response = self._model.invoke(value, *args, **kwargs)
        telemetry = _CURRENT_TELEMETRY.get()
        if telemetry is not None:
            telemetry.record_call(self._role, time.perf_counter() - started, value, response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
