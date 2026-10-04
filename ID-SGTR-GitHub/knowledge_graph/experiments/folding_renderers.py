"""Pure renderers for the controlled Topology Folding v2 factorial.

The path builder owns evidence selection.  This module is intentionally unable
to retrieve or select chunks: it receives an immutable :class:`FoldManifest`
and a mapping containing the source text for every selected chunk.  The four
factorial cells therefore differ only in two representation factors:

=======================  ===========  =============
variant                  path order   aligned trace
=======================  ===========  =============
graph_naive              no           no
path_order_source        yes          no
trace_score_source       no           yes
topology_folding_v2      yes          yes
=======================  ===========  =============

Source truncation is computed once, before a variant is rendered, in a
canonical order.  Consequently every cell sees byte-for-byte identical source
fragments for a given chunk ID.  Trace lines are never invented: an incomplete
or over-budget trace triggers the manifest's conservative Graph-Naive
fallback.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

from .path_manifest import FoldManifest, PathStep


class FoldingRenderError(ValueError):
    """Raised when a manifest cannot be rendered without changing evidence."""


class TokenBudgetError(FoldingRenderError):
    """Raised when fixed labels alone cannot fit the configured hard budget."""


class FoldingVariant(str, Enum):
    """Controlled factorial cells plus the frozen lossless final policy."""

    GRAPH_NAIVE = "graph_naive"
    PATH_ORDER_SOURCE = "path_order_source"
    TRACE_SCORE_SOURCE = "trace_score_source"
    TOPOLOGY_FOLDING_V2 = "topology_folding_v2"
    TOPOLOGY_FOLDING_LOSSLESS = "topology_folding_lossless"


FACTORIAL_VARIANTS: tuple[FoldingVariant, ...] = (
    FoldingVariant.GRAPH_NAIVE,
    FoldingVariant.PATH_ORDER_SOURCE,
    FoldingVariant.TRACE_SCORE_SOURCE,
    FoldingVariant.TOPOLOGY_FOLDING_V2,
)

# The manifest freezes evidence selection; this version identifies only the
# presentation policy used to expose that evidence to the answer model.
FOLDING_RENDERER_VERSION = "topology-renderer-v2.2-lossless-final"


class TokenCounter(Protocol):
    """Structural protocol accepted for an injectable token counter."""

    def __call__(self, text: str) -> int:
        ...


@dataclass(frozen=True)
class RenderBudgets:
    """Hard evidence budgets measured by the configured token counter.

    ``source_tokens`` includes ``[Source <id>]`` labels.  ``trace_tokens``
    includes step and branch labels.  ``total_tokens`` applies to the final
    rendered evidence string, including separators.
    """

    source_tokens: int = 1400
    trace_tokens: int = 96
    total_tokens: int = 1536

    def __post_init__(self) -> None:
        for name in ("source_tokens", "trace_tokens", "total_tokens"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.total_tokens < 1:
            raise ValueError("total_tokens must be positive")

    @classmethod
    def from_manifest(cls, manifest: FoldManifest) -> "RenderBudgets":
        return cls(
            source_tokens=int(manifest.source_token_budget),
            trace_tokens=int(manifest.trace_token_budget),
            total_tokens=int(manifest.total_evidence_token_budget),
        )


@dataclass(frozen=True)
class RenderedFold:
    """One rendered factorial cell plus auditable budget metadata."""

    variant: str
    text: str
    chunk_ids: tuple[str, ...]
    selected_chunk_ids: tuple[str, ...]
    source_fragments: tuple[tuple[str, str], ...]
    trace_lines: tuple[str, ...]
    source_tokens: int
    trace_tokens: int
    total_tokens: int
    trace_count: int
    order_changed: bool
    folding_fallback: bool
    folding_fallback_reason: str
    token_counter_name: str

    @property
    def source_text_by_chunk(self) -> dict[str, str]:
        return dict(self.source_fragments)


class _Codec:
    """Count and safely prefix-truncate text under one token definition."""

    def __init__(
        self,
        *,
        counter: Callable[[str], int],
        name: str,
        truncate: Callable[[str, int], str] | None = None,
    ) -> None:
        self._counter = counter
        self._truncate = truncate
        self.name = name

    def count(self, text: str) -> int:
        value = int(self._counter(str(text)))
        if value < 0:
            raise ValueError("token counter returned a negative value")
        return value

    def truncate(self, text: str, limit: int) -> str:
        text = str(text)
        limit = max(0, int(limit))
        if not text or limit == 0:
            return ""
        if self.count(text) <= limit:
            return text

        if self._truncate is not None:
            candidate = str(self._truncate(text, limit)).rstrip()
            # A tokenizer's decode options can occasionally introduce an
            # extra token.  The generic prefix pass below makes the bound hard.
            if self.count(candidate) <= limit:
                return candidate
            text = candidate

        low, high = 0, len(text)
        while low < high:
            midpoint = (low + high + 1) // 2
            if self.count(text[:midpoint].rstrip()) <= limit:
                low = midpoint
            else:
                high = midpoint - 1
        candidate = text[:low].rstrip()
        while candidate and self.count(candidate) > limit:
            candidate = candidate[:-1].rstrip()
        return candidate


_SIMPLE_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9_]+|[^\w\s]",
    flags=re.UNICODE,
)


def _simple_token_count(text: str) -> int:
    return len(_SIMPLE_TOKEN_RE.findall(str(text)))


def _tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def _tokenizer_decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))


def make_token_codec(
    *,
    token_counter: Callable[[str], int] | Any | None = None,
    tokenizer: Any | None = None,
) -> _Codec:
    """Create the token codec used by the hard-budget renderer.

    Callers may inject either a callable, an object exposing ``count`` (and
    optionally ``truncate``), or an already loaded Hugging Face tokenizer.
    Loading a tokenizer is deliberately left to the caller, so rendering never
    performs network I/O.
    """

    if token_counter is not None and tokenizer is not None:
        raise ValueError("pass token_counter or tokenizer, not both")
    if tokenizer is not None:
        return _Codec(
            counter=lambda text: len(_tokenizer_encode(tokenizer, text)),
            truncate=lambda text, limit: _tokenizer_decode(
                tokenizer, _tokenizer_encode(tokenizer, text)[:limit]
            ),
            name=f"hf:{type(tokenizer).__name__}",
        )
    if token_counter is None:
        return _Codec(counter=_simple_token_count, name="simple_unicode")
    if callable(token_counter):
        name = getattr(token_counter, "__name__", type(token_counter).__name__)
        return _Codec(counter=token_counter, name=str(name))
    count = getattr(token_counter, "count", None)
    if not callable(count):
        raise TypeError("token_counter must be callable or expose count(text)")
    truncate = getattr(token_counter, "truncate", None)
    return _Codec(
        counter=count,
        truncate=truncate if callable(truncate) else None,
        name=type(token_counter).__name__,
    )


def _clean_id(value: Any) -> str:
    return str(value).strip()


def _normalize_order(order: Sequence[Any], selected: Sequence[str]) -> tuple[str, ...]:
    selected_set = set(selected)
    normalized: list[str] = []
    for raw in order:
        chunk_id = _clean_id(raw)
        if chunk_id in selected_set and chunk_id not in normalized:
            normalized.append(chunk_id)
    for chunk_id in selected:
        if chunk_id not in normalized:
            normalized.append(chunk_id)
    return tuple(normalized)


def _extract_source_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "source_text", "content"):
            if key in value:
                return str(value[key]).strip()
    for name in ("text", "source_text", "content"):
        if hasattr(value, name):
            return str(getattr(value, name)).strip()
    return str(value).strip()


def _resolve_sources(
    manifest: FoldManifest,
    sources: Mapping[Any, Any],
) -> dict[str, str]:
    by_string_id = {_clean_id(key): value for key, value in sources.items()}
    resolved: dict[str, str] = {}
    for chunk_id in manifest.selected_chunk_ids:
        if chunk_id not in by_string_id:
            raise FoldingRenderError(
                f"manifest {manifest.query_id!r} selected chunk {chunk_id!r} "
                "but no source text was supplied"
            )
        resolved[chunk_id] = _extract_source_text(by_string_id[chunk_id])
    return resolved


def _source_block(chunk_id: str, fragment: str) -> str:
    header = f"[Source {chunk_id}]"
    return f"{header}\n{fragment}" if fragment else header


def _step_line(step: PathStep) -> str:
    source = " ".join(step.canonical_source_entity.split())
    relation = " ".join(step.relation.split())
    target = " ".join(step.canonical_target_entity.split())
    branch = " ".join(step.branch.split())
    if not source or not relation or not target or not branch or step.step < 1:
        return ""
    label = f"{branch}{step.step}" if branch.lower() not in {"main", "single"} else str(step.step)
    source_label = f"Source {step.supporting_chunk_id}"
    if step.traversal_direction == "reverse":
        traversal_source = " ".join(step.source_entity.split())
        traversal_target = " ".join(step.target_entity.split())
        return (
            f"[Step {label} | {source_label}] Fact: "
            f"{source} --{relation}--> {target}; "
            f"path traversal: {traversal_source} -> {traversal_target}"
        )
    return f"[Step {label} | {source_label}] {source} --{relation}--> {target}"


def _ordered_steps(
    manifest: FoldManifest,
    order: Sequence[str],
) -> tuple[PathStep, ...]:
    # Trace order is a property of the frozen path, never of source ordering.
    # This also handles a legitimate a,b,a supporting-chunk sequence without
    # reordering it to a,a,b.
    del order
    branch_positions: dict[str, int] = {}
    for step in manifest.path_steps:
        branch_positions.setdefault(step.branch, len(branch_positions))
    return tuple(
        sorted(
            manifest.path_steps,
            key=lambda step: (
                branch_positions[step.branch],
                step.step,
            ),
        )
    )


def _trace_parts(
    manifest: FoldManifest,
    order: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], bool]:
    """Return complete trace parts and their source alignment.

    The boolean is false when any selected path step lacks a real relation or
    a selected supporting source.  No placeholder trace is ever returned.
    """

    selected = set(manifest.selected_chunk_ids)
    if not manifest.path_steps:
        return (), {}, False
    for step in manifest.path_steps:
        if step.supporting_chunk_id not in selected or not _step_line(step):
            return (), {}, False

    grouped: dict[str, list[str]] = {chunk_id: [] for chunk_id in order}
    flat: list[str] = []
    active_branch: str | None = None
    for step in _ordered_steps(manifest, order):
        branch = step.branch
        if branch != active_branch and branch.lower() not in {"main", "single"}:
            heading = f"[Branch {branch}]"
            grouped[step.supporting_chunk_id].append(heading)
            flat.append(heading)
        line = _step_line(step)
        grouped[step.supporting_chunk_id].append(line)
        flat.append(line)
        active_branch = branch
    return (
        tuple(flat),
        {chunk_id: tuple(lines) for chunk_id, lines in grouped.items()},
        True,
    )


def _allocate_source_fragments(
    source_texts: Mapping[str, str],
    canonical_order: Sequence[str],
    source_budget: int,
    codec: _Codec,
) -> dict[str, str]:
    if not canonical_order:
        return {}
    empty_cost = sum(codec.count(_source_block(chunk_id, "")) for chunk_id in canonical_order)
    if empty_cost > source_budget:
        raise TokenBudgetError(
            "source token budget is smaller than the immutable source labels: "
            f"{empty_cost} > {source_budget}"
        )

    full_counts = {chunk_id: codec.count(source_texts[chunk_id]) for chunk_id in canonical_order}
    remaining = source_budget - empty_cost
    quotas = {chunk_id: 0 for chunk_id in canonical_order}

    # Fair deterministic allocation prevents the output order from changing
    # which source receives the truncation budget.
    active = [chunk_id for chunk_id in canonical_order if full_counts[chunk_id] > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for chunk_id in list(active):
            room = full_counts[chunk_id] - quotas[chunk_id]
            addition = min(room, share, remaining)
            if addition > 0:
                quotas[chunk_id] += addition
                remaining -= addition
                progressed = True
            if quotas[chunk_id] >= full_counts[chunk_id]:
                active.remove(chunk_id)
            if remaining == 0:
                break
        if not progressed:
            break

    fragments = {
        chunk_id: codec.truncate(source_texts[chunk_id], quotas[chunk_id])
        for chunk_id in canonical_order
    }
    return _shrink_to_source_budget(
        fragments, source_texts, canonical_order, source_budget, codec
    )


def _source_cost(
    fragments: Mapping[str, str],
    order: Sequence[str],
    codec: _Codec,
) -> int:
    # Sum per block so the measurement and allocation remain invariant to the
    # experimental ordering factor.
    return sum(codec.count(_source_block(chunk_id, fragments[chunk_id])) for chunk_id in order)


def _shrink_to_source_budget(
    fragments: dict[str, str],
    originals: Mapping[str, str],
    canonical_order: Sequence[str],
    budget: int,
    codec: _Codec,
) -> dict[str, str]:
    while _source_cost(fragments, canonical_order, codec) > budget:
        counts = {chunk_id: codec.count(fragments[chunk_id]) for chunk_id in canonical_order}
        reducible = [chunk_id for chunk_id in canonical_order if counts[chunk_id] > 0]
        if not reducible:
            raise TokenBudgetError("source labels exceed source token budget")
        chunk_id = max(
            reducible,
            key=lambda value: (counts[value], -canonical_order.index(value)),
        )
        fragments[chunk_id] = codec.truncate(originals[chunk_id], counts[chunk_id] - 1)
    return fragments


def _render_text(
    order: Sequence[str],
    fragments: Mapping[str, str],
    trace_lines: Sequence[str] | None,
) -> str:
    blocks: list[str] = []
    for chunk_id in order:
        blocks.append(_source_block(chunk_id, fragments[chunk_id]))
    if trace_lines:
        # Source text is the lossless evidence and must be read before the
        # lossy graph projection.  Putting a large trace first caused the
        # answer model to anchor on intermediate frontier entities.  The trace
        # therefore remains globally ordered, but is explicitly advisory and
        # follows all authoritative passages.
        blocks.append(
            "[Advisory Topology]\n"
            "Navigation hint only; source passages are authoritative.\n"
            + "\n".join(trace_lines)
        )
    return "\n\n".join(blocks)


def _render_raw_variants(
    *,
    score_order: Sequence[str],
    path_order: Sequence[str],
    fragments: Mapping[str, str],
    score_trace: Sequence[str],
    path_trace: Sequence[str],
    fallback: bool,
) -> dict[FoldingVariant, tuple[tuple[str, ...], str]]:
    if fallback:
        text = _render_text(score_order, fragments, None)
        return {
            variant: (tuple(score_order), text)
            for variant in FACTORIAL_VARIANTS
        }
    return {
        FoldingVariant.GRAPH_NAIVE: (
            tuple(score_order), _render_text(score_order, fragments, None)
        ),
        FoldingVariant.PATH_ORDER_SOURCE: (
            tuple(path_order), _render_text(path_order, fragments, None)
        ),
        FoldingVariant.TRACE_SCORE_SOURCE: (
            tuple(score_order), _render_text(score_order, fragments, score_trace)
        ),
        FoldingVariant.TOPOLOGY_FOLDING_V2: (
            tuple(path_order), _render_text(path_order, fragments, path_trace)
        ),
    }


def _fit_total_budget(
    *,
    fragments: dict[str, str],
    originals: Mapping[str, str],
    canonical_order: Sequence[str],
    total_budget: int,
    codec: _Codec,
    render: Callable[[Mapping[str, str]], dict[FoldingVariant, tuple[tuple[str, ...], str]]],
) -> tuple[dict[str, str], dict[FoldingVariant, tuple[tuple[str, ...], str]]] | None:
    outputs = render(fragments)
    while max((codec.count(text) for _, text in outputs.values()), default=0) > total_budget:
        counts = {chunk_id: codec.count(fragments[chunk_id]) for chunk_id in canonical_order}
        reducible = [chunk_id for chunk_id in canonical_order if counts[chunk_id] > 0]
        if not reducible:
            return None
        chunk_id = max(
            reducible,
            key=lambda value: (counts[value], -canonical_order.index(value)),
        )
        fragments[chunk_id] = codec.truncate(originals[chunk_id], counts[chunk_id] - 1)
        outputs = render(fragments)
    return fragments, outputs


def render_all_variants(
    manifest: FoldManifest,
    sources: Mapping[Any, Any],
    *,
    budgets: RenderBudgets | None = None,
    token_counter: Callable[[str], int] | Any | None = None,
    tokenizer: Any | None = None,
    validate_manifest: bool = True,
) -> dict[str, RenderedFold]:
    """Render all four factorial cells from one fixed manifest.

    The returned mapping uses string variant names to simplify CSV/JSON
    integration.  Rendering never inspects question answers or gold evidence.
    """

    if not isinstance(manifest, FoldManifest):
        manifest = FoldManifest.from_dict(manifest)
    if validate_manifest:
        manifest.validate()
    budgets = budgets or RenderBudgets.from_manifest(manifest)
    codec = make_token_codec(token_counter=token_counter, tokenizer=tokenizer)

    selected = tuple(manifest.selected_chunk_ids)
    if not selected:
        return {
            variant.value: RenderedFold(
                variant=variant.value,
                text="",
                chunk_ids=(),
                selected_chunk_ids=(),
                source_fragments=(),
                trace_lines=(),
                source_tokens=0,
                trace_tokens=0,
                total_tokens=0,
                trace_count=0,
                order_changed=False,
                folding_fallback=bool(manifest.fallback_to_graph_naive),
                folding_fallback_reason=manifest.fallback_reason,
                token_counter_name=codec.name,
            )
            for variant in FACTORIAL_VARIANTS
        }

    score_order = _normalize_order(manifest.score_order_chunk_ids, selected)
    path_order = _normalize_order(manifest.path_order_chunk_ids, selected)
    source_texts = _resolve_sources(manifest, sources)
    # score order is the canonical allocation order; it is fixed by the
    # manifest and never by the requested renderer.
    fragments = _allocate_source_fragments(
        source_texts, score_order, budgets.source_tokens, codec
    )

    explicit_fallback = (
        not manifest.foldable
        or manifest.fallback_to_graph_naive
        or not manifest.path_continuous
        or not manifest.branch_complete
    )
    fallback = explicit_fallback
    fallback_reason = manifest.fallback_reason if explicit_fallback else ""

    score_lines: tuple[str, ...] = ()
    path_lines: tuple[str, ...] = ()
    score_trace: dict[str, tuple[str, ...]] = {}
    path_trace: dict[str, tuple[str, ...]] = {}
    if not fallback:
        score_lines, score_trace, score_valid = _trace_parts(manifest, score_order)
        path_lines, path_trace, path_valid = _trace_parts(manifest, path_order)
        if not score_valid or not path_valid:
            fallback = True
            fallback_reason = "invalid_or_unaligned_trace"
        else:
            score_trace_tokens = codec.count("\n".join(score_lines))
            path_trace_tokens = codec.count("\n".join(path_lines))
            if max(score_trace_tokens, path_trace_tokens) > budgets.trace_tokens:
                fallback = True
                fallback_reason = "trace_token_budget_exceeded"

    def render_with(current: Mapping[str, str]):
        return _render_raw_variants(
            score_order=score_order,
            path_order=path_order,
            fragments=current,
            score_trace=score_lines,
            path_trace=path_lines,
            fallback=fallback,
        )

    fitted = _fit_total_budget(
        fragments=dict(fragments),
        originals=source_texts,
        canonical_order=score_order,
        total_budget=budgets.total_tokens,
        codec=codec,
        render=render_with,
    )
    if fitted is None and not fallback:
        # A complete trace is atomic.  If it cannot coexist with the immutable
        # labels under the total budget, drop all trace/order factors and use
        # the documented Graph-Naive fallback instead of emitting a partial or
        # fabricated path.
        fallback = True
        fallback_reason = "total_token_budget_exceeded"
        score_lines = path_lines = ()
        score_trace = path_trace = {}
        fragments = _allocate_source_fragments(
            source_texts, score_order, budgets.source_tokens, codec
        )
        fitted = _fit_total_budget(
            fragments=dict(fragments),
            originals=source_texts,
            canonical_order=score_order,
            total_budget=budgets.total_tokens,
            codec=codec,
            render=render_with,
        )
    if fitted is None:
        raise TokenBudgetError(
            "total token budget is smaller than the immutable source labels"
        )

    fragments, raw_outputs = fitted
    shared_fragments = tuple((chunk_id, fragments[chunk_id]) for chunk_id in score_order)
    source_tokens = _source_cost(fragments, score_order, codec)
    output: dict[str, RenderedFold] = {}
    for variant in FACTORIAL_VARIANTS:
        order, text = raw_outputs[variant]
        uses_trace = (
            not fallback
            and variant in {
                FoldingVariant.TRACE_SCORE_SOURCE,
                FoldingVariant.TOPOLOGY_FOLDING_V2,
            }
        )
        lines = (
            score_lines
            if variant is FoldingVariant.TRACE_SCORE_SOURCE
            else path_lines
            if variant is FoldingVariant.TOPOLOGY_FOLDING_V2
            else ()
        ) if uses_trace else ()
        trace_tokens = codec.count("\n".join(lines)) if lines else 0
        total_tokens = codec.count(text)
        if source_tokens > budgets.source_tokens:
            raise AssertionError("renderer exceeded source token budget")
        if trace_tokens > budgets.trace_tokens:
            raise AssertionError("renderer exceeded trace token budget")
        if total_tokens > budgets.total_tokens:
            raise AssertionError("renderer exceeded total token budget")
        output[variant.value] = RenderedFold(
            variant=variant.value,
            text=text,
            chunk_ids=tuple(order),
            selected_chunk_ids=selected,
            source_fragments=shared_fragments,
            trace_lines=tuple(lines),
            source_tokens=source_tokens,
            trace_tokens=trace_tokens,
            total_tokens=total_tokens,
            trace_count=len(manifest.path_steps) if uses_trace else 0,
            order_changed=tuple(order) != score_order,
            folding_fallback=fallback,
            folding_fallback_reason=fallback_reason,
            token_counter_name=codec.name,
        )
    return output


def render_manifest(
    manifest: FoldManifest,
    sources: Mapping[Any, Any],
    variant: FoldingVariant | str,
    **kwargs: Any,
) -> RenderedFold:
    """Render one cell while retaining the shared-truncation guarantees."""

    variant = FoldingVariant(variant)
    render_variant = (
        FoldingVariant.PATH_ORDER_SOURCE
        if variant is FoldingVariant.TOPOLOGY_FOLDING_LOSSLESS
        else variant
    )
    rendered = render_all_variants(
        manifest, sources, **kwargs,
    )[render_variant.value]
    if variant is FoldingVariant.TOPOLOGY_FOLDING_LOSSLESS:
        return replace(rendered, variant=variant.value)
    return rendered


# Explicit alias used by CLI/replay code.
render_manifest_variants = render_all_variants


__all__ = [
    "FACTORIAL_VARIANTS",
    "FOLDING_RENDERER_VERSION",
    "FoldingRenderError",
    "FoldingVariant",
    "RenderBudgets",
    "RenderedFold",
    "TokenBudgetError",
    "make_token_codec",
    "render_all_variants",
    "render_manifest",
    "render_manifest_variants",
]
