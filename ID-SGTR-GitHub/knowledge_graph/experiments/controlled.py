"""Fixed-candidate replay for the controlled Topology Folding experiment."""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping

import pandas as pd

from .evidence import EvidenceAssembler, EvidenceItem, EvidenceVariant
from .telemetry import QueryTelemetry, TrackedChatModel


def parse_sequence(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def response_draft(response: Any) -> str:
    """Collect visible content and provider reasoning from an AI message."""
    content = str(getattr(response, "content", "") or "").strip()
    extra = getattr(response, "additional_kwargs", {}) or {}
    metadata = getattr(response, "response_metadata", {}) or {}
    reasoning = str(
        extra.get("reasoning") or extra.get("reasoning_content")
        or metadata.get("reasoning") or metadata.get("reasoning_content")
        or getattr(response, "reasoning", "") or ""
    ).strip()
    return "\n".join(part for part in (reasoning, content) if part).strip()


class ControlledAblationRunner:
    """Answer from a frozen candidate manifest without rerunning graph retrieval."""

    def __init__(self, model: Any, chunks: pd.DataFrame, *, formatter_model: Any | None = None, budget: int = 3, seed: int = 42):
        required = {"chunk_id", "text"}
        if missing := required.difference(chunks.columns):
            raise ValueError(f"chunk table is missing columns: {sorted(missing)}")
        chunk_copy = chunks.copy()
        chunk_copy["chunk_id"] = chunk_copy["chunk_id"].astype(str)
        self.text_by_id = chunk_copy.drop_duplicates("chunk_id").set_index("chunk_id")["text"].astype(str).to_dict()
        self.model = model
        self.formatter_model = formatter_model
        self.assembler = EvidenceAssembler(budget=budget, random_seed=seed)

    def _items(self, row: Mapping[str, object], variant: EvidenceVariant) -> list[EvidenceItem]:
        if variant is EvidenceVariant.ORACLE:
            return [
                EvidenceItem(str(chunk_id), self.text_by_id.get(str(chunk_id), ""), path_position=position, is_gold=True)
                for position, chunk_id in enumerate(parse_sequence(row.get("gold_evidence")), start=1)
            ]
        candidates = [candidate for candidate in parse_sequence(row.get("candidate_evidence")) if isinstance(candidate, dict)]
        has_graph_hops = any(
            int(candidate.get("hop", 0) or 0) > 0
            or int(candidate.get("path_position", 0) or 0) >= 1000
            for candidate in candidates
        )
        items: list[EvidenceItem] = []
        for position, candidate in enumerate(candidates, start=1):
            raw_position = int(candidate.get("path_position", position) or position)
            hop = int(candidate.get("hop", 0) or 0)
            if hop <= 0 and raw_position >= 1000:
                hop, raw_position = divmod(raw_position, 1000)
            if has_graph_hops and hop <= 0:
                continue
            chunk_id = str(candidate.get("chunk_id", ""))
            if not chunk_id or chunk_id not in self.text_by_id:
                continue
            topology_trace = []
            for trace in candidate.get("topology_trace", []) or []:
                if not isinstance(trace, dict):
                    continue
                trace_triple = str(trace.get("triple", ""))
                if not trace_triple.strip():
                    continue
                topology_trace.append((
                    int(trace.get("hop", hop) or hop),
                    int(trace.get("path_position", raw_position) or raw_position),
                    trace_triple,
                ))
            items.append(EvidenceItem(
                chunk_id=chunk_id,
                text=self.text_by_id[chunk_id],
                score=float(candidate.get("score", 0.0)),
                path_position=raw_position,
                triple=str(candidate.get("triple", "")),
                hop=hop,
                topology_trace=tuple(topology_trace),
            ))
        return items

    def run_row(self, row: Mapping[str, object], variant: EvidenceVariant | str) -> dict[str, object]:
        variant = EvidenceVariant(variant)
        query_id = str(row.get("query_id", ""))
        telemetry = QueryTelemetry(query_id=query_id)
        telemetry.start()
        items = self._items(row, variant)
        references, selected_ids = self.assembler.render(items, variant, query_id=query_id)
        evidence = chr(10).join(references) if references else 'No evidence available.'
        reasoning_prompt = f"""Solve the multi-hop question using ONLY the fixed evidence below.

Follow every relation required by the question. For comparisons, resolve the
requested attribute for both alternatives. Think carefully and end with a
short Final Answer.

Question: {row.get('question', '')}

Evidence:
{evidence}

End with: Final Answer: <shortest answer span>"""
        reasoning_response = TrackedChatModel(self.model, telemetry, "answer").invoke(reasoning_prompt)
        response = reasoning_response
        if self.formatter_model is not None:
            draft = response_draft(reasoning_response)
            formatter_prompt = f"""Normalize the answer using the question, evidence, and reasoning draft.

Question: {row.get('question', '')}

Evidence:
{evidence}

Reasoning draft:
{draft or 'No reasoning draft was returned.'}

Final Answer: <shortest answer span>

Return exactly one line. The span must be only the entity, place, date,
number, yes, or no. Do not return reasoning, a sentence, an intermediate
entity, or a different attribute."""
            response = TrackedChatModel(self.formatter_model, telemetry, "answer").invoke(formatter_prompt)
        telemetry.retrieved_evidence = selected_ids
        telemetry.stop()
        raw_answer = str(getattr(response, "content", response)).strip()
        answer = raw_answer.split("Final Answer:")[-1].strip() if "Final Answer:" in raw_answer else raw_answer
        answer = answer.splitlines()[0].strip() if answer else ""
        binary = re.match(r"^(yes|no)\b", answer, flags=re.IGNORECASE)
        if binary:
            answer = binary.group(1).lower()
        result = dict(row)
        result.update({
            "pred_answer": answer,
            "evidence_variant": variant.value,
            "fixed_candidate_count": len({item.chunk_id for item in items}),
            "selected_evidence_set": sorted(set(selected_ids)),
            **telemetry.to_dict(),
        })
        return result
