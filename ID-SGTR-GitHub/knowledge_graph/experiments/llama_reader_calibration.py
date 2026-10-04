"""Frozen-context Llama reader calibration without gold-evidence access."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests


SYSTEM = """You are a precise multi-hop question-answering system. Use only the supplied evidence. Resolve every relation in the question, distinguish entities with similar names, and preserve exact dates, numbers, names, and yes/no polarity."""


def _ids(value: object) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return list(dict.fromkeys(str(item) for item in parsed))


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "on"}


def _topology_facts(value: object, limit: int = 24) -> list[str]:
    try:
        candidates = ast.literal_eval(str(value))
    except Exception:
        return []
    facts: list[str] = []
    seen: set[str] = set()
    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        values = [candidate.get("triple")]
        values.extend(
            step.get("triple")
            for step in candidate.get("topology_trace", [])
            if isinstance(step, dict)
        )
        for fact in values:
            fact = str(fact or "").strip()
            if fact and fact not in seen:
                seen.add(fact)
                facts.append(fact)
                if len(facts) >= limit:
                    return facts
    return facts


def _clean_answer(text: str) -> str:
    text = str(text or "").strip()
    match = re.findall(r"Final Answer\s*:\s*(.+)", text, flags=re.I)
    if match:
        text = match[-1]
    text = text.splitlines()[0].strip().strip("`\"'")
    text = re.sub(r"^\*+|\*+$", "", text).strip()
    text = re.sub(r"^(?:the answer is|answer)\s*[:\-]?\s*", "", text, flags=re.I)
    yes_no = re.match(r"^(yes|no)\b", text, flags=re.I)
    return yes_no.group(1).lower() if yes_no else text


class Client:
    def __init__(self, endpoint: str, model: str, timeout: float):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.local = threading.local()

    def invoke(self, prompt: str, max_tokens: int) -> tuple[str, int, int, float]:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            self.local.session = session
        started = time.perf_counter()
        response = session.post(
            self.endpoint,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "seed": 42,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        return (
            payload["choices"][0]["message"].get("content") or "",
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            time.perf_counter() - started,
        )


def _evidence(row: dict, chunks_by_context: dict[str, list[dict]], chunks_by_id: dict[str, dict], variant: str) -> list[dict]:
    priority_ids = _ids(row.get("accessed_evidence")) or _ids(row.get("retrieved_evidence"))
    priority = [chunks_by_id[item] for item in priority_ids if item in chunks_by_id]
    if variant == "path5_verify":
        return priority[:5]
    context = chunks_by_context.get(str(row.get("context_id")), [])
    if variant == "full_context_direct":
        return context
    seen = {str(item["chunk_id"]) for item in priority}
    return priority + [item for item in context if str(item["chunk_id"]) not in seen]


def _render(items: list[dict], priority_count: int) -> str:
    lines = []
    for position, item in enumerate(items, start=1):
        label = "PATH-PRIORITY" if position <= priority_count else "LOCAL-CONTEXT"
        text = str(item.get("text") or "").replace("\n", " ").strip()
        title = str(item.get("title") or "").strip()
        lines.append(f"[{label} Ref {item['chunk_id']}] {title}: {text}")
    return "\n".join(lines)


def _solve(row: dict, client: Client, chunks_by_context: dict[str, list[dict]], chunks_by_id: dict[str, dict], variant: str) -> dict:
    skip_reader = (
        variant == "gated_trace_judge" and not _truth(row.get("stage0_early_exit"))
    ) or (
        variant == "terminal_trace_judge" and _truth(row.get("stage0_early_exit"))
    )
    if skip_reader:
        result = dict(row)
        result.update(
            reader_variant=variant,
            reader_calls=0,
            reader_input_tokens=0,
            reader_output_tokens=0,
            reader_time_s=0.0,
            reader_evidence_count=len(_ids(row.get("accessed_evidence"))),
            reader_gate_applied=False,
        )
        return result
    items = _evidence(row, chunks_by_context, chunks_by_id, variant)
    priority_count = min(len(_ids(row.get("accessed_evidence"))), len(items))
    evidence = _render(items, priority_count)
    question = str(row["question"])
    analyst_prompt = f"""Question: {question}

Evidence:
{evidence}

Work through the exact relation chain needed by the question. For a comparison, derive both branches before comparing. Identify distractors and the requested answer type. End with exactly:
Final Answer: <shortest exact answer span>"""
    draft, in1, out1, time1 = client.invoke(analyst_prompt, max_tokens=512)
    total_in, total_out, elapsed, calls = in1, out1, time1, 1
    final_text = draft
    if variant in {"triangulated_judge", "gated_trace_judge", "terminal_trace_judge"}:
        path_items = _evidence(row, chunks_by_context, chunks_by_id, "path5_verify")
        path_evidence = _render(path_items, len(path_items))
        path_prompt = f"""Question: {question}

Topology-selected evidence:
{path_evidence}

Derive the answer by following the exact relation chain. State the requested answer type, resolve both sides of comparisons, and end with:
Final Answer: <shortest exact answer span>"""
        path_draft, in2, out2, time2 = client.invoke(path_prompt, max_tokens=512)
        facts = _topology_facts(row.get("candidate_evidence"))
        facts_text = "\n".join(f"- {fact}" for fact in facts) or "- No reliable graph trace was recorded."
        judge_prompt = f"""Act as a conservative multi-hop adjudicator. The graph agent answered too early. Reconstruct the requested relation chain from the topology facts and source evidence, then correct Candidate A only when the reconstructed chain supports a different answer.

Question: {question}

Executed topology facts:
{facts_text}

Evidence:
{evidence}

Candidate A (existing graph agent): {row.get('pred_answer', '')}
Candidate B (full-context analyst): {draft}
Candidate C (topology-path analyst): {path_draft}

Rules:
1. Write down the requested answer type internally. Do not return an intermediate person when the question asks for that person's distinction, work, location, date, or attribute.
2. For yes/no, verify every named subject separately; answer yes only if every required proposition is true.
3. For comparisons, compute both branches explicitly before choosing.
4. Prefer the complete canonical name/title present in evidence. Preserve geographic qualifiers when they disambiguate the answer.
5. For a "what year" question return only the year. For "when" return a year unless the question explicitly requests an exact date.
6. Do not say that information is missing when a relation chain in the evidence supplies it.
7. Candidate A is the current system answer. Preserve it if the evidence does not clearly establish a correction; never switch merely because another candidate is worded more confidently.

Return exactly one line:
Final Answer: <shortest exact answer span>"""
        final_text, in3, out3, time3 = client.invoke(judge_prompt, max_tokens=128)
        total_in += in2 + in3
        total_out += out2 + out3
        elapsed += time2 + time3
        calls += 2
    elif variant.endswith("_verify"):
        verify_prompt = f"""Independently verify a proposed answer using the evidence. Correct it if it follows the wrong entity, wrong relation, wrong comparison branch, wrong date, or wrong answer type.

Question: {question}

Evidence:
{evidence}

First-pass draft:
{draft}

Return exactly one line and nothing else:
Final Answer: <shortest exact answer span>"""
        final_text, in2, out2, time2 = client.invoke(verify_prompt, max_tokens=128)
        total_in += in2
        total_out += out2
        elapsed += time2
        calls += 1
    answer = _clean_answer(final_text)
    result = dict(row)
    result.update(
        pred_answer=answer,
        reader_variant=variant,
        reader_calls=calls,
        reader_input_tokens=total_in,
        reader_output_tokens=total_out,
        reader_time_s=elapsed,
        reader_evidence_count=len(items),
        reader_gate_applied=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("path5_verify", "full_context_direct", "pathguided_full_verify", "triangulated_judge", "gated_trace_judge", "terminal_trace_judge"), required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="llama-3.1-8b-instruct")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, sep="|")
    if args.limit:
        frame = frame.iloc[: args.limit].copy()
    chunks = pd.read_csv(args.chunks, sep="|")
    chunks["chunk_id"] = chunks["chunk_id"].astype(str)
    chunks["context_id"] = chunks["context_id"].astype(str)
    records = chunks.to_dict(orient="records")
    by_id = {str(item["chunk_id"]): item for item in records}
    by_context: dict[str, list[dict]] = {}
    for item in records:
        by_context.setdefault(str(item["context_id"]), []).append(item)

    client = Client(args.endpoint, args.model, args.timeout)
    rows = frame.to_dict(orient="records")
    completed: dict[int, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_solve, row, client, by_context, by_id, args.variant): index
            for index, row in enumerate(rows)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            completed[index] = future.result()
            if len(completed) % 20 == 0:
                print(f"completed {len(completed)}/{len(rows)}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([completed[index] for index in range(len(rows))]).to_csv(output, sep="|", index=False)
    print(f"wrote {len(rows)} reader predictions to {output}")


if __name__ == "__main__":
    main()
