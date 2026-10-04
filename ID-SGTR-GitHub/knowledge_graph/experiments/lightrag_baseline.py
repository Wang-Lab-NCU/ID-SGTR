"""Official LightRAG baseline on the frozen ID-SGTR corpora.

Index construction and online querying are deliberately separate.  The former
is an offline corpus operation and is never included in per-query latency.  The
online adapter uses LightRAG's structured ``aquery_llm`` API so retrieval,
evidence IDs, and answer generation are produced by one pipeline execution.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any

import aiohttp
import numpy as np
import pandas as pd


CHUNK_MARKER = re.compile(r"\[CHUNK_ID:\s*([^\]]+)\]")
FINAL_RE = re.compile(r"(?:final\s+answer|answer)\s*:\s*(.+)", re.I)
UPSTREAM_COMMIT = "20ac1b72ace481e2fd1c4697e7baa5b8bbc77b23"
STRICT_ANSWER_PROMPT = """\
Answer the question using only the retrieved evidence.
Output exactly one line in this format:
FINAL_ANSWER: <short answer>

The answer must be only the shortest entity, date, number, yes, or no that
answers the question. Use at most 12 words. Do not explain, justify, cite
sources, repeat the question, or add any other text. If the evidence is
insufficient, output exactly: FINAL_ANSWER: unknown
"""


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|", dtype=str).fillna("")


def _ids(value: object) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    return [part.strip() for part in text.split(",") if part.strip()]


def _clean_answer(value: object) -> str:
    text = str(value or "").strip().strip("`").strip()
    match = FINAL_RE.search(text)
    if match:
        text = match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    return text.strip().strip("`").strip('"').strip("'").rstrip(".").strip()


@dataclass
class CallStats:
    calls: int = 0
    elapsed_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    kinds: list[str] = field(default_factory=list)


ACTIVE_STATS: ContextVar[CallStats | None] = ContextVar(
    "lightrag_active_stats", default=None
)


class OpenAICompatibleServices:
    def __init__(
        self,
        *,
        chat_base_url: str,
        chat_model: str,
        embedding_base_url: str,
        embedding_model: str,
        max_tokens: int,
        seed: int,
        timeout_s: float,
    ) -> None:
        self.chat_url = chat_base_url.rstrip("/") + "/chat/completions"
        self.chat_model = chat_model
        self.embedding_url = embedding_base_url.rstrip("/") + "/embeddings"
        self.embedding_model = embedding_model
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.timeout = aiohttp.ClientTimeout(total=float(timeout_s))

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        error: Exception | None = None
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.post(
                        url,
                        headers={"Authorization": "Bearer EMPTY"},
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                error = exc
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"request failed after three attempts: {error}")

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        keyword_extraction: bool = False,
        **kwargs: Any,
    ) -> str:
        # LightRAG does not consistently forward ``keyword_extraction=True``
        # to custom providers, so also recognize its official keyword prompt.
        is_keyword = keyword_extraction or (
            "high_level_keywords" in str(prompt)
            and "low_level_keywords" in str(prompt)
        )
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        for message in history_messages or []:
            if message.get("role") and message.get("content") is not None:
                messages.append(
                    {"role": str(message["role"]), "content": str(message["content"])}
                )
        messages.append({"role": "user", "content": str(prompt)})
        # A final assistant prefill is substantially more reliable than a
        # natural-language brevity instruction for Qwen3.  vLLM returns only
        # the continuation, which is already the answer span consumed by the
        # deterministic parser.  Keyword JSON generation must remain
        # unaffected.
        if not is_keyword:
            messages.append({"role": "assistant", "content": "FINAL_ANSWER:"})
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if not is_keyword:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        started = time.perf_counter()
        data = await self._post(self.chat_url, payload)
        elapsed = time.perf_counter() - started
        usage = data.get("usage") or {}
        message = data["choices"][0]["message"]
        content = message.get("content") or message.get("reasoning_content") or ""
        stats = ACTIVE_STATS.get()
        if stats is not None:
            stats.calls += 1
            stats.elapsed_s += elapsed
            stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            stats.completion_tokens += int(usage.get("completion_tokens") or 0)
            stats.kinds.append("keyword" if is_keyword else "answer")
        return str(content).strip()

    async def embed(self, texts: list[str]) -> np.ndarray:
        data = await self._post(
            self.embedding_url,
            {"model": self.embedding_model, "input": list(texts)},
        )
        ordered = sorted(data.get("data", []), key=lambda item: int(item["index"]))
        if len(ordered) != len(texts):
            raise RuntimeError(
                f"embedding response count mismatch: {len(ordered)} != {len(texts)}"
            )
        return np.asarray([item["embedding"] for item in ordered], dtype=np.float32)


def _document(row: Any) -> str:
    return (
        f"[CHUNK_ID: {row.chunk_id}] [CONTEXT_ID: {row.context_id}] "
        f"Wikipedia Title: {row.title}\n{row.text}"
    )


def _retrieved_ids(result: dict[str, Any]) -> list[str]:
    data = result.get("data") or {}
    found: list[str] = []
    seen: set[str] = set()
    for chunk in data.get("chunks", []) or []:
        match = CHUNK_MARKER.search(str(chunk.get("content", "")))
        if match and match.group(1) not in seen:
            found.append(match.group(1))
            seen.add(match.group(1))
    return found


async def _make_rag(args: argparse.Namespace):
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import EmbeddingFunc

    services = OpenAICompatibleServices(
        chat_base_url=args.chat_base_url,
        chat_model=args.chat_model,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        max_tokens=args.max_tokens,
        seed=args.seed,
        timeout_s=args.timeout,
    )
    rag = LightRAG(
        working_dir=str(args.working_dir),
        llm_model_func=services.complete,
        llm_model_name=args.chat_model,
        llm_model_max_async=args.max_async,
        max_parallel_insert=args.max_async,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=services.embed,
        ),
        embedding_func_max_async=args.max_async,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=True,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def _close_rag(rag: Any) -> None:
    finalize = getattr(rag, "finalize_storages", None)
    if finalize is not None:
        await finalize()


async def command_index(args: argparse.Namespace) -> None:
    chunks = _read(args.chunks)
    if args.limit is not None:
        chunks = chunks.head(args.limit)
    args.working_dir.mkdir(parents=True, exist_ok=True)
    rag = await _make_rag(args)
    started = time.perf_counter()
    try:
        documents = [_document(row) for row in chunks.itertuples(index=False)]
        document_ids = [f"source-chunk-{value}" for value in chunks["chunk_id"]]
        await rag.ainsert(documents, ids=document_ids)
    finally:
        await _close_rag(rag)
    elapsed = time.perf_counter() - started
    metadata = {
        "dataset": args.dataset,
        "documents": len(chunks),
        "elapsed_s": elapsed,
        "upstream_commit": UPSTREAM_COMMIT,
        "chat_model": args.chat_model,
        "embedding_model": args.embedding_model,
        "offline_max_async": args.max_async,
    }
    output = args.working_dir / "id_sgtr_index_metadata.json"
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"LIGHTRAG_INDEX_COMPLETE {json.dumps(metadata)}", flush=True)


def _save(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, sep="|", index=False)
    temporary.replace(output)


async def command_query(args: argparse.Namespace) -> None:
    from lightrag import QueryParam

    subset = _read(args.subset)
    if args.limit is not None:
        subset = subset.head(args.limit)
    rag = await _make_rag(args)
    results: list[dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume and args.output.exists():
        previous = _read(args.output)
        # A failed internal LLM call can still leave a syntactically valid row
        # in the checkpoint.  Resume only rows with a non-empty prediction and
        # no recorded error; failed rows must be retried instead of silently
        # becoming part of the formal score.
        error = previous.get(
            "error", pd.Series("", index=previous.index, dtype=str)
        ).fillna("").astype(str).str.strip()
        prediction = previous["pred_answer"].fillna("").astype(str).str.strip()
        valid = error.eq("") & prediction.ne("")
        dropped = int((~valid).sum())
        previous = previous.loc[valid].copy()
        results = previous.to_dict(orient="records")
        completed = set(previous["query_id"].astype(str))
        if dropped:
            print(f"resume will retry {dropped} failed/empty rows", flush=True)
    pending = [
        row
        for row in subset.to_dict(orient="records")
        if str(row["query_id"]) not in completed
    ]
    param = QueryParam(
        mode=args.mode,
        response_type="Short Exact Answer",
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        enable_rerank=False,
        user_prompt=STRICT_ANSWER_PROMPT,
    )
    print(
        f"LightRAG dataset={args.dataset} queries={len(subset)} pending={len(pending)} "
        f"mode={args.mode} online_concurrency={args.max_workers} cache=disabled",
        flush=True,
    )

    semaphore = asyncio.Semaphore(max(1, int(args.max_workers)))

    async def process_row(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            stats = CallStats()
            token = ACTIVE_STATS.set(stats)
            started = time.perf_counter()
            error = ""
            try:
                result = await rag.aquery_llm(str(row["question"]), param=param)
                response = (result.get("llm_response") or {}).get("content") or ""
                prediction = _clean_answer(response)
                retrieved = _retrieved_ids(result)
            except Exception as exc:
                response = ""
                prediction = ""
                retrieved = []
                error = f"{type(exc).__name__}: {exc}"
            finally:
                ACTIVE_STATS.reset(token)
            total_time = time.perf_counter() - started
            answer_calls = sum(kind == "answer" for kind in stats.kinds)
            auxiliary_calls = stats.calls - answer_calls
            return {
                "query_id": str(row["query_id"]),
                "question": str(row["question"]),
                "gold_answer": str(row.get("answer", "")),
                "pred_answer": prediction,
                "context_id": str(row.get("context_id", "")),
                "gold_evidence": str(_ids(row.get("gold_evidence", ""))),
                "retrieved_evidence": str(retrieved),
                "candidate_evidence": str(retrieved),
                "strategy": f"LightRAG-{args.mode}",
                "raw_answer": response,
                "error": error,
                "total_llm_calls": stats.calls,
                "answer_calls": answer_calls,
                "auxiliary_calls": auxiliary_calls,
                "retrieval_rounds": 1,
                "retrieval_time_s": max(0.0, total_time - stats.elapsed_s),
                "generation_time_s": stats.elapsed_s,
                "total_time_s": total_time,
                "input_tokens": stats.prompt_tokens,
                "output_tokens": stats.completion_tokens,
                "fallback": bool(error),
                "model": args.chat_model,
                "run_mode": "retrieval",
                "evidence_variant": f"lightrag_official_{args.mode}",
                "temperature": 0,
                "max_output_tokens": args.max_tokens,
                "answer_protocol": "strict_final_answer_prefill_v1",
                "upstream_commit": UPSTREAM_COMMIT,
            }

    try:
        tasks = [asyncio.create_task(process_row(row)) for row in pending]
        for position, task in enumerate(asyncio.as_completed(tasks), start=1):
            record = await task
            results.append(record)
            _save(results, args.output)
            print(
                f"[{position}/{len(pending)}] {record['query_id']} "
                f"calls={record['total_llm_calls']} "
                f"evidence={len(_ids(record['retrieved_evidence']))} "
                f"answer={record['pred_answer']!r}",
                flush=True,
            )
    finally:
        await _close_rag(rag)
    print(f"wrote {len(results)} LightRAG predictions to {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "query"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--dataset", required=True)
        subparser.add_argument("--working-dir", type=Path, required=True)
        subparser.add_argument("--chat-base-url", default="http://127.0.0.1:8001/v1")
        subparser.add_argument("--chat-model", default="qwen3-8b")
        subparser.add_argument(
            "--embedding-base-url", default="http://127.0.0.1:30000/v1"
        )
        subparser.add_argument("--embedding-model", default="./bge-m3")
        subparser.add_argument("--max-tokens", type=int, default=512)
        subparser.add_argument("--max-async", type=int, default=16)
        subparser.add_argument("--max-workers", type=int, default=1)
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--timeout", type=float, default=300.0)
        subparser.add_argument("--limit", type=int)
    index_parser = subparsers.choices["index"]
    index_parser.add_argument("--chunks", type=Path, required=True)
    query_parser = subparsers.choices["query"]
    query_parser.add_argument("--subset", type=Path, required=True)
    query_parser.add_argument("--output", type=Path, required=True)
    query_parser.add_argument("--mode", choices=["local", "global", "hybrid", "mix", "naive"], default="hybrid")
    query_parser.add_argument("--top-k", type=int, default=10)
    query_parser.add_argument("--chunk-top-k", type=int, default=10)
    query_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "index":
        asyncio.run(command_index(args))
    else:
        asyncio.run(command_query(args))


if __name__ == "__main__":
    main()
