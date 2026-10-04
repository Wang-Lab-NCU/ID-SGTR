"""HippoRAG 2 baseline on the frozen ID-SGTR corpora.

The upstream implementation is loaded from ``HIPPORAG_ROOT``.  Offline graph
construction and online query evaluation are separate so indexing cost is not
included in per-query latency.
"""

from __future__ import annotations

import argparse
import ast
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import pandas as pd
import requests


UPSTREAM_SNAPSHOT = "OSU-NLP-Group/HippoRAG-main-2026-07-29"


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
    if "Answer:" in text:
        text = text.rsplit("Answer:", 1)[-1].strip()
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


ACTIVE_STATS: ContextVar[CallStats | None] = ContextVar("hipporag_stats", default=None)


class OpenAICompatibleLLM:
    """Small provider shim matching HippoRAG's ``BaseLLM.infer`` contract."""

    def __init__(self, base_url: str, model: str, max_tokens: int, seed: int):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.session = requests.Session()

    @staticmethod
    def _kind(messages: list[dict[str, Any]]) -> str:
        text = "\n".join(str(item.get("content", "")) for item in messages)
        if "fact_after_filter" in text:
            return "auxiliary"
        if "named_entities" in text or '"triples"' in text:
            return "offline_index"
        return "answer"

    def infer(self, messages: list[dict[str, Any]], **kwargs: Any):
        kind = self._kind(messages)
        max_tokens = int(
            kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens")
            or self.max_tokens
        )
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if kind == "offline_index":
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.post(
                    self.url,
                    headers={"Authorization": "Bearer EMPTY"},
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                error = exc
                if attempt == 2:
                    raise RuntimeError(f"HippoRAG LLM request failed: {error}") from exc
                time.sleep(2**attempt)
        elapsed = time.perf_counter() - started
        usage = data.get("usage") or {}
        content = (data["choices"][0]["message"].get("content") or "").strip()
        metadata = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "finish_reason": data["choices"][0].get("finish_reason", "stop"),
        }
        stats = ACTIVE_STATS.get()
        if stats is not None:
            stats.calls += 1
            stats.elapsed_s += elapsed
            stats.prompt_tokens += metadata["prompt_tokens"]
            stats.completion_tokens += metadata["completion_tokens"]
            stats.kinds.append(kind)
        return content, metadata, False


def _load_upstream(args: argparse.Namespace):
    root = args.hipporag_root.resolve()
    support = args.hipporag_python.resolve()
    for value in (str(support), str(root / "src")):
        if value not in sys.path:
            sys.path.insert(0, value)
    from hipporag import HippoRAG
    from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
    from hipporag.utils.config_utils import BaseConfig
    from hipporag.utils.misc_utils import Chunk

    return HippoRAG, OpenAIEmbeddingModel, BaseConfig, Chunk


def _dataset_name(dataset: str) -> str:
    return {"hotpot": "hotpotqa", "2wiki": "2wikimultihopqa", "musique": "musique"}[dataset]


def _make_rag(args: argparse.Namespace):
    HippoRAG, OpenAIEmbeddingModel, BaseConfig, _ = _load_upstream(args)
    index_llm_name = getattr(args, "index_llm_model", None) or args.chat_model
    config = BaseConfig(
        save_dir=str(args.working_dir),
        dataset=_dataset_name(args.dataset),
        # HippoRAG includes llm_name in its persisted index namespace.  Keep
        # that name separate from the online QA model so frozen indexes can be
        # used for controlled backbone-transfer experiments.
        llm_name=index_llm_name,
        llm_base_url=args.chat_base_url,
        embedding_model_name=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_len=8192,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
        temperature=0,
        retrieval_top_k=args.top_k,
        linking_top_k=args.linking_top_k,
        qa_top_k=args.qa_top_k,
        openie_mode="online",
        rerank_dspy_file_path=str(
            args.hipporag_root
            / "src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json"
        ),
        force_index_from_scratch=args.force,
        force_openie_from_scratch=args.force,
    )
    llm = OpenAICompatibleLLM(
        args.chat_base_url, args.chat_model, args.max_tokens, args.seed
    )
    embedding = OpenAIEmbeddingModel(
        global_config=config, embedding_model_name=args.embedding_model
    )
    return HippoRAG(
        global_config=config,
        extraction_llm=llm,
        qa_llm=llm,
        embedding_model=embedding,
    )


def _documents(frame: pd.DataFrame, Chunk: Any) -> list[Any]:
    docs = []
    for row in frame.itertuples(index=False):
        docs.append(
            Chunk(
                content=(
                    f"Wikipedia Title: {row.title}\n"
                    f"[CHUNK_ID: {row.chunk_id}] {row.text}"
                ),
                source_id=str(row.chunk_id),
                metadata={
                    "chunk_id": str(row.chunk_id),
                    "context_id": str(row.context_id),
                    "title": str(row.title),
                },
            )
        )
    return docs


def command_index(args: argparse.Namespace) -> None:
    _, _, _, Chunk = _load_upstream(args)
    chunks = _read(args.chunks)
    if args.context_id:
        chunks = chunks[chunks["context_id"].astype(str) == str(args.context_id)]
    if args.limit is not None:
        chunks = chunks.head(args.limit)
    args.working_dir.mkdir(parents=True, exist_ok=True)
    rag = _make_rag(args)
    started = time.perf_counter()
    rag.index(_documents(chunks, Chunk))
    metadata = {
        "dataset": args.dataset,
        "documents": len(chunks),
        "elapsed_s": time.perf_counter() - started,
        "upstream_snapshot": UPSTREAM_SNAPSHOT,
        "chat_model": args.chat_model,
        "embedding_model": args.embedding_model,
    }
    (args.working_dir / "id_sgtr_hipporag2_index_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"HIPPORAG2_INDEX_COMPLETE {json.dumps(metadata)}", flush=True)


def _save(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, sep="|", index=False)
    temporary.replace(output)


def command_query(args: argparse.Namespace) -> None:
    subset = _read(args.subset)
    if args.limit is not None:
        subset = subset.head(args.limit)
    rag = _make_rag(args)
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume and args.output.exists():
        previous = _read(args.output)
        error = previous.get(
            "error", pd.Series("", index=previous.index, dtype=str)
        ).fillna("").astype(str).str.strip()
        prediction = previous["pred_answer"].fillna("").astype(str).str.strip()
        valid = error.eq("") & prediction.ne("")
        dropped = int((~valid).sum())
        previous = previous.loc[valid].copy()
        rows = previous.to_dict(orient="records")
        completed = set(previous["query_id"].astype(str))
        if dropped:
            print(f"resume will retry {dropped} failed/empty rows", flush=True)
    pending = [row for row in subset.to_dict(orient="records") if str(row["query_id"]) not in completed]
    for position, row in enumerate(pending, start=1):
        stats = CallStats()
        token = ACTIVE_STATS.set(stats)
        started = time.perf_counter()
        error = ""
        retrieval_time = generation_time = 0.0
        raw_answer = prediction = ""
        evidence: list[str] = []
        try:
            retrieval_started = time.perf_counter()
            solution = rag.retrieve([str(row["question"])], num_to_retrieve=args.top_k)[0]
            retrieval_time = time.perf_counter() - retrieval_started
            evidence = [
                str(item.get("chunk_id", ""))
                for item in (solution.doc_metadata or [])[: args.top_k]
                if item.get("chunk_id", "") != ""
            ]
            generation_started = time.perf_counter()
            solutions, raw_answers, _ = rag.qa([solution])
            generation_time = time.perf_counter() - generation_started
            raw_answer = str(raw_answers[0])
            prediction = _clean_answer(solutions[0].answer)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            ACTIVE_STATS.reset(token)
        total_time = time.perf_counter() - started
        answer_calls = sum(kind == "answer" for kind in stats.kinds)
        record = {
            "query_id": str(row["query_id"]),
            "question": str(row["question"]),
            "gold_answer": str(row.get("answer", "")),
            "pred_answer": prediction,
            "context_id": str(row.get("context_id", "")),
            "gold_evidence": str(_ids(row.get("gold_evidence", ""))),
            "retrieved_evidence": str(evidence),
            "candidate_evidence": str(evidence),
            "strategy": "HippoRAG2-PPR",
            "raw_answer": raw_answer,
            "error": error,
            "total_llm_calls": stats.calls,
            "answer_calls": answer_calls,
            "auxiliary_calls": stats.calls - answer_calls,
            "retrieval_rounds": 1,
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            "total_time_s": total_time,
            "input_tokens": stats.prompt_tokens,
            "output_tokens": stats.completion_tokens,
            "fallback": bool(error),
            "model": args.chat_model,
            "run_mode": "retrieval",
            "evidence_variant": "hipporag2_ppr",
            "temperature": 0,
            "max_output_tokens": args.max_tokens,
            "upstream_snapshot": UPSTREAM_SNAPSHOT,
        }
        rows.append(record)
        _save(rows, args.output)
        print(
            f"[{position}/{len(pending)}] {record['query_id']} calls={stats.calls} "
            f"evidence={len(evidence)} answer={prediction!r} error={error!r}",
            flush=True,
        )
    print(f"wrote {len(rows)} HippoRAG 2 predictions to {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "query"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--dataset", choices=["hotpot", "2wiki", "musique"], required=True)
        sub.add_argument("--working-dir", type=Path, required=True)
        sub.add_argument("--hipporag-root", type=Path, default=Path("third_party/HippoRAG"))
        sub.add_argument("--hipporag-python", type=Path, default=Path("third_party/hipporag_py"))
        sub.add_argument("--chat-base-url", default="http://127.0.0.1:8001/v1")
        sub.add_argument("--chat-model", default="qwen3-8b")
        sub.add_argument(
            "--index-llm-model",
            help=(
                "LLM name encoded in an existing frozen HippoRAG index; "
                "defaults to --chat-model."
            ),
        )
        sub.add_argument("--embedding-base-url", default="http://127.0.0.1:30000/v1")
        sub.add_argument("--embedding-model", default="./bge-m3")
        sub.add_argument("--embedding-batch-size", type=int, default=32)
        sub.add_argument("--max-tokens", type=int, default=512)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--top-k", type=int, default=10)
        sub.add_argument("--linking-top-k", type=int, default=5)
        sub.add_argument("--qa-top-k", type=int, default=5)
        sub.add_argument("--limit", type=int)
        sub.add_argument("--force", action="store_true")
    index_parser = subparsers.choices["index"]
    index_parser.add_argument("--chunks", type=Path, required=True)
    index_parser.add_argument("--context-id")
    query_parser = subparsers.choices["query"]
    query_parser.add_argument("--subset", type=Path, required=True)
    query_parser.add_argument("--output", type=Path, required=True)
    query_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    if args.command == "index":
        command_index(args)
    else:
        command_query(args)


if __name__ == "__main__":
    main()
