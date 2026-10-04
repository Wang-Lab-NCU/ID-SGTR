"""IRCoT baseline adapted to the frozen ID-SGTR evaluation corpora.

The control flow follows the released IRCoT implementation: retrieve first,
generate one chain-of-thought sentence, retrieve with the latest sentence, and
repeat until an answer is emitted or the step budget is exhausted.  A separate
reader call produces the final answer from all accumulated passages.

This adapter deliberately changes only infrastructure that cannot be shared
with the original release (Elasticsearch/Codex).  It uses the frozen chunk
corpus and an OpenAI-compatible local endpoint so IRCoT and ID-SGTR can be
measured on identical query IDs, source text, model service, and telemetry.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
ANSWER_RE = re.compile(r"(?:so\s+)?the\s+answer\s+is\s*:\s*(.+)", re.I)
FINAL_RE = re.compile(r"final\s+answer\s*:\s*(.+)", re.I)
SENTENCE_RE = re.compile(r"^(.+?[.!?])(?:\s|$)", re.S)


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


def _tokens(value: object) -> list[str]:
    return TOKEN_RE.findall(str(value).lower())


def _clean_answer(value: object) -> str:
    text = str(value or "").strip().strip("`").strip()
    match = FINAL_RE.search(text) or ANSWER_RE.search(text)
    if match:
        text = match.group(1).strip()
    text = text.splitlines()[0].strip() if text.splitlines() else ""
    return text.strip().strip("`").strip('"').strip("'").rstrip(".").strip()


def _first_sentence(value: object) -> str:
    """Mirror IRCoT's one-sentence-per-hop generation contract."""
    text = " ".join(str(value or "").strip().split())
    match = SENTENCE_RE.match(text)
    return (match.group(1) if match else text).strip()


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float


class OpenAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        enable_thinking: bool,
        timeout_s: float,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.enable_thinking = bool(enable_thinking)
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()

    def complete(self, system: str, user: str) -> Completion:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        error: Exception | None = None
        for attempt in range(3):
            started = time.perf_counter()
            try:
                response = self.session.post(
                    self.url,
                    headers={"Authorization": "Bearer EMPTY"},
                    json=payload,
                    timeout=(10, self.timeout_s),
                )
                response.raise_for_status()
                data = response.json()
                elapsed = time.perf_counter() - started
                usage = data.get("usage") or {}
                message = data["choices"][0]["message"]
                text = message.get("content") or ""
                if not text:
                    text = message.get("reasoning_content") or ""
                return Completion(
                    text=str(text).strip(),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    elapsed_s=elapsed,
                )
            except (requests.RequestException, KeyError, ValueError) as exc:
                error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM request failed after three attempts: {error}")


class BM25Corpus:
    def __init__(self, chunks: pd.DataFrame) -> None:
        required = {"chunk_id", "title", "text"}
        missing = required - set(chunks.columns)
        if missing:
            raise ValueError(f"chunk file is missing columns: {sorted(missing)}")
        self.chunks = chunks.reset_index(drop=True).copy()
        self.chunk_ids = self.chunks["chunk_id"].astype(str).tolist()
        self.tokenized = [
            _tokens(f"{row.title} {row.text}")
            for row in self.chunks.itertuples(index=False)
        ]
        self.index = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, count: int) -> list[int]:
        scores = np.asarray(self.index.get_scores(_tokens(query)), dtype=float)
        if scores.size == 0:
            return []
        count = min(int(count), scores.size)
        if count <= 0:
            return []
        indexes = np.argpartition(scores, -count)[-count:]
        return sorted(
            (int(index) for index in indexes),
            key=lambda index: (-float(scores[index]), self.chunk_ids[index]),
        )

    def passage(self, index: int, max_words: int) -> str:
        row = self.chunks.iloc[index]
        words = str(row["text"]).split()
        text = " ".join(words[:max_words])
        return f"Wikipedia Title: {row['title']}\n{text}"


class IRCoTRunner:
    def __init__(
        self,
        *,
        corpus: BM25Corpus,
        client: OpenAIChatClient,
        retrieval_count: int,
        max_passages: int,
        max_steps: int,
        max_passage_words: int,
        run_mode: str = "retrieval",
    ) -> None:
        self.corpus = corpus
        self.client = client
        self.retrieval_count = retrieval_count
        self.max_passages = max_passages
        self.max_steps = max_steps
        self.max_passage_words = max_passage_words
        self.run_mode = str(run_mode)

    def _context(self, selected: list[int]) -> str:
        return "\n\n".join(
            self.corpus.passage(index, self.max_passage_words)
            for index in selected
        )

    def run(self, row: dict[str, Any]) -> dict[str, Any]:
        question = str(row["question"])
        wall_started = time.perf_counter()
        retrieval_time = 0.0
        generation_time = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        selected: list[int] = []
        selected_set: set[int] = set()
        reasoning: list[str] = []
        retrieval_rounds = 0
        llm_calls = 0
        terminal_draft = ""

        query = question
        for _step in range(self.max_steps):
            retrieval_started = time.perf_counter()
            candidates = self.corpus.retrieve(query, self.retrieval_count)
            for index in candidates:
                if index not in selected_set and len(selected) < self.max_passages:
                    selected.append(index)
                    selected_set.add(index)
            retrieval_time += time.perf_counter() - retrieval_started
            retrieval_rounds += 1

            context = self._context(selected)
            chain = " ".join(reasoning) if reasoning else "(none yet)"
            completion = self.client.complete(
                "You are the reasoning component of IRCoT. Produce exactly one new "
                "reasoning sentence grounded in the supplied passages. If the answer "
                "is now determined, the sentence must end with 'So the answer is: "
                "<answer>.' Do not invent unsupported facts.",
                f"{context}\n\nQuestion: {question}\n"
                f"Reasoning so far: {chain}\nNext reasoning sentence:",
            )
            llm_calls += 1
            generation_time += completion.elapsed_s
            prompt_tokens += completion.prompt_tokens
            completion_tokens += completion.completion_tokens
            # The released implementation parses the model output with spaCy and
            # appends only ``new_sents[0]``.  Keeping the whole completion here
            # would collapse a multi-sentence chain into one apparent hop.
            thought = _first_sentence(completion.text)
            if thought:
                reasoning.append(thought)
            match = ANSWER_RE.search(thought)
            if match:
                terminal_draft = _clean_answer(match.group(1))
                break
            query = thought or question

        final = self.client.complete(
            "Answer the question using only the supplied passages and the reasoning "
            "trace. Return exactly one line in the form 'Final Answer: <answer>'.",
            f"{self._context(selected)}\n\nQuestion: {question}\n"
            f"Reasoning trace: {' '.join(reasoning)}",
        )
        llm_calls += 1
        generation_time += final.elapsed_s
        prompt_tokens += final.prompt_tokens
        completion_tokens += final.completion_tokens
        prediction = _clean_answer(final.text) or terminal_draft
        retrieved_ids = [self.corpus.chunk_ids[index] for index in selected]
        total_time = time.perf_counter() - wall_started

        return {
            "query_id": str(row["query_id"]),
            "question": question,
            "gold_answer": str(row.get("answer", row.get("gold_answer", ""))),
            "pred_answer": prediction,
            "context_id": str(row.get("context_id", "")),
            "gold_evidence": str(_ids(row.get("gold_evidence", ""))),
            "retrieved_evidence": str(retrieved_ids),
            "candidate_evidence": str(retrieved_ids),
            "strategy": "IRCoT-BM25-Iterative",
            "reasoning_trace": json.dumps(reasoning, ensure_ascii=False),
            "terminal_draft": terminal_draft,
            "total_llm_calls": llm_calls,
            "answer_calls": llm_calls,
            "auxiliary_calls": 0,
            "retrieval_rounds": retrieval_rounds,
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            "total_time_s": total_time,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "fallback": False,
            "model": self.client.model,
            "run_mode": self.run_mode,
            "evidence_variant": "ircot_bm25_iterative",
            "temperature": self.client.temperature,
            "max_output_tokens": self.client.max_tokens,
        }


def _save(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, sep="|", index=False)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--retrieval-count", type=int, default=6)
    parser.add_argument("--max-passages", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-passage-words", type=int, default=350)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--run-mode",
        choices=("retrieval", "reasoning"),
        default="retrieval",
        help="Use the full corpus (retrieval) or the query's given local context.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    subset = _read(args.subset)
    if args.limit is not None:
        subset = subset.head(args.limit)
    chunks = _read(args.chunks)
    if args.run_mode == "reasoning" and "context_id" not in chunks.columns:
        raise ValueError("reasoning mode requires context_id in the chunk file")
    corpus = BM25Corpus(chunks)
    # Warm the index before any timed query, as required by the efficiency protocol.
    corpus.retrieve("warmup query", args.retrieval_count)
    results: list[dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume and args.output.exists():
        previous = _read(args.output)
        results = previous.to_dict(orient="records")
        completed = set(previous["query_id"].astype(str))

    pending = [
        row for row in subset.to_dict(orient="records")
        if str(row["query_id"]) not in completed
    ]
    print(
        f"IRCoT corpus={len(chunks)} queries={len(subset)} pending={len(pending)} "
        f"model={args.model} concurrency={args.max_workers}",
        flush=True,
    )

    def process_row(row: dict[str, Any]) -> dict[str, Any]:
        try:
            row_corpus = corpus
            if args.run_mode == "reasoning":
                context_id = str(row.get("context_id", ""))
                local_chunks = chunks.loc[
                    chunks["context_id"].astype(str) == context_id
                ]
                if local_chunks.empty:
                    raise ValueError(
                        f"no chunks found for context_id={context_id!r}"
                    )
                # Reasoning Setting supplies the local context.  IRCoT still
                # performs its official iterative BM25/CoT control flow, but
                # retrieval is restricted to those 10/20 candidate chunks.
                row_corpus = BM25Corpus(local_chunks)
            # A client/session and runner are scoped to one query.  This keeps
            # requests.Session and the reasoning-mode corpus isolated across
            # worker threads while safely sharing the read-only global BM25
            # index in Retrieval Setting.
            row_client = OpenAIChatClient(
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
                enable_thinking=args.enable_thinking,
                timeout_s=args.timeout,
            )
            row_runner = IRCoTRunner(
                corpus=row_corpus,
                client=row_client,
                retrieval_count=args.retrieval_count,
                max_passages=args.max_passages,
                max_steps=args.max_steps,
                max_passage_words=args.max_passage_words,
                run_mode=args.run_mode,
            )
            result = row_runner.run(row)
        except Exception as exc:  # Preserve the row and allow the formal run to finish.
            result = {
                "query_id": str(row["query_id"]),
                "question": str(row["question"]),
                "gold_answer": str(row.get("answer", "")),
                "pred_answer": "",
                "context_id": str(row.get("context_id", "")),
                "gold_evidence": str(_ids(row.get("gold_evidence", ""))),
                "retrieved_evidence": "[]",
                "candidate_evidence": "[]",
                "strategy": "IRCoT-Error",
                "error": f"{type(exc).__name__}: {exc}",
                "total_llm_calls": 0,
                "answer_calls": 0,
                "auxiliary_calls": 0,
                "retrieval_rounds": 0,
                "retrieval_time_s": 0.0,
                "generation_time_s": 0.0,
                "total_time_s": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "fallback": True,
                "model": args.model,
                "run_mode": args.run_mode,
                "evidence_variant": "ircot_bm25_iterative",
                "temperature": args.temperature,
                "max_output_tokens": args.max_tokens,
            }
        return result

    max_workers = max(1, int(args.max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for row in pending]
        for position, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            _save(results, args.output)
            print(
                f"[{position}/{len(pending)}] {result['query_id']} "
                f"calls={result['total_llm_calls']} "
                f"rounds={result['retrieval_rounds']} "
                f"answer={result['pred_answer']!r}",
                flush=True,
            )
    print(f"wrote {len(results)} IRCoT predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
