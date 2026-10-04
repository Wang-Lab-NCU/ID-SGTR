"""Command-line utilities for reproducible P0 experiment subsets and reports."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import importlib
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from knowledge_graph.adapt.adapt import (
    HIDDEN_DIM,
    INPUT_DIM,
    IntentClassifier,
    dynamic_weight_modulation,
)

if __package__ in {None, ""}:
    # Support: python knowledge_graph/experiments/run_p0.py ...
    # The package-style imports below also keep controlled.py's relative imports valid.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.metrics import evaluate_predictions, paired_bootstrap
    from experiments.gold import annotate_gold_evidence
    from experiments.controlled import ControlledAblationRunner
    from experiments.local_controlled import (
        LocalControlled333Runner,
        LocalControlledRunner,
        LocalControlledV2Runner,
        LocalUniverseIndex,
    )
    from experiments.folding_renderers import FoldingVariant, RenderBudgets
    from experiments.path_manifest import load_jsonl, save_jsonl
    from experiments.topology_folding_v2 import FolderConfig, TopologyFolderV2
    from experiments.runtime_manifest import RuntimeManifestBuilder
    from experiments.question_aligned_runtime import (
        QuestionAlignedRuntimeManifestBuilder,
    )
    from experiments.hybrid_audit import (
        audit_hybrid_connectivity,
        summarize_hybrid_audit,
    )
    from experiments.stratified_analysis import stratify_results
else:
    from .metrics import evaluate_predictions, paired_bootstrap
    from .gold import annotate_gold_evidence
    from .controlled import ControlledAblationRunner
    from .local_controlled import (
        LocalControlled333Runner,
        LocalControlledRunner,
        LocalControlledV2Runner,
        LocalUniverseIndex,
    )
    from .folding_renderers import FoldingVariant, RenderBudgets
    from .path_manifest import load_jsonl, save_jsonl
    from .topology_folding_v2 import FolderConfig, TopologyFolderV2
    from .runtime_manifest import RuntimeManifestBuilder
    from .question_aligned_runtime import (
        QuestionAlignedRuntimeManifestBuilder,
    )
    from .hybrid_audit import audit_hybrid_connectivity, summarize_hybrid_audit
    from .stratified_analysis import stratify_results


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|" if path.suffix.lower() == ".csv" else ",", dtype=str)


def _write_json(value: object, path: Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _column(frame: pd.DataFrame, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"none of the required columns exist: {', '.join(candidates)}")


def command_subset(args: argparse.Namespace) -> None:
    frame = _read(args.qa)
    if len(frame) < args.size:
        raise ValueError(f"requested {args.size} rows but {args.qa} contains only {len(frame)}")
    subset = frame.sample(n=args.size, random_state=args.seed).sort_index().copy()
    subset.insert(0, "query_id", [f"{args.dataset}:{index}" for index in subset.index])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, sep="|", index=False)
    print(f"wrote {len(subset)} deterministic shared queries to {args.output}")


def command_annotate_gold(args: argparse.Namespace) -> None:
    subset = _read(args.subset)
    chunks = _read(args.chunks)
    annotated = annotate_gold_evidence(subset, chunks, dataset=args.dataset, raw_path=args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.output, sep="|", index=False)
    complete = int(annotated["gold_mapping_complete"].sum())
    print(f"wrote {len(annotated)} rows to {args.output}; complete mappings: {complete}/{len(annotated)}")


def _score_frame(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = _read(path)
    pred_column = _column(frame, ("pred_answer", "prediction", "answer_prediction"))
    gold_column = _column(frame, ("gold_answer", "gold", "answer"))
    def parse_ids(value: object) -> list[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            pass
        return [item.strip() for item in text.split(",") if item.strip()]

    rows = []
    has_evidence = "retrieved_evidence" in frame and "gold_evidence" in frame
    for _, row in frame.iterrows():
        item = {"prediction": row[pred_column], "gold": row[gold_column]}
        if has_evidence:
            item["retrieved_evidence"] = parse_ids(row["retrieved_evidence"])
            item["gold_evidence"] = parse_ids(row["gold_evidence"])
        rows.append(item)
    summary, per_query = evaluate_predictions(rows)
    scored = frame.copy()
    for metric in per_query[0]:
        scored[metric] = [row[metric] for row in per_query]
    return scored, summary


def command_evaluate(args: argparse.Namespace) -> None:
    scored, summary = _score_frame(args.results)
    if args.scored_output:
        args.scored_output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.scored_output, sep="|", index=False)
    _write_json(summary, args.output)


def command_compare(args: argparse.Namespace) -> None:
    if args.precomputed:
        frame_a = _read(args.a)
        frame_b = _read(args.b)
        missing = [
            str(path)
            for path, frame in ((args.a, frame_a), (args.b, frame_b))
            if args.metric not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"precomputed metric {args.metric!r} is missing from: "
                + ", ".join(missing)
            )
    else:
        frame_a, _ = _score_frame(args.a)
        frame_b, _ = _score_frame(args.b)
    id_a = _column(frame_a, ("query_id", "context_id", "question"))
    id_b = _column(frame_b, ("query_id", "context_id", "question"))
    if frame_a[id_a].astype(str).duplicated().any():
        raise ValueError(f"{args.a} contains duplicate paired IDs")
    if frame_b[id_b].astype(str).duplicated().any():
        raise ValueError(f"{args.b} contains duplicate paired IDs")
    ids_a = set(frame_a[id_a].astype(str))
    ids_b = set(frame_b[id_b].astype(str))
    if ids_a != ids_b:
        missing_from_a = sorted(ids_b - ids_a)[:5]
        missing_from_b = sorted(ids_a - ids_b)[:5]
        raise ValueError(
            "paired result ID sets differ; "
            f"missing from A={missing_from_a}, missing from B={missing_from_b}"
        )
    paired = frame_a[[id_a, args.metric]].merge(
        frame_b[[id_b, args.metric]], left_on=id_a, right_on=id_b, suffixes=("_a", "_b"), validate="one_to_one"
    )
    result = paired_bootstrap(
        paired[f"{args.metric}_a"].astype(float).tolist(),
        paired[f"{args.metric}_b"].astype(float).tolist(),
        metric=args.metric,
        samples=args.samples,
        seed=args.seed,
    )
    _write_json(result.__dict__, args.output)


def command_stratify(args: argparse.Namespace) -> None:
    """Report the fixed, pre-registered secondary analysis strata."""

    frame, _ = _score_frame(args.results)
    if args.scored_output:
        args.scored_output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.scored_output, sep="|", index=False)
    summary = stratify_results(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, sep="|", index=False)
    if args.json_output:
        records = summary.where(pd.notna(summary), None).to_dict(orient="records")
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {len(summary)} fixed-stratum rows to {args.output}")


def command_efficiency(args: argparse.Namespace) -> None:
    frame = _read(args.results)
    required = (
        "total_llm_calls", "answer_calls", "retrieval_rounds",
        "retrieval_time_s", "generation_time_s", "total_time_s",
        "input_tokens",
    )
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"missing telemetry columns: {', '.join(missing)}")
    optional_numeric = tuple(
        column for column in ("auxiliary_calls", "output_tokens")
        if column in frame
    )
    numeric_columns = required + optional_numeric
    numeric = frame[list(numeric_columns)].apply(pd.to_numeric, errors="raise")
    total = numeric["total_time_s"]
    summary = {
        column: float(numeric[column].mean())
        for column in numeric_columns
    }
    if "fallback" in frame:
        fallback = frame["fallback"].fillna("").astype(str).str.lower().isin({"1", "true", "yes"})
    else:
        fallback = pd.Series(False, index=frame.index)
    summary.update({
        "count": int(len(frame)),
        "retrieval_time_p50_s": float(numeric["retrieval_time_s"].quantile(0.50)),
        "retrieval_time_p95_s": float(numeric["retrieval_time_s"].quantile(0.95)),
        "generation_time_p50_s": float(numeric["generation_time_s"].quantile(0.50)),
        "generation_time_p95_s": float(numeric["generation_time_s"].quantile(0.95)),
        "total_time_p50_s": float(total.quantile(0.50)),
        "total_time_p95_s": float(total.quantile(0.95)),
        "zero_answer_call_rate": float((numeric["answer_calls"] == 0).mean()),
        "one_answer_call_rate": float((numeric["answer_calls"] == 1).mean()),
        "two_answer_call_rate": float((numeric["answer_calls"] == 2).mean()),
        "three_plus_answer_call_rate": float((numeric["answer_calls"] >= 3).mean()),
        "fallback_rate": float(fallback.mean()),
    })
    for column in (
        "model", "run_mode", "evidence_variant", "temperature",
        "max_output_tokens",
    ):
        if column not in frame:
            continue
        values = frame[column].dropna().astype(str).unique().tolist()
        if len(values) == 1:
            summary[column] = values[0]
    _write_json(summary, args.output)


def command_hybrid_audit(args: argparse.Namespace) -> None:
    subset = _read(args.subset)
    explicit_graph = _read(args.graph)
    implicit_graph = _read(args.proximity)
    audited = audit_hybrid_connectivity(
        subset, explicit_graph, implicit_graph, dataset=args.dataset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audited.to_csv(args.output, sep="|", index=False)
    summary = summarize_hybrid_audit(audited, dataset=args.dataset)
    print(f"wrote {len(audited)} hybrid-audit rows to {args.output}")
    _write_json(summary, args.summary_output)


def command_controlled(args: argparse.Namespace) -> None:
    os.environ.setdefault("ID_SGTR_TEMPERATURE", "0")
    os.environ.setdefault("ID_SGTR_MAX_TOKENS", "512")
    os.environ.setdefault("ID_SGTR_ENABLE_THINKING", "false")
    model_module = importlib.import_module(f"knowledge_graph.{args.dataset}.utils")
    # Controlled variants compare evidence representations, so answer
    # rendering is deterministic and does not spend tokens on hidden thought.
    model = model_module.get_chat_model(task_type="final_reasoning")
    formatter_model = model_module.get_chat_model(task_type="final_answer")
    reference = _read(args.reference)
    chunks = _read(args.chunks)
    if "candidate_evidence" not in reference and args.variant != "oracle":
        raise ValueError("reference results must contain candidate_evidence from a topology_folding run")
    runner = ControlledAblationRunner(
        model, chunks, formatter_model=formatter_model,
        budget=args.budget, seed=args.seed,
    )
    records = reference.to_dict(orient="records")
    max_workers = max(1, int(os.getenv("ID_SGTR_MAX_WORKERS", "1")))
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(records)))) as executor:
        rows = list(executor.map(lambda row: runner.run_row(row, args.variant), records))
    output = pd.DataFrame(rows)
    output["model"] = os.getenv("SILICONFLOW_MODEL", "")
    output["temperature"] = os.getenv("ID_SGTR_TEMPERATURE", "0")
    output["max_output_tokens"] = os.getenv("ID_SGTR_MAX_TOKENS", "512")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="|", index=False)
    print(f"wrote {len(output)} controlled {args.variant} predictions to {args.output}")


def _run_controlled_local(
    args: argparse.Namespace,
    *,
    stage0_early_exit: bool,
) -> None:
    """Run a local-universe protocol with or without a Stage-0 early exit."""
    os.environ.setdefault("ID_SGTR_TEMPERATURE", "0")
    os.environ.setdefault("ID_SGTR_MAX_TOKENS", "2048")
    model_module = importlib.import_module(f"knowledge_graph.{args.dataset}.utils")
    reasoner = model_module.get_chat_model(task_type="final_reasoning")
    formatter = model_module.get_chat_model(task_type="final_answer")
    embedding_model = model_module.get_embeddings_model()

    subset = _read(args.subset)
    chunks = _read(args.chunks)
    required_subset = {"query_id", "question", "context_id", "gold_evidence"}
    missing_subset = required_subset.difference(subset.columns)
    if missing_subset:
        raise ValueError(f"subset is missing columns: {sorted(missing_subset)}")
    if not ({"answer", "gold_answer"} & set(subset.columns)):
        raise ValueError("subset must contain answer or gold_answer")

    embeddings_path = args.embeddings or args.chunks.with_name("chunks_with_embeddings.parquet")
    graph_path = args.graph or args.chunks.with_name("graph.csv")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"missing precomputed chunk embeddings: {embeddings_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"missing local graph: {graph_path}")
    embeddings = pd.read_parquet(embeddings_path)
    graph = _read(graph_path)

    index = LocalUniverseIndex(
        chunks,
        embeddings,
        graph,
        budget=args.budget,
        seed=args.seed,
        loose_threshold=args.tau_loose,
        strict_threshold=args.tau_strict,
    )
    runner_class = LocalControlled333Runner if stage0_early_exit else LocalControlledRunner
    runner = runner_class(reasoner, formatter, index)
    variant = (
        "topology_folding" if stage0_early_exit
        else args.variant
    )

    questions = subset["question"].fillna("").astype(str).tolist()
    query_vectors: list[list[float]] = []
    for start in range(0, len(questions), args.embedding_batch_size):
        batch = questions[start:start + args.embedding_batch_size]
        query_vectors.extend(embedding_model.embed_documents(batch))
    if len(query_vectors) != len(subset):
        raise RuntimeError("query embedding count does not match subset row count")

    intent_model = IntentClassifier(INPUT_DIM, HIDDEN_DIM).to(torch.device("cpu"))
    intent_path = Path(__file__).resolve().parents[1] / "adapt" / "intent_classifier_struct.pth"
    if not intent_path.exists():
        raise FileNotFoundError(f"missing intent classifier weights: {intent_path}")
    intent_model.load_state_dict(torch.load(intent_path, map_location="cpu"))
    intent_model.eval()
    intent_tensor = torch.tensor(query_vectors, dtype=torch.float32)
    intent_probabilities = intent_model.predict_proba(intent_tensor, questions)
    intents: list[tuple[str, str]] = []
    for probabilities, question in zip(intent_probabilities, questions):
        _, strategy = dynamic_weight_modulation(
            probabilities, question, dataset=args.dataset,
        )
        intent_type = next(
            (name for name in ("Retrieval", "Reasoning", "Comparative")
             if strategy.startswith(f"({name},")),
            "Default",
        )
        intents.append((intent_type, strategy))

    records = subset.to_dict(orient="records")
    max_workers = max(1, int(os.getenv("ID_SGTR_MAX_WORKERS", "1")))
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(records)))) as executor:
        rows = list(executor.map(
            lambda item: runner.run_row(
                item[0], item[1], variant,
                intent_type=item[2][0], intent_strategy=item[2][1],
            ),
            zip(records, query_vectors, intents),
        ))
    output = pd.DataFrame(rows)
    output["model"] = os.getenv("SILICONFLOW_MODEL", "")
    output["temperature"] = os.getenv("ID_SGTR_TEMPERATURE", "0")
    output["max_output_tokens"] = os.getenv("ID_SGTR_MAX_TOKENS", "2048")
    output["local_candidate_protocol"] = (
        "complete_context_333_stage0_early_exit"
        if stage0_early_exit else "complete_context"
    )
    output["tau_loose"] = index.loose_threshold
    output["tau_strict"] = index.strict_threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="|", index=False)
    print(
        f"wrote {len(output)} "
        f"{'controlled-local-333 topology_folding' if stage0_early_exit else f'controlled-local {variant}'} predictions "
        f"to {args.output}"
    )


def command_controlled_local(args: argparse.Namespace) -> None:
    """Run a strict Stage-0-disabled controlled local variant."""
    _run_controlled_local(args, stage0_early_exit=False)


def command_controlled_local_333(args: argparse.Namespace) -> None:
    """Run controlled B=(3,3,3) topology folding with Stage-0 early exit."""
    _run_controlled_local(args, stage0_early_exit=True)


def _require_local_subset(subset: pd.DataFrame) -> None:
    required = {"query_id", "question", "context_id"}
    missing = required.difference(subset.columns)
    if missing:
        raise ValueError(f"subset is missing columns: {sorted(missing)}")
    if subset["query_id"].astype(str).duplicated().any():
        raise ValueError("subset contains duplicate query_id values")


def _embed_questions(
    embedding_model: object,
    questions: list[str],
    batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(questions), batch_size):
        vectors.extend(embedding_model.embed_documents(
            questions[start:start + batch_size]
        ))
    if len(vectors) != len(questions):
        raise RuntimeError("query embedding count does not match subset row count")
    return vectors


def _classify_intents(
    questions: list[str],
    query_vectors: list[list[float]],
    dataset: str,
) -> list[tuple[str, str]]:
    intent_model = IntentClassifier(INPUT_DIM, HIDDEN_DIM).to(torch.device("cpu"))
    intent_path = Path(__file__).resolve().parents[1] / "adapt" / "intent_classifier_struct.pth"
    if not intent_path.exists():
        raise FileNotFoundError(f"missing intent classifier weights: {intent_path}")
    intent_model.load_state_dict(torch.load(intent_path, map_location="cpu"))
    intent_model.eval()
    probabilities = intent_model.predict_proba(
        torch.tensor(query_vectors, dtype=torch.float32), questions,
    )
    intents: list[tuple[str, str]] = []
    for values, question in zip(probabilities, questions):
        _, strategy = dynamic_weight_modulation(values, question, dataset=dataset)
        intent_type = next(
            (name for name in ("Retrieval", "Reasoning", "Comparative")
             if strategy.startswith(f"({name},")),
            "Default",
        )
        intents.append((intent_type, strategy))
    return intents


def command_build_path_manifest(args: argparse.Namespace) -> None:
    """Build one deterministic, gold-independent path manifest per query."""

    subset = _read(args.subset)
    chunks = _read(args.chunks)
    _require_local_subset(subset)
    embeddings_path = args.embeddings or args.chunks.with_name(
        "chunks_with_embeddings.parquet"
    )
    graph_path = args.graph or args.chunks.with_name("graph.csv")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"missing precomputed chunk embeddings: {embeddings_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"missing local graph: {graph_path}")
    embeddings = pd.read_parquet(embeddings_path)
    graph = _read(graph_path)
    model_module = importlib.import_module(f"knowledge_graph.{args.dataset}.utils")
    embedding_model = model_module.get_embeddings_model()
    questions = subset["question"].fillna("").astype(str).tolist()
    query_vectors = _embed_questions(
        embedding_model, questions, args.embedding_batch_size,
    )
    intents = _classify_intents(questions, query_vectors, args.dataset)
    config = FolderConfig(
        budget=args.budget,
        loose_threshold=args.tau_loose,
        strict_threshold=args.tau_strict,
        beam_width=args.beam_width,
        max_path_steps=args.max_path_steps,
        min_edge_score=args.min_edge_score,
        path_confidence_threshold=args.path_confidence_threshold,
        anchor_threshold=args.anchor_threshold,
        reverse_penalty=args.reverse_penalty,
        source_token_budget=args.source_token_budget,
        trace_token_budget=args.trace_token_budget,
        total_evidence_token_budget=args.total_evidence_token_budget,
    )
    folder = TopologyFolderV2(
        chunks, embeddings, graph, dataset=args.dataset, config=config,
    )
    records = subset.to_dict(orient="records")
    payload = list(zip(records, query_vectors, intents))
    max_workers = max(1, int(os.getenv("ID_SGTR_MAX_WORKERS", "1")))

    def build(item: tuple[dict[str, object], list[float], tuple[str, str]]):
        row, vector, intent = item
        return folder.build_manifest(
            query_id=str(row["query_id"]),
            context_id=str(row["context_id"]),
            question=str(row["question"]),
            query_vector=vector,
            intent_type=intent[0],
            intent_strategy=intent[1],
        )

    with ThreadPoolExecutor(
        max_workers=min(max_workers, max(1, len(payload)))
    ) as executor:
        manifests = list(executor.map(build, payload))
    save_jsonl(args.output, manifests)
    fallback_counts = Counter(
        manifest.fallback_reason for manifest in manifests if not manifest.foldable
    )
    print(
        f"wrote {len(manifests)} frozen path manifests to {args.output}; "
        f"foldable={sum(manifest.foldable for manifest in manifests)}, "
        f"fallback={sum(not manifest.foldable for manifest in manifests)}"
    )
    if fallback_counts:
        print("fallback reasons:", dict(sorted(fallback_counts.items())))


def command_build_runtime_manifest(args: argparse.Namespace) -> None:
    """Freeze manifests from actual pre-answer online graph trajectories."""

    reference = _read(args.reference)
    chunks = _read(args.chunks)
    graph = _read(args.graph)
    question_aligned = args.selection_policy in {
        "question-aligned",
        "question-aligned-conservative",
        "question-aligned-conservative-factorial",
    }
    builder_class = (
        QuestionAlignedRuntimeManifestBuilder
        if question_aligned
        else RuntimeManifestBuilder
    )
    builder_kwargs = {}
    if question_aligned:
        builder_kwargs["beam_width"] = args.beam_width
        builder_kwargs["conservative_gate"] = (
            args.selection_policy in {
                "question-aligned-conservative",
                "question-aligned-conservative-factorial",
            }
        )
        builder_kwargs["factorial_clean"] = (
            args.selection_policy
            == "question-aligned-conservative-factorial"
        )
    builder = builder_class(
        chunks,
        graph,
        dataset=args.dataset,
        budget=args.budget,
        source_token_budget=args.source_token_budget,
        trace_token_budget=args.trace_token_budget,
        total_evidence_token_budget=args.total_evidence_token_budget,
        **builder_kwargs,
    )
    manifests = builder.build_frame(reference)
    save_jsonl(args.output, manifests)
    fallback_counts = Counter(
        manifest.fallback_reason for manifest in manifests if not manifest.foldable
    )
    print(
        f"wrote {len(manifests)} runtime trajectory manifests to {args.output}; "
        f"foldable={sum(manifest.foldable for manifest in manifests)}, "
        f"fallback={sum(not manifest.foldable for manifest in manifests)}"
    )
    if fallback_counts:
        print("fallback reasons:", dict(sorted(fallback_counts.items())))


def command_validate_manifest(args: argparse.Namespace) -> None:
    """Validate manifest invariants and optional source-table alignment."""

    manifests = load_jsonl(args.manifest, validate=True)
    if not manifests:
        raise ValueError("path manifest must not be empty")
    by_id = {manifest.query_id: manifest for manifest in manifests}
    if args.subset:
        subset = _read(args.subset)
        _require_local_subset(subset)
        for row in subset.to_dict(orient="records"):
            query_id = str(row["query_id"])
            if query_id not in by_id:
                raise ValueError(f"subset query {query_id!r} is missing from manifest")
            manifest = by_id[query_id]
            if str(row["context_id"]) != manifest.context_id:
                raise ValueError(f"context mismatch for query {query_id!r}")
            if str(row["question"]).strip() != manifest.question:
                raise ValueError(f"question mismatch for query {query_id!r}")
    if args.chunks:
        chunks = _read(args.chunks)
        available = {
            (str(row.context_id), str(row.chunk_id))
            for row in chunks.itertuples(index=False)
        }
        for manifest in manifests:
            missing = [
                chunk_id for chunk_id in manifest.candidate_chunk_ids
                if (manifest.context_id, chunk_id) not in available
            ]
            if missing:
                raise ValueError(
                    f"manifest {manifest.query_id!r} contains unknown chunks: {missing[:5]}"
                )
    if args.graph:
        graph = _read(args.graph)
        relation_rows = {
            (
                str(row.context_id), str(row.chunk_id), str(row.node_1).strip(),
                str(row.edge).strip(), str(row.node_2).strip(),
            )
            for row in graph.itertuples(index=False)
        }
        for manifest in manifests:
            for step in manifest.path_steps:
                key = (
                    manifest.context_id,
                    step.supporting_chunk_id,
                    step.canonical_source_entity,
                    step.relation,
                    step.canonical_target_entity,
                )
                if key not in relation_rows:
                    raise ValueError(
                        f"manifest {manifest.query_id!r} contains a path step "
                        "that is not aligned to graph.csv"
                    )
    summary = {
        "count": len(manifests),
        "unique_hashes": len({manifest.sha256 for manifest in manifests}),
        "foldable": sum(manifest.foldable for manifest in manifests),
        "foldable_rate": (
            sum(manifest.foldable for manifest in manifests) / len(manifests)
            if manifests else 0.0
        ),
        "fallback": sum(manifest.fallback_to_graph_naive for manifest in manifests),
        "fallback_reasons": dict(sorted(Counter(
            manifest.fallback_reason for manifest in manifests
            if manifest.fallback_reason
        ).items())),
        "mean_selected_count": (
            sum(manifest.selected_count for manifest in manifests) / len(manifests)
            if manifests else 0.0
        ),
        "mean_unused_budget": (
            sum(manifest.unused_budget for manifest in manifests) / len(manifests)
            if manifests else 0.0
        ),
        "manifest_version": sorted({manifest.manifest_version for manifest in manifests}),
        "implementation_version": sorted({
            manifest.implementation_version for manifest in manifests
        }),
    }
    _write_json(summary, args.output)


def _load_local_tokenizer(path: Path | None) -> object | None:
    if path is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "--tokenizer requires transformers in the current environment"
        ) from exc
    return AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True,
    )


def command_controlled_local_v2(args: argparse.Namespace) -> None:
    """Replay one frozen manifest using exactly one representation factor cell."""

    os.environ.setdefault("ID_SGTR_TEMPERATURE", "0")
    os.environ.setdefault("ID_SGTR_MAX_TOKENS", "2048")
    subset = _read(args.subset)
    chunks = _read(args.chunks)
    _require_local_subset(subset)
    if not ({"answer", "gold_answer"} & set(subset.columns)):
        raise ValueError("subset must contain answer or gold_answer")
    manifests = load_jsonl(args.manifest, validate=True)
    if not manifests:
        raise ValueError("path manifest must not be empty")
    wrong_dataset = [
        manifest.query_id for manifest in manifests
        if manifest.dataset != args.dataset
    ]
    if wrong_dataset:
        raise ValueError(
            f"manifest dataset does not match --dataset {args.dataset!r}: "
            f"{wrong_dataset[:5]}"
        )
    by_id = {manifest.query_id: manifest for manifest in manifests}
    subset_ids = set(subset["query_id"].astype(str))
    manifest_ids = set(by_id)
    if subset_ids != manifest_ids:
        raise ValueError(
            "subset and path manifest query IDs differ; "
            f"missing manifests={sorted(subset_ids - manifest_ids)[:5]}, "
            f"extra manifests={sorted(manifest_ids - subset_ids)[:5]}"
        )
    missing = [
        query_id for query_id in subset["query_id"].astype(str)
        if query_id not in by_id
    ]
    if missing:
        raise ValueError(f"queries missing from path manifest: {missing[:5]}")

    model_module = importlib.import_module(f"knowledge_graph.{args.dataset}.utils")
    reasoner = model_module.get_chat_model(task_type="final_reasoning")
    formatter = model_module.get_chat_model(task_type="final_answer")
    tokenizer = _load_local_tokenizer(args.tokenizer)
    requested_budget_values = (
        args.source_token_budget,
        args.trace_token_budget,
        args.total_evidence_token_budget,
    )
    budgets = None
    if any(value is not None for value in requested_budget_values):
        if not all(value is not None for value in requested_budget_values):
            raise ValueError(
                "budget verification requires all three budget arguments"
            )
        budgets = RenderBudgets(
            source_tokens=args.source_token_budget,
            trace_tokens=args.trace_token_budget,
            total_tokens=args.total_evidence_token_budget,
        )
        mismatched = [
            manifest.query_id for manifest in manifests
            if RenderBudgets.from_manifest(manifest) != budgets
        ]
        if mismatched:
            raise ValueError(
                "CLI budgets differ from the frozen manifest for queries: "
                f"{mismatched[:5]}"
            )
    runner = LocalControlledV2Runner(
        reasoner, formatter, chunks, budgets=budgets, tokenizer=tokenizer,
    )
    records = subset.to_dict(orient="records")
    max_workers = max(1, int(os.getenv("ID_SGTR_MAX_WORKERS", "1")))
    with ThreadPoolExecutor(
        max_workers=min(max_workers, max(1, len(records)))
    ) as executor:
        rows = list(executor.map(
            lambda row: runner.run_row(
                row, by_id[str(row["query_id"])], args.variant,
            ),
            records,
        ))
    output = pd.DataFrame(rows)
    output["model"] = os.getenv("SILICONFLOW_MODEL", "")
    output["temperature"] = os.getenv("ID_SGTR_TEMPERATURE", "0")
    output["max_output_tokens"] = os.getenv("ID_SGTR_MAX_TOKENS", "2048")
    output["local_candidate_protocol"] = "frozen_path_manifest_v2"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="|", index=False)
    print(
        f"wrote {len(output)} controlled-local-v2 {args.variant} predictions "
        f"to {args.output}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subset = subparsers.add_parser("subset", help="create deterministic shared query IDs")
    subset.add_argument("--qa", type=Path, required=True)
    subset.add_argument("--dataset", required=True)
    subset.add_argument("--size", type=int, default=200)
    subset.add_argument("--seed", type=int, default=42)
    subset.add_argument("--output", type=Path, required=True)
    subset.set_defaults(func=command_subset)

    annotate = subparsers.add_parser("annotate-gold", help="map gold supporting titles to chunk IDs")
    annotate.add_argument("--subset", type=Path, required=True)
    annotate.add_argument("--chunks", type=Path, required=True)
    annotate.add_argument("--raw", type=Path, required=True)
    annotate.add_argument("--dataset", choices=("hotpot", "2wiki", "musique"), required=True)
    annotate.add_argument("--output", type=Path, required=True)
    annotate.set_defaults(func=command_annotate_gold)

    evaluate = subparsers.add_parser("evaluate", help="compute consistent EM/F1 metrics")
    evaluate.add_argument("--results", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--scored-output", type=Path)
    evaluate.set_defaults(func=command_evaluate)

    compare = subparsers.add_parser("compare", help="paired bootstrap comparison")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    compare.add_argument("--metric", choices=("em", "f1"), default="f1")
    compare.add_argument("--samples", type=int, default=10_000)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument(
        "--precomputed",
        action="store_true",
        help=(
            "read the requested metric directly from both input files "
            "instead of recomputing it from prediction/gold columns"
        ),
    )
    compare.add_argument("--output", type=Path)
    compare.set_defaults(func=command_compare)

    stratify = subparsers.add_parser(
        "stratify",
        help="summarize fixed pre-registered Topology Folding v2 strata",
    )
    stratify.add_argument("--results", type=Path, required=True)
    stratify.add_argument("--output", type=Path, required=True)
    stratify.add_argument("--scored-output", type=Path)
    stratify.add_argument("--json-output", type=Path)
    stratify.set_defaults(func=command_stratify)

    efficiency = subparsers.add_parser("efficiency", help="aggregate full-query latency/call telemetry")
    efficiency.add_argument("--results", type=Path, required=True)
    efficiency.add_argument("--output", type=Path)
    efficiency.set_defaults(func=command_efficiency)

    hybrid_audit = subparsers.add_parser(
        "hybrid-audit",
        help="audit explicit connectivity and implicit-edge repairs without an LLM",
    )
    hybrid_audit.add_argument("--subset", type=Path, required=True)
    hybrid_audit.add_argument("--graph", type=Path, required=True)
    hybrid_audit.add_argument("--proximity", type=Path, required=True)
    hybrid_audit.add_argument(
        "--dataset", choices=("hotpot", "2wiki", "musique"), required=True,
    )
    hybrid_audit.add_argument("--output", type=Path, required=True)
    hybrid_audit.add_argument("--summary-output", type=Path)
    hybrid_audit.set_defaults(func=command_hybrid_audit)

    controlled = subparsers.add_parser("controlled", help="replay one frozen retrieval manifest")
    controlled.add_argument("--reference", type=Path, required=True)
    controlled.add_argument("--chunks", type=Path, required=True)
    controlled.add_argument("--dataset", choices=("hotpot", "2wiki", "musique"), required=True)
    controlled.add_argument(
        "--variant",
        choices=("triple_only", "source_score", "source_random", "topology_folding", "oracle"),
        required=True,
    )
    controlled.add_argument("--budget", type=int, default=3)
    controlled.add_argument("--seed", type=int, default=42)
    controlled.add_argument("--output", type=Path, required=True)
    controlled.set_defaults(func=command_controlled)

    controlled_local = subparsers.add_parser(
        "controlled-local",
        help="select B chunks from each query's complete Reasoning-Setting context",
    )
    controlled_local.add_argument("--subset", type=Path, required=True)
    controlled_local.add_argument("--chunks", type=Path, required=True)
    controlled_local.add_argument("--embeddings", type=Path)
    controlled_local.add_argument("--graph", type=Path)
    controlled_local.add_argument(
        "--dataset", choices=("hotpot", "2wiki", "musique"), required=True,
    )
    controlled_local.add_argument(
        "--variant",
        choices=(
            "triple_only", "source_score", "source_random",
            "graph_naive", "topology_folding", "oracle", "entity_to_chunk",
        ),
        required=True,
    )
    controlled_local.add_argument("--budget", type=int, default=3)
    controlled_local.add_argument("--seed", type=int, default=42)
    controlled_local.add_argument("--embedding-batch-size", type=int, default=32)
    controlled_local.add_argument("--tau-loose", type=float)
    controlled_local.add_argument("--tau-strict", type=float)
    controlled_local.add_argument("--output", type=Path, required=True)
    controlled_local.set_defaults(func=command_controlled_local)

    controlled_local_333 = subparsers.add_parser(
        "controlled-local-333",
        help=(
            "run B=(3,3,3) topology folding over the complete local context "
            "with a Stage-0 early exit"
        ),
    )
    controlled_local_333.add_argument("--subset", type=Path, required=True)
    controlled_local_333.add_argument("--chunks", type=Path, required=True)
    controlled_local_333.add_argument("--embeddings", type=Path)
    controlled_local_333.add_argument("--graph", type=Path)
    controlled_local_333.add_argument(
        "--dataset", choices=("hotpot", "2wiki", "musique"), required=True,
    )
    controlled_local_333.add_argument("--seed", type=int, default=42)
    controlled_local_333.add_argument("--embedding-batch-size", type=int, default=32)
    controlled_local_333.add_argument("--tau-loose", type=float)
    controlled_local_333.add_argument("--tau-strict", type=float)
    controlled_local_333.add_argument("--output", type=Path, required=True)
    controlled_local_333.set_defaults(
        func=command_controlled_local_333,
        budget=3,
    )

    build_manifest = subparsers.add_parser(
        "build-path-manifest",
        help="build one gold-independent directed path manifest per local query",
    )
    build_manifest.add_argument("--subset", type=Path, required=True)
    build_manifest.add_argument("--chunks", type=Path, required=True)
    build_manifest.add_argument("--embeddings", type=Path)
    build_manifest.add_argument("--graph", type=Path)
    build_manifest.add_argument(
        "--dataset", choices=("hotpot", "2wiki", "musique"), required=True,
    )
    build_manifest.add_argument("--budget", type=int, default=3)
    build_manifest.add_argument("--embedding-batch-size", type=int, default=32)
    build_manifest.add_argument("--tau-loose", type=float, default=0.25)
    build_manifest.add_argument("--tau-strict", type=float, default=0.45)
    build_manifest.add_argument("--beam-width", type=int, default=12)
    build_manifest.add_argument("--max-path-steps", type=int, default=3)
    build_manifest.add_argument("--min-edge-score", type=float, default=0.18)
    build_manifest.add_argument(
        "--path-confidence-threshold", type=float, default=0.24,
    )
    build_manifest.add_argument("--anchor-threshold", type=float, default=0.34)
    build_manifest.add_argument("--reverse-penalty", type=float, default=0.04)
    build_manifest.add_argument("--source-token-budget", type=int, default=1400)
    build_manifest.add_argument("--trace-token-budget", type=int, default=96)
    build_manifest.add_argument(
        "--total-evidence-token-budget", type=int, default=1536,
    )
    build_manifest.add_argument("--output", type=Path, required=True)
    build_manifest.set_defaults(func=command_build_path_manifest)

    runtime_manifest = subparsers.add_parser(
        "build-runtime-manifest",
        help=(
            "freeze source-aligned manifests from actual ID-SGTR runtime "
            "trajectory telemetry"
        ),
    )
    runtime_manifest.add_argument("--reference", type=Path, required=True)
    runtime_manifest.add_argument("--chunks", type=Path, required=True)
    runtime_manifest.add_argument("--graph", type=Path, required=True)
    runtime_manifest.add_argument(
        "--dataset", choices=("hotpot", "2wiki", "musique"), required=True,
    )
    runtime_manifest.add_argument("--budget", type=int, default=3)
    runtime_manifest.add_argument(
        "--selection-policy",
        choices=(
            "runtime-final",
            "question-aligned",
            "question-aligned-conservative",
            "question-aligned-conservative-factorial",
        ),
        default="runtime-final",
        help=(
            "runtime-final preserves the online final evidence set; "
            "question-aligned reselects B chunks from the frozen pre-answer "
            "runtime candidate pool using relation-slot path coverage; "
            "question-aligned-conservative additionally rejects one-hop, "
            "multi-replacement, Retrieval, and old-core-evicting proposals; "
            "question-aligned-conservative-factorial also prevents path-edge "
            "scores from changing the Graph-Naive semantic score order"
        ),
    )
    runtime_manifest.add_argument(
        "--beam-width",
        type=int,
        default=24,
        help="beam width used only by question-aligned selection",
    )
    runtime_manifest.add_argument("--source-token-budget", type=int, default=1400)
    runtime_manifest.add_argument("--trace-token-budget", type=int, default=96)
    runtime_manifest.add_argument(
        "--total-evidence-token-budget", type=int, default=1536,
    )
    runtime_manifest.add_argument("--output", type=Path, required=True)
    runtime_manifest.set_defaults(func=command_build_runtime_manifest)

    validate_manifest = subparsers.add_parser(
        "validate-manifest",
        help="validate path continuity, IDs, graph alignment, budgets, and hashes",
    )
    validate_manifest.add_argument("--manifest", type=Path, required=True)
    validate_manifest.add_argument("--subset", type=Path)
    validate_manifest.add_argument("--chunks", type=Path)
    validate_manifest.add_argument("--graph", type=Path)
    validate_manifest.add_argument("--output", type=Path)
    validate_manifest.set_defaults(func=command_validate_manifest)

    controlled_local_v2 = subparsers.add_parser(
        "controlled-local-v2",
        help="replay a frozen path manifest without retrieval or ID reselection",
    )
    controlled_local_v2.add_argument("--subset", type=Path, required=True)
    controlled_local_v2.add_argument("--chunks", type=Path, required=True)
    controlled_local_v2.add_argument("--manifest", type=Path, required=True)
    controlled_local_v2.add_argument(
        "--dataset", choices=("hotpot", "2wiki", "musique"), required=True,
    )
    controlled_local_v2.add_argument(
        "--variant",
        choices=tuple(variant.value for variant in FoldingVariant),
        required=True,
    )
    controlled_local_v2.add_argument("--tokenizer", type=Path)
    controlled_local_v2.add_argument("--source-token-budget", type=int)
    controlled_local_v2.add_argument("--trace-token-budget", type=int)
    controlled_local_v2.add_argument(
        "--total-evidence-token-budget", type=int,
    )
    controlled_local_v2.add_argument("--output", type=Path, required=True)
    controlled_local_v2.set_defaults(func=command_controlled_local_v2)
    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
