# ID-SGTR

[English](README.md) | [简体中文](README.zh-CN.md)

ID-SGTR is a multi-hop question-answering system combining explicit relation edges, contextual-proximity edges, and original source passages. Graph paths guide evidence selection; the answer model reads the passages rather than triples alone. This repository contains code and fixed query subsets for HotpotQA, 2WikiMultiHopQA, and MuSiQue. Model weights and generated graph data are separate.

## Inference settings

- **Retrieval:** BGE-M3 retrieves chunks globally; BGE-Reranker-v2-M3 reranks candidate contexts; one context is locked for subsequent graph search and evidence selection.
- **Reasoning:** the benchmark supplies a local context, so global context selection is skipped. The remaining reranking, graph, evidence, and answering pipeline is shared.

The setting is selected by the entry point (`query_global.py` or `query_local.py`), not by changing a flag on the same script. In Retrieval, the locked context is a boundary: later evidence selection and fallback must stay inside it. In Reasoning, the question's provided context is that boundary from the start.

```text
Retrieval: global chunks → candidate contexts → rerank → one context ┐
                                                                  ├→ seed entities → bounded graph paths
Reasoning: benchmark-provided local context ───────────────────────┘
→ rerank local chunks → PCEF v2 (B=3) → source-text assembly
→ initial grounded answer check → bounded hops if needed → terminal synthesis
```

The balanced profile selects path-constrained evidence with budget `B=3`. PCEF v2 starts from relevance Top-B, protects Top-1, and searches nearby sets subject to a relevance-loss cap. At most `floor(B/2)` passages may be replaced: one for B=3, two for B=5. Original passages are assembled with path information. The initial answer exits early only when the grounded confidence check passes. Otherwise, bounded hops construct state-aware queries and dynamically rerank evidence; local fallback and Terminal Recovery v2.1 handle unresolved cases. The additional **hop-level confidence gate is OFF**; ordinary intermediate-answer logic remains active.

At hops, cached original-question scores and dynamic-query scores have configured weights `0.35` and `0.45` (normalized to 43.75% and 56.25%). The separate Hop Bridge weight is `0`. The older `0.75:0.25` description is not this profile.

The seed stage considers up to 20 candidates and retains up to five; low-confidence or generic-hub cases can invoke the chat model for seed filtering. A hop-aware query uses entities and relations found so far to change the reranking target. It is not another rerank of every chunk against the unchanged original question. The initial `confidence_grounded` check requires HIGH model confidence plus a nonempty answer supported by cited or deterministically inferred passages in the current evidence window. It only controls **early exit**; it does not replace the rest of the retrieval policy. Terminal Recovery v2.1 has a separate evidence budget and deduplicates identical passage text before final synthesis.

## Requirements and data

Python 3.11 is recommended. Install with `pip install -r requirements.txt`. Inference needs an OpenAI-compatible Qwen3-8B chat service, BGE-M3 embeddings, and BGE-Reranker-v2-M3 (local or HTTP). Copy `.env.example` to `.env` and configure service URLs and credentials locally; never commit the populated file.

The model servers are **not** included in this repository. The HTTP profile expects the following interfaces; the served model names must match the names in `.env`:

| Service    | Default URL                 | Required interface                                           |
| ---------- | --------------------------- | ------------------------------------------------------------ |
| Chat model | `http://127.0.0.1:8001/v1`  | OpenAI-compatible chat completions; served name `qwen3-8b`   |
| Embeddings | `http://127.0.0.1:30000/v1` | OpenAI-compatible embeddings; served name `BAAI/bge-m3`      |
| Reranker   | `http://127.0.0.1:30001`    | `POST /rerank` accepting `model`, `query`, and `documents`; returns indexed relevance scores |

Check that the services are reachable before starting a dataset run:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/v1/models
Invoke-RestMethod http://127.0.0.1:30000/v1/models
$body = @{ model = 'BAAI/bge-reranker-v2-m3'; query = 'capital of France'; documents = @('Paris is the capital of France.') } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:30001/rerank -Method Post -ContentType 'application/json' -Body $body
```

The code also supports `ID_SGTR_RERANK_BACKEND=local` with `ID_SGTR_RERANK_MODEL` pointing to a local model directory. The supplied balanced profile uses the HTTP backend so that multiple query workers do not each load a copy of the reranker.

Download prepared input and graph archives from [ID-SGTR data on Google Drive](https://drive.google.com/drive/folders/1L9U1EToW3R_VtNAW6VyGM4cSq65kg6gf?usp=drive_link). Extract at the repository root, preserving `knowledge_graph/data_input/` and `knowledge_graph/data_output/`. Query entry points expect:

| Dataset         | Prepared-data directory                                |
| --------------- | ------------------------------------------------------ |
| HotpotQA        | `knowledge_graph/data_output/dataset/hotpot/ds1000_2/` |
| 2WikiMultiHopQA | `knowledge_graph/data_output/dataset/2wiki/ds1000/`    |
| MuSiQue         | `knowledge_graph/data_output/dataset/musique/ds1000/`  |

Each prepared split needs `qa.csv`, `graph.csv`, `chunk.csv`, `contextual_proximity.csv`, `chunks_with_embeddings.parquet`, and `concepts_merged_with_vectors.parquet`. The code release includes the intent-classifier checkpoint at `knowledge_graph/adapt/intent_classifier_struct.pth`. Dataset and model licenses apply separately.

| File                                   | Used for                                                    |
| -------------------------------------- | ----------------------------------------------------------- |
| `qa.csv`                               | Questions, answers, and context IDs used by the entry point |
| `chunk.csv`                            | Original source passages and their chunk/context IDs        |
| `graph.csv`                            | Extracted explicit relation edges                           |
| `contextual_proximity.csv`             | Contextual-proximity links                                  |
| `chunks_with_embeddings.parquet`       | Chunk and title vectors for retrieval                       |
| `concepts_merged_with_vectors.parquet` | Normalized entities, aliases, and vectors                   |

The fixed `*_1000_gold.csv`, `*_200_gold.csv`, and `*_blind800_gold.csv` files under `experiments/subsets/` define shared query IDs. They are **query cohorts**, not a replacement for the graph and passage files. Each query subset must have `query_id`, `question`, `answer`, and `context_id`; the supplied gold subsets also contain evidence annotations. Do not mix dataset variants or score files with different query IDs.

## Run the balanced profile

First install dependencies in a Python environment. The commands below use the author's local `andelie` Conda environment; on another machine activate any environment with the installed requirements. Run from this repository folder in Windows PowerShell:

```powershell
conda activate andelie
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env for your chat, embedding, and reranker services.
. .\scripts\pcef_v2_balanced.ps1
$env:ID_SGTR_RERANK_BACKEND = 'http'
$env:ID_SGTR_SAMPLE_SIZE = '3'
$env:ID_SGTR_SUBSET_FILE = 'experiments/subsets/2wiki_200_gold.csv'
$env:ID_SGTR_OUTPUT = 'results/2wiki-retrieval-smoke3.csv'
New-Item -ItemType Directory -Force results | Out-Null
python knowledge_graph/2wiki/query_global.py
```

This three-query smoke run checks the end-to-end interfaces; its score is not a paper result. Once it succeeds, run the frozen 1000-query cohort:

```powershell
$env:ID_SGTR_SAMPLE_SIZE = '1000'
$env:ID_SGTR_SUBSET_FILE = 'experiments/subsets/2wiki_1000_gold.csv'
$env:ID_SGTR_OUTPUT = 'results/2wiki-retrieval-seed42.csv'
python knowledge_graph/2wiki/query_global.py
```

For Reasoning Setting, change the output name and run `python knowledge_graph/2wiki/query_local.py`. Substitute `hotpot` or `musique` in both the script and subset path for the other datasets. `ID_SGTR_SUBSET_FILE` fixes the query cohort; without it the entry point takes the first rows of `qa.csv`. The `ID_SGTR_SAMPLE_SIZE` cap still applies when a subset is supplied. The `ID_SGTR_OUTPUT` directory must exist. On Linux, export the variables from `scripts/pcef_v2_balanced.ps1` in your shell; that PowerShell script configures parameters only and does not start model services.

The balanced profile sets PCEF v2 with protected Top-1, relevance-loss cap `0.25`, structural weight `0.35`, initial `confidence_grounded` exit ON, Hop Confidence Gate OFF, normal evidence budget `3`, terminal budget `5`, seed `42`, temperature `0`, thinking ON for main reasoning, maximum output `2048` tokens, and up to 16 client workers. Terminal Recovery v2.1 separately uses thinking OFF and up to `384` output tokens. These are experiment settings, not all module defaults. Change the seed explicitly for repeated runs.

| Configuration                                                |                  Balanced value | Meaning                                                      |
| ------------------------------------------------------------ | ------------------------------: | ------------------------------------------------------------ |
| `ID_SGTR_PCEF_VERSION` / `ID_SGTR_PATH_CONSTRAINED_SELECTION` |                   `v2` / `true` | Enable PCEF v2 evidence-set search                           |
| `ID_SGTR_PCEF_PROTECT`                                       |                             `1` | Keep the highest-scoring Top-B passage                       |
| `ID_SGTR_PCEF_MAX_RELEVANCE_DROP` / `ID_SGTR_PCEF_STRUCTURE_WEIGHT` |                 `0.25` / `0.35` | Bound relevance sacrifice; weight structural complementarity |
| `ID_SGTR_EVIDENCE_BUDGET` / `ID_SGTR_TERMINAL_EVIDENCE_BUDGET` |                       `3` / `5` | Normal evidence window / terminal synthesis window           |
| `ID_SGTR_STAGE0_POLICY` / `ID_SGTR_HOP_CONFIDENCE_GATE`      | `confidence_grounded` / `false` | Check initial early exit; do not apply an additional hard gate at hops |
| `ID_SGTR_HOP_ORIGINAL_WEIGHT` / `ID_SGTR_HOP_DYNAMIC_WEIGHT` / `ID_SGTR_HOP_BRIDGE_WEIGHT` |           `0.35` / `0.45` / `0` | Cached, dynamic-query, and separate bridge score weights     |
| `ID_SGTR_SEED_CANDIDATE_LIMIT` / `ID_SGTR_SEED_LIMIT`        |                      `20` / `5` | Candidate and retained seed limits                           |
| `ID_SGTR_MAX_WORKERS`                                        |                            `16` | Maximum client-side query tasks, not guaranteed concurrent GPU requests |

Do not change the reader model, evidence budget, data version, query IDs, or gate settings between methods in a paired comparison. Changing `ID_SGTR_SEED` from 42 to 43 or 44 gives repeated runs on the **same** queries, not new test samples.

## Evaluate and test

```powershell
python -m knowledge_graph.experiments.run_p0 evaluate --results results/2wiki-retrieval-seed42.csv --scored-output results/2wiki-retrieval-seed42-scored.csv --output results/2wiki-retrieval-seed42-metrics.json
python -m knowledge_graph.experiments.run_p0 efficiency --results results/2wiki-retrieval-seed42.csv --output results/2wiki-retrieval-seed42-efficiency.json
python -m pip install pytest
python -m pytest tests -q
```

The evaluator reports answer EM/F1/precision/recall and evidence recall, precision, and complete-evidence-set rate. The result CSV also records query IDs, predictions, evidence, model-call counts, retrieval rounds, time, and token telemetry. The efficiency summary includes means and latency percentiles. Verify the reported `count` and check for empty predictions before comparing systems; a finished process does not by itself guarantee all intended rows were scored.

For two runs on the **same query IDs**, use a paired comparison:

```powershell
python -m knowledge_graph.experiments.run_p0 compare --a results/method-a-scored.csv --b results/method-b-scored.csv --metric f1 --precomputed --samples 10000 --seed 42 --output results/a-vs-b-f1.json
```

The comparison command rejects mismatched or duplicate paired IDs. Retrieval and Reasoning are different evaluation settings and must be reported separately. Inspect all CLI subcommands with `python -m knowledge_graph.experiments.run_p0 --help`.

## Troubleshooting and publication

- **Missing `qa.csv` or parquet files:** re-extract the prepared-data archive under the repository root; check the dataset-specific split name above.
- **Model not found or HTTP error:** compare the served names and ports with `.env`; the reranker endpoint must return one indexed score per document.
- **Low GPU `Running` count with 16 workers:** workers include retrieval, reranking, and graph work. It is not an assertion that 16 LLM requests are continuously in flight.
- **Slow first run:** loading the entity index and embeddings precedes query processing; use the three-query smoke test before a 1000-query run.
- **Publication scope:** this repository does not include generated datasets, model weights, private credentials, or historical result files. Follow the separate licenses of the benchmarks and pretrained models.

## Repository map

- `knowledge_graph/{hotpot,2wiki,musique}/query_global.py`: Retrieval entry points.
- `knowledge_graph/{hotpot,2wiki,musique}/query_local.py`: Reasoning entry points.
- `knowledge_graph/experiments/pcef_v2.py`: constrained evidence-set selection.
- `knowledge_graph/experiments/query_reranker.py`: cross-encoder and dynamic hop reranking.
- `knowledge_graph/experiments/stage0_gate.py`, `terminal_recovery.py`: answer acceptance and terminal synthesis.
- `scripts/pcef_v2_balanced.ps1`: balanced-profile configuration.
- `tests/`: algorithm and evaluation checks.

Generated data, results, logs, caches, credentials, and model weights are not included in this code release.
