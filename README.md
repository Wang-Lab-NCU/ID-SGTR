# ID-SGTR: Hybrid Knowledge Graph Reasoning for Multi-hop QA

This repository contains code for hybrid knowledge graph construction and reasoning for multi-hop question answering. The project mainly targets HotpotQA, 2WikiMultiHopQA, and MuSiQue. It converts textual corpora into a hybrid graph composed of explicit entity relations, contextual proximity edges, and dense vector representations, then answers questions through intent-aware graph reasoning.

> Note: the code in `knowledge_graph/2wiki/`, `knowledge_graph/hotpot/`, and `knowledge_graph/musique/` follows almost the same structure. This README uses `knowledge_graph/hotpot/` as the main example. To run another dataset, the main change is usually the data path.

## Repository Structure

```text
.
+-- knowledge_graph/
|   +-- 2wiki/              # Code for 2WikiMultiHopQA
|   +-- hotpot/             # Code for HotpotQA, used as the default example
|   +-- musique/            # Code for MuSiQue
|   +-- adapt/              # Intent classification network
|   +-- data_input/         # Raw datasets and training data
|   +-- data_output/        # Generated graphs, intermediate files, vectors, and QA files
|   +-- models/             # Local model files
|   `-- environment.yml     # Conda environment
+-- data_output/            # Optional external/generated data directory
+-- requirements.txt        # Pip dependency list
`-- pyproject.toml          # Poetry dependency configuration
```

## Main Components

| Path                                                         | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `knowledge_graph/2wiki/`, `knowledge_graph/hotpot/`, `knowledge_graph/musique/` | Dataset-specific code for graph construction, querying, baselines, and evaluation. The logic is largely shared across the three directories. |
| `knowledge_graph/adapt/`                                     | Intent classification network. It classifies questions into `Retrieval`, `Reasoning`, and `Comparative`, then modulates graph reasoning weights. |
| `knowledge_graph/hotpot/build_kg.ipynb`                      | Notebook for hybrid knowledge graph construction. In some dataset folders, the filename may appear as `bulid_kg.ipynb`. |
| `knowledge_graph/hotpot/seed.py`                             | Four-dimensional seed node anchoring module. It combines entity-name embeddings, description embeddings, exact matching, and BM25 sparse matching. |
| `knowledge_graph/hotpot/query_global.py`                     | Main method under the global reasoning setting. It performs global seed anchoring, intent-aware weight modulation, and multi-hop graph reasoning. |
| `knowledge_graph/hotpot/query_local.py`                      | Local reasoning setting used in experiments. It usually restricts reasoning to the sample-level context or local subgraph. |
| `knowledge_graph/hotpot/run_naive_rag.py`                    | Naive dense-vector RAG baseline.                             |
| `knowledge_graph/hotpot/evalue.py`                           | Evaluation script with EM, F1, precision, recall, and contain-match style metrics. |

## Method Overview

The pipeline has three main stages:

1. Hybrid knowledge graph construction  
   Raw QA data is processed into text chunks, resolved chunks, extracted entities or concepts, explicit relation edges, contextual proximity edges, and vector representations.

2. Seed node anchoring  
   `seed.py` uses `SemanticMatcher` to locate starting nodes in the graph. The matching score combines four signals: entity-name vector similarity, entity-description vector similarity, exact lexical matching, and BM25 sparse matching.

3. Graph reasoning and answer generation  
   `query_global.py` or `query_local.py` loads the graph, intent classifier, and vector files. The system predicts the question intent, dynamically adjusts reasoning weights over explicit, semantic, and contextual edges, then performs multi-hop search and calls an LLM to generate the final answer.

If `knowledge_graph/assets/system.png` exists, GitHub will render the method figure below:

![Method](knowledge_graph/assets/system.png)

## Environment Setup

Python 3.11 is recommended. You can install the dependencies with pip:

```bash
pip install -r requirements.txt
```

If you prefer Conda, create the environment from the provided file:

```bash
conda env create -f knowledge_graph/environment.yml
conda activate knowledge-graph
pip install -r requirements.txt
```

Poetry is also supported:

```bash
poetry install
```

## Data Preparation

Raw datasets are usually placed under:

```text
knowledge_graph/data_input/dataset/
```

Generated graph files, intermediate files, and vector files are usually stored under:

```text
knowledge_graph/data_output/dataset/{dataset_name}/{split_or_size}/
```

For HotpotQA, the query scripts expect a directory similar to:

```text
knowledge_graph/data_output/dataset/hotpot/ds1000_2/
+-- qa.csv
+-- graph.csv
+-- chunk.csv
+-- chunks_with_embeddings.parquet
+-- concepts_merged_with_vectors.parquet
`-- contextual_proximity.csv
```

Common generated files:

| File                                   | Description                                      |
| -------------------------------------- | ------------------------------------------------ |
| `qa.csv`                               | Questions, answers, and context IDs.             |
| `chunk.csv`                            | Original text chunks.                            |
| `resolved_chunks.csv`                  | Text chunks after coreference resolution.        |
| `graph.csv`                            | Explicit entity relation edges.                  |
| `contextual_proximity.csv`             | Contextual co-occurrence edges.                  |
| `chunks_with_embeddings.parquet`       | Text chunks with dense embeddings.               |
| `concepts_merged_with_vectors.parquet` | Canonicalized entities or concepts with vectors. |

## Running the Project

### 1. Build the Knowledge Graph

Open and run the dataset-specific notebook:

```text
knowledge_graph/hotpot/build_kg.ipynb
```

For other datasets, the notebook may be named:

```text
knowledge_graph/2wiki/build_kg.ipynb
knowledge_graph/musique/build_kg.ipynb
```

After running the notebook, check that `knowledge_graph/data_output/dataset/...` contains files such as `graph.csv`, `chunk.csv`, `qa.csv`, `contextual_proximity.csv`, and the corresponding Parquet vector files.

### 2. Train or Load the Intent Classifier

The intent classifier is located in `knowledge_graph/adapt/`. To retrain it:

```bash
python knowledge_graph/adapt/train_adapt.py
```

Training data examples:

```text
knowledge_graph/adapt/train.csv
knowledge_graph/adapt/train2.csv
```

The trained weights are saved by default to:

```text
knowledge_graph/adapt/intent_classifier_struct.pth
```

This model is loaded by `query_global.py` and `query_local.py` to produce dynamic graph reasoning weights.

### 3. Run the Main Method: Global Reasoning

For HotpotQA:

```bash
python knowledge_graph/hotpot/query_global.py
```

Before running, check the path configuration near the bottom of the script. The HotpotQA script uses repository-relative paths by default:

```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data_output", "dataset", "hotpot", "ds10_2")
ADAPT_ROOT = os.path.join(PROJECT_ROOT, "adapt")
```

To switch to 2WikiMultiHopQA or MuSiQue, modify `DATA_ROOT` to the corresponding output directory, or run the `query_global.py` script inside the matching dataset folder.

### 4. Run the Local Reasoning Setting

The local reasoning setting is used for experimental comparison:

```bash
python knowledge_graph/hotpot/query_local.py
```

This script typically anchors and reasons within a sample-level context or local subgraph, making it useful for comparison with the global setting.

### 5. Run Baselines and Evaluation

Naive dense-vector RAG baseline:

```bash
python knowledge_graph/hotpot/run_naive_rag.py
```

Evaluation:

```bash
python knowledge_graph/hotpot/evalue.py
```

The evaluation script reads result CSV files and computes metrics such as EM, F1, precision, recall, and contain-match. Check the input result path in the script before running.

## Model and API Configuration

Model loading utilities are centralized in `utils.py`. Depending on the provider enabled in your local code, you may need to configure API keys and model names in `.env` or environment variables:

```text
SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=
SILICONFLOW_MODEL=
SILICONFLOW_EMBEDDINGS_MODEL =
```

The exact variables required depend on the model provider and functions enabled in `utils.py`.

## Notes

- Some scripts still contain local absolute paths. Before reproducing experiments or publishing the repository, consider replacing them with relative paths or a unified configuration file.
- The code in `knowledge_graph/2wiki/`, `knowledge_graph/hotpot/`, and `knowledge_graph/musique/` is mostly parallel. Reading `knowledge_graph/hotpot/` first is recommended.
- Knowledge graph construction and vector generation may call LLM or embedding APIs. Runtime and cost depend on the dataset size.
- Query scripts use `parallel_llm_processor` for concurrent inference. If your GPU memory or API rate limit is constrained, reduce `max_workers`.
- Experiment outputs are usually saved as CSV files with fields such as `question`, `gold_answer`, `pred_answer`, `strategy`, and `context_id`.

## Recommended Reading Order

1. `knowledge_graph/hotpot/build_kg.ipynb`
2. `knowledge_graph/hotpot/seed.py`
3. `knowledge_graph/adapt/adapt.py`
4. `knowledge_graph/hotpot/query_global.py`
5. `knowledge_graph/hotpot/query_local.py`
6. `knowledge_graph/hotpot/evalue.py`

This order starts from graph construction, then moves to seed anchoring, intent-aware weight modulation, and finally multi-hop question answering and evaluation.
