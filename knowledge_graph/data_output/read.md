# data_output

This directory stores intermediate files, graph files, vector files, and QA files generated during knowledge graph construction. These files can be large, so the complete generated data is not included in the repository by default.

## How to Get the Data

There are two ways to obtain the data for this directory.

1. Generate the data with `build_kg.ipynb`

   The project provides three dataset-specific knowledge graph construction notebooks. You can run any one of them depending on the dataset you need:

   ```text
   knowledge_graph/hotpot/build_kg.ipynb
   knowledge_graph/2wiki/build_kg.ipynb
   knowledge_graph/musique/build_kg.ipynb
   ```

   Their default output directories are:

   ```text
   knowledge_graph/data_output/dataset/hotpot/ds1000_2/
   knowledge_graph/data_output/dataset/2wiki/ds1000/
   knowledge_graph/data_output/dataset/musique/ds1000/
   ```

   To write the generated files to another location, modify `OUTPUT_DIR` at the beginning of the corresponding notebook.

2. Download the preprocessed data directly

   You can also download the generated data from Google Drive:

   <https://drive.google.com/drive/folders/1IlzrBSbQo0Rs1FyzG72vG8ERcb_7bcrF?usp=sharing>

   After downloading, keep the directory structure unchanged and place the files under the `data_output/dataset/...` path expected by the code.

## Main Files

Each dataset output directory usually contains the following files:

| File | Description |
| --- | --- |
| `qa.csv` | Questions, answers, and context IDs |
| `chunk.csv` | Original text chunks |
| `resolved_chunks.csv` | Text chunks after coreference resolution |
| `extracted_concepts.csv` | Extracted entities or concepts |
| `dp_extracted_concepts.csv` | Standardized entities or concepts |
| `graph.csv` | Explicit entity relation edges |
| `contextual_proximity.csv` | Contextual proximity edges built from co-occurrence |
| `chunks_with_embeddings.parquet` | Text chunks with dense vector representations |
| `concepts_merged_with_vectors.parquet` | Merged entity or concept vectors |

## Notes

- Knowledge graph construction may call LLM and embedding APIs. Runtime and cost depend on the dataset size.
- Before running query scripts, make sure `DATA_ROOT` points to the actual data output directory.
- If you change `MAX_CONTEXTS`, `START_INDEX`, `END_INDEX`, or `OUTPUT_DIR` in a notebook, it is recommended to also change the output directory name to avoid overwriting existing results.
