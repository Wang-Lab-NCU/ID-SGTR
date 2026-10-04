import uuid
import ast
import pandas as pd
import numpy as np
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any, Dict, List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from helper import parallel_llm_processor
from zhipu import extractConcepts, graphPrompt, resolve_coreferences
from utils import get_llm_model, get_chat_model


# ======================================================================
# 0. General helper functions (new)
# ======================================================================

def parse_list(x):
    """Parse a string list; return empty list on failure"""
    if isinstance(x, list): return x
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            return val if isinstance(val, list) else []
        except:
            return []
    return []

def build_enriched_name_text(row):
    """
    Build format: "Standard_Entity, synonyms: Synonym A, Synonym B"
    Used to improve accuracy of vector retrieval.
    """
    entity_name = str(row['Standard_Entity']).strip()
    synonyms_list = row['synonyms'] if isinstance(row['synonyms'], list) else []
    
    # Filter out synonyms that are identical to the standard name to avoid redundancy
    valid_syns = [s for s in synonyms_list if s and str(s).strip() != entity_name]
    
    if valid_syns:
        # Truncate overly long synonym lists (first 10)
        syn_str = ", ".join(valid_syns[:10]) 
        return f"{entity_name}, synonyms: {syn_str}"
    
    return entity_name

# ======================================================================
# 1. Basic data processing
# ======================================================================

def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        clean_text = chunk.page_content.replace('\n', ' ').replace('\r', ' ')
        row = {
            "text": clean_text,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ======================================================================
# 2. Entity aggregation logic (new - from merge.py)
# ======================================================================

def merge_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate entities with the same (context_id, Standard_Entity),
    merging descriptions and synonyms.
    """
    print("🔄 [Helper] Aggregating entity data...")
    df = df.copy()
    
    # 1. Parse list columns
    for col in ['synonyms']:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)
        else:
            df[col] = [[] for _ in range(len(df))] # Fill with empty lists
    
    if 'description' not in df.columns:
        df['description'] = ""
    df['description'] = df['description'].fillna("")
    
    # 2. Add the original mention 'Entity' into the synonym candidates
    df['synonyms'] = df.apply(lambda row: row['synonyms'] + [str(row['Entity'])], axis=1)

    # 3. Define aggregation functions
    def merge_descriptions(series):
        unique_descs = set()
        for d in series:
            d_str = str(d).strip()
            if d_str and d_str != '-1':
                unique_descs.add(d_str)
        return ". ".join(sorted(list(unique_descs)))

    def merge_lists(series):
        merged = set()
        for lst in series:
            merged.update(lst)
        return list(merged)

    # 4. Perform GroupBy
    # Ensure Standard_Entity is not empty
    df = df[df['Standard_Entity'].notna() & (df['Standard_Entity'] != "")]
    
    df_grouped = df.groupby(['context_id', 'Standard_Entity', 'category'], as_index=False).agg({
        'chunk_id': 'first', # Take the first occurring chunk_id as representative
        'cluster_id': 'first',
        'category': 'first',
        'description': merge_descriptions,
        'synonyms': merge_lists,
    })
    # [Change] Ensure the aggregated data is still sorted by chunk_id
    df_grouped = df_grouped.sort_values(by=['chunk_id']).reset_index(drop=True)
    
    print(f"✅ [Helper] Aggregation complete. Original rows: {len(df)} -> Aggregated rows: {len(df_grouped)}")
    return df_grouped

# ======================================================================
# 3. LLM task wrappers
# ======================================================================
def ExtractConcepts(dataframe: pd.DataFrame, model=None, max_workers: int = 100) -> pd.DataFrame:
    """
    Extract concept nodes.
    """
    def concept_extractor(i, row):
        text_input = row.get('resolved_text', row['text'])
        # If 429 or JSON error occurs here, it will be caught and retried by parallel_llm_processor
        current_model = get_chat_model(task_type="extraction")
        concepts_list = extractConcepts(text_input, model=current_model)
        return (row['chunk_id'], concepts_list)

    start_msg = f"--- Step: Starting parallel concept extraction on {len(dataframe)} chunks ---"
    successful_results = parallel_llm_processor(
        dataframe=dataframe,
        processing_func=concept_extractor,
        start_message=start_msg,
        max_workers=max_workers
    )

    final_records = []
    id_map = dataframe.set_index('chunk_id')['context_id'].to_dict()

    for chunk_id, concepts_list in successful_results:
        context_id = id_map.get(chunk_id)
        if not concepts_list: continue
        for concept in concepts_list:
            record = {
                'context_id': context_id,
                'chunk_id': chunk_id,
                'Entity': concept.get('entity'),
                'category': concept.get('category'),
                'description': concept.get('description'),
                'synonyms': concept.get('synonyms'),
            }
            final_records.append(record)

    df_concepts = pd.DataFrame(final_records)
    if not df_concepts.empty:
        df_concepts = df_concepts.sort_values(by=['chunk_id']).reset_index(drop=True)
        
    print(f"✅ Concept extraction complete, extracted {len(df_concepts)} entities.")
    return df_concepts


def ResolveCoreferences(dataframe: pd.DataFrame, model=None, max_workers=15) -> pd.DataFrame:
    """
    Coreference resolution.
    """
    def coref_resolver(i, row):
        # resolve_coreferences no longer swallows 429 errors internally
        current_model = get_chat_model(task_type="extraction")
        resolved_text = resolve_coreferences(row.text, model=current_model) 
        return (row['chunk_id'], resolved_text)

    start_msg = f"--- Step: Starting parallel coreference resolution on {len(dataframe)} chunks ---"
    successful_results = parallel_llm_processor(
        dataframe=dataframe,
        processing_func=coref_resolver,
        start_message=start_msg,
        max_workers=max_workers
    )

    results_dict = {res[0]: res[1] for res in successful_results}
    dataframe['resolved_text'] = dataframe['chunk_id'].map(results_dict)
    # If final failure (after 3 retries), fallback to original text
    dataframe['resolved_text'] = dataframe['resolved_text'].fillna(dataframe['text'])
    print(f"✅ Coreference resolution complete.")
    return dataframe

# ======================================================================
# 4. Graph construction and co‑occurrence
# ======================================================================

def df2Graph(dataframe: pd.DataFrame, entity_map: dict, model=None, max_workers=100) -> list:
    """
    Parallel relation extraction.
    """
    def graph_extractor(i, row):
        current_map = entity_map.get((row['context_id'], row['chunk_id']), {})
        if not current_map: return (row['chunk_id'], [])
        metadata = {"context_id": row['context_id'], "chunk_id": row['chunk_id']}
        current_model = get_chat_model(task_type="extraction")
        # Exceptions raised by graphPrompt will be caught by the retry mechanism
        result = graphPrompt(row.resolved_text, current_map, metadata, current_model)
        return (row['chunk_id'], result)

    start_msg = f"--- Step: Starting parallel relation extraction ---"
    successful_results = parallel_llm_processor(
        dataframe=dataframe,
        processing_func=graph_extractor,
        start_message=start_msg,
        max_workers=max_workers
    )

    concept_list = []
    for _, relations_list in successful_results:
        if relations_list: concept_list.extend(relations_list)
    print(f"✅ Relation extraction complete, extracted {len(concept_list)} edges.")
    return concept_list

def graph2Df(nodes_list) -> pd.DataFrame:
    if not nodes_list:
        return pd.DataFrame(columns=["context_id", "node_1", "node_2", "edge", "chunk_id"])
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    required_cols = ["context_id", "node_1", "node_2", "edge", "chunk_id"]
    for col in required_cols:
        if col not in graph_dataframe.columns: graph_dataframe[col] = np.nan
    graph_dataframe = graph_dataframe.filter(items=required_cols, axis=1)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    # [Change] Ensure edge table is sorted by chunk_id
    if 'chunk_id' in graph_dataframe.columns:
        graph_dataframe['chunk_id'] = pd.to_numeric(graph_dataframe['chunk_id'], errors='coerce')
        graph_dataframe = graph_dataframe.sort_values(by=['chunk_id']).reset_index(drop=True)
    return graph_dataframe


def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["context_id", "node_1", "node_2", "chunk_id", "count", "edge"])

    def unique_chunks_str(x):
        return sorted(list(set(str(i) for i in x)))

    dfg_long = pd.melt(df, id_vars=["chunk_id", "context_id"], value_vars=["node_1", "node_2"], value_name="node")
    dfg_long.drop(columns=["variable"], inplace=True)
    dfg_long["chunk_id"] = dfg_long["chunk_id"].astype(str)

    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    dfg_wide = dfg_wide[dfg_wide["node_1"] != dfg_wide["node_2"]]

    dfg2 = (
        # ✅ Add context_id_1 to the grouping key to cut off cross‑document merging
        dfg_wide.groupby(["context_id_1", "node_1", "node_2"])
        .agg({
            "chunk_id": [lambda x: ",".join(unique_chunks_str(x)), "count"]
        })
        .reset_index()
    )
    
    # ✅ Fix: rename columns according to the exact order generated by reset_index()
    dfg2.columns = ["context_id", "node_1", "node_2", "chunk_id", "count"]
    
    dfg2["context_id"] = dfg2["context_id"].astype(str)
    
    # Now the "count" column is indeed numeric, comparisons will no longer fail
    dfg2 = dfg2[dfg2["count"] > 2]
    dfg2["edge"] = "contextual_proximity"
    
    return dfg2[["context_id", "node_1", "node_2", "chunk_id", "count", "edge"]]
