import time
import random
import re
from tqdm import tqdm
import ast
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from difflib import SequenceMatcher
from typing import Callable, Any, Dict, List, Tuple
# Ensure LangChain related types are imported from the correct location, assumed shared with kg_query_engine.py
from langchain_core.messages import BaseMessage 


# ==========================================
# Parallel Executor
# ==========================================
def parallel_llm_processor(
    dataframe: pd.DataFrame,
    processing_func: Callable[[int, pd.Series], Tuple[Any, Any]],
    start_message: str,
    max_workers: int = 15,
    max_retries: int = 6,
    initial_delay: int = 2,
    task_timeout: int = 100  # Force timeout for a single task
) -> List[Tuple[Any, Any]]:
    """
    Optimized parallel executor:
    1. Uses tqdm to show progress.
    2. Resolves the issue of stuck threads at the end caused by 'with' statement (via wait=False).
    """
    print(start_message)
    results_list: List[Tuple[Any, Any]] = []
    
    # Internal retry logic
    def _run_with_retry(i: int, row: pd.Series) -> Tuple[int, Any, Any]:
        identifier = row.get('chunk_id', i)
        for attempt in range(max_retries):
            try:
                # Warning: the called processing_func should also have request timeout set internally
                current_identifier, result = processing_func(i, row)
                return (i, current_identifier, result)
            except Exception as e:
                error_msg = str(e)
                # Simple error classification logging
                if attempt + 1 == max_retries:
                    # Only print ERROR on the last attempt to avoid spamming during retries
                    print(f"\n🛑 ID {identifier} final failure: {error_msg[:100]}...")
                    return (i, identifier, None)
                
                # Exponential backoff
                delay = initial_delay * (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(delay)
        return (i, identifier, None)

    # --- Core modification: manually create executor, do not use 'with' context ---
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}

    try:
        # 1. Submit all tasks
        print(f"🚀 Submitting {len(dataframe)} tasks to thread pool...")
        for i, row in dataframe.iterrows():
            future = executor.submit(_run_with_retry, i, row)
            futures[future] = row.get('chunk_id', i)

        # 2. Process progress using tqdm
        print(f"⏳ Starting parallel processing...")
        # total=len(futures) tells the progress bar the total count
        with tqdm(total=len(futures), unit="chunk") as pbar:
            for future in as_completed(futures):
                identifier = futures[future]
                try:
                    # Wait for result, set timeout to prevent the main thread from being blocked forever by a single task
                    original_index, current_id, result = future.result(timeout=task_timeout)
                    
                    if result is not None:
                        results_list.append((current_id, result))
                    
                except TimeoutError:
                    print(f"\n⏰ Task ID {identifier} result retrieval timed out (thread may be stuck), skipping.")
                    # Do not append, just skip
                except Exception as e:
                    print(f"\n🔴 Task ID {identifier} raised unknown exception: {e}")
                finally:
                    # Progress bar advances regardless of success or failure
                    pbar.update(1)

    finally:
        # --- Core fix ---
        # wait=False means: do not wait for those still running (stuck) threads, just shut down the entry point,
        # the main program continues execution.
        # For Python 3.9+ can add cancel_futures=True
        print("\n🧹 Cleaning up thread pool (discarding stuck threads)...")
        executor.shutdown(wait=False)
        print(f"✅ Processing finished. Successfully obtained results: {len(results_list)}/{len(dataframe)}")

    return results_list

def get_unique_chunks(chunk_ids):
    if pd.isna(chunk_ids) or not chunk_ids:
        return set()
    return {c.strip() for c in str(chunk_ids).split(',') if c.strip().isdigit()}


def safe_json_parse(json_string: str, default: List = None) -> Any:
    """
    Attempt to safely parse a JSON string, handling failures caused by invisible characters,
    truncation, missing separators, or single quotes.
    """
    if default is None:
        default = []
    
    if not json_string or not isinstance(json_string, str):
        return default

    # 1. Ultimate cleaning: remove BOM, invisible characters, strip Markdown
    cleaned_string = re.sub(r'[\ufeff\u200b\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', json_string).strip()
    
    # Remove Markdown code blocks (```json ... ```)
    # Optimized regex: non-greedy match from start until newline or end
    cleaned_string = re.sub(r'^```(?:json)?\s*', '', cleaned_string, flags=re.IGNORECASE)
    cleaned_string = re.sub(r'\s*```$', '', cleaned_string)
    cleaned_string = cleaned_string.strip()

    # 2. Edge trimming: ensure string starts with [ or { and ends with ] or }
    # This step removes chatter from LLM before or after the JSON
    match = re.search(r'(\[|\{).*(\]|\})', cleaned_string, re.DOTALL)
    if match:
        cleaned_string = match.group(0)
    else:
        # If matching brackets not found, try to find any possible starting point (fault tolerance)
        start_match = re.search(r'[\{\[]', cleaned_string)
        if start_match:
            cleaned_string = cleaned_string[start_match.start():]
        else:
            return default

    # 3. Attempt standard JSON parsing
    try:
        # strict=False allows control characters (like newlines) in strings
        result = json.loads(cleaned_string, strict=False)
        if isinstance(result, dict):
            return [result]
        return result
    except json.JSONDecodeError:
        pass  # Continue to other methods

    # 4. Attempt Python literal parsing (handles single-quote JSON)
    # Many LLMs like to output Python dict format {'key': 'value'} instead of {"key": "value"}
    try:
        result = ast.literal_eval(cleaned_string)
        if isinstance(result, (list, dict)):
            if isinstance(result, dict):
                return [result]
            return result
    except (ValueError, SyntaxError):
        pass  # Continue

    # 5. Robust repair: missing comma repair
    # Common error: no comma between objects, e.g., }{ or ] [
    try:
        repaired_string = re.sub(r'\}\s*\{', '},{', cleaned_string)
        repaired_string = re.sub(r'\]\s*\[', '],[', repaired_string)
        result = json.loads(repaired_string, strict=False)
        if isinstance(result, dict):
            return [result]
        return result
    except json.JSONDecodeError:
        pass

    # 6. Ultimate fallback: regex extraction of individual objects
    # If the overall structure is broken (e.g., missing closing ]), try to extract complete {...} inside
    print(f"⚠️ Overall parsing failed, attempting to extract objects one by one...")
    final_records: List[Dict] = []
    
    # Regex explanation: non-greedy match { ... }
    # Note: if a value contains }, this simple regex might truncate.
    # For simple graph relations (flat structure), this regex is usually sufficient.
    object_pattern = re.compile(r'\{[^{}]+\}', re.DOTALL) 
    # If you need to handle nested structures (values with {}), more complex logic is needed, but not for the current scenario.
    
    object_strings = object_pattern.findall(cleaned_string)

    for obj_str in object_strings:
        try:
            # Attempt to parse a single object
            # Also try both json and ast methods
            try:
                obj = json.loads(obj_str, strict=False)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(obj_str)
                except:
                    continue  # Abandon this object
            
            # 💥 Key modification: keep any dictionary, no longer check for specific keys like 'entity'
            if isinstance(obj, dict):
                final_records.append(obj)
                
        except Exception:
            continue
    
    if final_records:
        print(f"✅ Robust extraction successfully recovered {len(final_records)} records.")
        return final_records
    
    # Complete failure
    print(f"❌ Failed to parse JSON. Original content snippet: {json_string[:100]}...")
    return default
    

def apply_genealogical_penalty(entities: list, dist_matrix: np.ndarray, penalty_value: float = 10.0) -> np.ndarray:
    """
    Apply 'genealogical conflict penalty' to the distance matrix.
    If two entities have explicitly conflicting generational suffixes (e.g., Sr. vs Jr., II vs III),
    force their distance to be penalty_value (increase distance, prevent clustering).
    
    Args:
        entities: List of entity names (corresponding to rows/columns of dist_matrix)
        dist_matrix: Precomputed cosine distance matrix (N x N)
        penalty_value: Penalty value, usually set to 10.0 or larger to exceed any eps
    
    Returns:
        Modified dist_matrix (in-place modification)
    """
    
    # 1. Define suffix mapping table (map different spellings to a generation ID)
    # Same ID means same generation (no conflict), different IDs mean explicit conflict
    SUFFIX_MAP = {
        # Son / Second
        'jr': 1, 'junior': 1, 'ii': 1, '2nd': 1,
        # Father / First / Senior
        'sr': 2, 'senior': 2, 'i': 2, '1st': 2,
        # Third
        'iii': 3, '3rd': 3,
        # Fourth
        'iv': 4, '4th': 4,
        # Fifth
        'v': 5, '5th': 5
    }

    # 2. Compile regex: match complete word at the end of string, case-insensitive, allow dot
    # e.g., match: "Ed Wood Jr", "King George V", "John Smith, 3rd"
    # \b ensures word boundary, \.? allows optional dot
    pattern_str = r'\b(' + '|'.join(SUFFIX_MAP.keys()) + r')\.?$'
    regex = re.compile(pattern_str, re.IGNORECASE)

    n = len(entities)
    
    # Pre-extract generation ID for each entity (None if no suffix)
    generations = []
    for name in entities:
        # Strip extra spaces, convert to lower for matching
        clean_name = name.strip().lower()
        match = regex.search(clean_name)
        if match:
            # Extract matched suffix (remove possible dot)
            suffix = match.group(1).replace('.', '')
            gen_id = SUFFIX_MAP.get(suffix)
            generations.append(gen_id)
        else:
            generations.append(None)

    # 3. Iterate over matrix and apply penalty
    # Only consider conflict when both have suffixes and their generation IDs differ
    # Note: one has suffix (Jr) and the other has no suffix usually is NOT considered a conflict (might be abbreviation)
    for i in range(n):
        gen_i = generations[i]
        if gen_i is None: continue  # skip if i has no suffix
        
        for j in range(i + 1, n):
            gen_j = generations[j]
            
            # Only compare if both have suffixes
            if gen_j is not None:
                if gen_i != gen_j:
                    # 💥 Conflict detected (e.g., Jr vs Sr), apply penalty
                    dist_matrix[i, j] = penalty_value
                    dist_matrix[j, i] = penalty_value
                    
                    # (Optional) print debug info to see who was forced apart
                    # print(f"🔨 Forced separation: '{entities[i]}' vs '{entities[j]}'")

    return dist_matrix

def clean_entity(entity_name):
    """
    Clean entity name: remove titles, common suffixes, non-alphanumeric characters, and all spaces.
    """
    if pd.isna(entity_name):
        return ""
    entity_name = str(entity_name).strip()
    
    # Remove common titles and suffixes
    entity_name = re.sub(r'\s+(Jr|Sr|Dr|Prof|King|Sultan|President|Queen|Princess)\b\.?', '', entity_name, flags=re.IGNORECASE).strip()
    
    # Remove all non-letter and non-space characters (e.g., commas, quotes)
    entity_name = re.sub(r'[^a-zA-Z0-9\s]', '', entity_name, flags=re.IGNORECASE) 
    
    # Replace multiple spaces with a single space
    entity_name = re.sub(r'\s+', ' ', entity_name).strip()
    
    # =======================================================
    # 🚨 Key fix: remove all spaces for substring matching
    # =======================================================
    cleaned_no_space = entity_name.replace(' ', '')
    
    return cleaned_no_space.lower()  # Convert to lower for case-insensitive comparison

def post_process_person_entities(df_standardization_map):
    """
    Post-process entities of category 'Person', merging full names and abbreviations based on substring containment.
    
    Args:
        df_standardization_map (pd.DataFrame): Mapping table containing Original_Entity, Standard_Entity, context_id.
        
    Returns:
        pd.DataFrame: Updated mapping table.
    """
    print("\n⏳ Starting post-processing of 'Person' type full names/abbreviations...")
    
    # Only process Person type entities
    person_map = df_standardization_map[df_standardization_map['category'] == 'Person'].copy()
    
    # Store updates to be applied (Original_Entity -> New_Standard_Entity)
    updates = {}
    
    # 1. Add a cleaned entity name column to the current mapping table
    person_map['Clean_Entity'] = person_map['Original_Entity'].apply(clean_entity)
    
    # 2. Iterate over each context_id
    for context_id, group in person_map.groupby('context_id'):
        unique_entities = group['Original_Entity'].unique()
        
        # Build a dictionary mapping (Entity -> (Clean_Entity, Current_Standard)) for this context
        entity_info = group.set_index('Original_Entity')[['Clean_Entity', 'Standard_Entity']].to_dict('index')
        
        # 3. Check all entity pairs
        for i in range(len(unique_entities)):
            for j in range(i + 1, len(unique_entities)):
                entity_a = unique_entities[i]
                entity_b = unique_entities[j]

                clean_a = entity_info[entity_a]['Clean_Entity']
                clean_b = entity_info[entity_b]['Clean_Entity']
                
                std_a = entity_info[entity_a]['Standard_Entity']
                std_b = entity_info[entity_b]['Standard_Entity']
                
                # If they are already merged into the same Standard_Entity by DBSCAN, skip
                if std_a == std_b:
                    continue

                # Heuristic rule: check substring relationship (assumes already cleaned)
                is_sub_a = clean_a in clean_b and clean_a != clean_b
                is_sub_b = clean_b in clean_a and clean_b != clean_a

                if is_sub_a or is_sub_b:
                    
                    # Determine the longer entity as the new Standard_Entity
                    if len(entity_a) > len(entity_b):
                        new_standard = entity_a
                        shorter_entity = entity_b
                    else:
                        new_standard = entity_b
                        shorter_entity = entity_a
                        
                    # 4. Update mapping
                    # Goal: unify all involved entities (including their original Standard_Entity) to the new Standard_Entity
                    
                    # Ensure the new Standard Entity is the chosen one
                    updates[(context_id, shorter_entity)] = new_standard
                    
                    # Additionally, point the whole clusters represented by current std_a and std_b (if different) to the new Standard
                    # Since we only operate at Original_Entity level, we only update Original_Entity -> New_Standard
                    updates[(context_id, entity_a)] = new_standard
                    updates[(context_id, entity_b)] = new_standard

    # 5. Apply updates to person_map
    for (cid, orig_ent), new_std in updates.items():
        # Find and update at group level
        # Note: need exact match on context_id and Original_Entity
        mask = (person_map['context_id'] == cid) & (person_map['Original_Entity'] == orig_ent)
        person_map.loc[mask, 'Standard_Entity'] = new_std

    # 6. Merge the updated Person mapping back into the original mapping table
    
    # Remove Person category from the original mapping
    df_without_person = df_standardization_map[df_standardization_map['category'] != 'Person']
    
    # Concatenate
    df_updated_map = pd.concat([df_without_person, person_map], ignore_index=True)
    print(f"✅ Post-processing complete. Updated {len(updates)} mapping relationships.")
    return df_updated_map
