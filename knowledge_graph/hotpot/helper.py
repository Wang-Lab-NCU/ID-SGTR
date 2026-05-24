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
# Make sure LangChain types are imported from correct location, assuming they are shared with kg_query_engine.py
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
    task_timeout: int = 100 # Forced timeout for a single task
) -> List[Tuple[Any, Any]]:
    """
    Optimized parallel executor:
    1. Displays progress using tqdm.
    2. Solves the issue of last few threads getting stuck due to 'with' statement (by using wait=False).
    """
    print(start_message)
    results_list: List[Tuple[Any, Any]] = []
    
    # Internal retry logic
    def _run_with_retry(i: int, row: pd.Series) -> Tuple[int, Any, Any]:
        identifier = row.get('chunk_id', i)
        for attempt in range(max_retries):
            try:
                # Warning: the processing_func called here should also have requests timeout settings internally
                current_identifier, result = processing_func(i, row)
                return (i, current_identifier, result)
            except Exception as e:
                error_msg = str(e)
                # Simple error classification logging
                if attempt + 1 == max_retries:
                    # Print ERROR only on last attempt to avoid spamming during retries
                    print(f"\n🛑 ID {identifier} final failure: {error_msg[:100]}...")
                    return (i, identifier, None)
                
                # Exponential backoff
                delay = initial_delay * (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(delay)
        return (i, identifier, None)

    # --- Core modification: manually create executor, do not use with context ---
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}

    try:
        # 1. Submit all tasks
        print(f"🚀 Submitting {len(dataframe)} tasks to thread pool...")
        for i, row in dataframe.iterrows():
            future = executor.submit(_run_with_retry, i, row)
            futures[future] = row.get('chunk_id', i)

        # 2. Use tqdm to track progress
        print(f"⏳ Starting parallel processing...")
        # total=len(futures) lets the progress bar know the total count
        with tqdm(total=len(futures), unit="chunk") as pbar:
            for future in as_completed(futures):
                identifier = futures[future]
                try:
                    # Wait for result, set timeout to prevent main thread from being permanently blocked by a single task
                    original_index, current_id, result = future.result(timeout=task_timeout)
                    
                    if result is not None:
                        results_list.append((current_id, result))
                    
                except TimeoutError:
                    print(f"\n⏰ Task ID {identifier} result retrieval timeout (thread may be stuck), skipping.")
                    # No need to append, just skip
                except Exception as e:
                    print(f"\n🔴 Task ID {identifier} raised unknown exception: {e}")
                finally:
                    # Progress bar advances regardless of success or failure
                    pbar.update(1)

    finally:
        # --- Core fix ---
        # wait=False: do not wait for stuck (still running) threads, close the entrance and let the main program continue
        # For Python 3.9+, cancel_futures=True can be added
        print("\n🧹 Cleaning up thread pool (dropping stuck threads)...")
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
    # Optimized regex: non-greedy match from start until a newline or end
    cleaned_string = re.sub(r'^```(?:json)?\s*', '', cleaned_string, flags=re.IGNORECASE)
    cleaned_string = re.sub(r'\s*```$', '', cleaned_string)
    cleaned_string = cleaned_string.strip()

    # 2. Edge cropping: ensure the string starts with [ or { and ends with ] or }
    # This removes any chitchat from the LLM before/after the JSON
    match = re.search(r'(\[|\{).*(\]|\})', cleaned_string, re.DOTALL)
    if match:
        cleaned_string = match.group(0)
    else:
        # If no matching brackets can be found, try to find any possible start (fault tolerance)
        start_match = re.search(r'[\{\[]', cleaned_string)
        if start_match:
            cleaned_string = cleaned_string[start_match.start():]
        else:
            return default

    # 3. Try standard JSON parsing
    try:
        # strict=False allows control characters (e.g., newlines) inside strings
        result = json.loads(cleaned_string, strict=False)
        if isinstance(result, dict):
            return [result]
        return result
    except json.JSONDecodeError:
        pass # Continue to other methods

    # 4. Try Python literal parsing (handles single‑quote JSON)
    # Many LLMs prefer Python dict format {'key': 'value'} instead of {"key": "value"}
    try:
        result = ast.literal_eval(cleaned_string)
        if isinstance(result, (list, dict)):
            if isinstance(result, dict):
                return [result]
            return result
    except (ValueError, SyntaxError):
        pass # Continue

    # 5. Robust fix: missing comma repair
    # Common error: objects missing a comma, e.g. }{ or ][
    try:
        repaired_string = re.sub(r'\}\s*\{', '},{', cleaned_string)
        repaired_string = re.sub(r'\]\s*\[', '],[', repaired_string)
        result = json.loads(repaired_string, strict=False)
        if isinstance(result, dict):
            return [result]
        return result
    except json.JSONDecodeError:
        pass

    # 6. Ultimate fallback: brute‑force extraction of individual objects
    # If the overall structure is broken (e.g., missing closing ]), try to extract complete {...} inside
    print(f"⚠️ Overall parsing failed, attempting to extract objects one by one...")
    final_records: List[Dict] = []
    
    # Regex explanation: non‑greedy match of { ... }
    # Note: if a value contains '}', this simple regex may truncate.
    # For simple graph relations (flat structure), this regex is usually sufficient.
    object_pattern = re.compile(r'\{[^{}]+\}', re.DOTALL) 
    # If you need to handle nested structures (values containing {}), more complex logic is needed, but not required here.
    
    object_strings = object_pattern.findall(cleaned_string)

    for obj_str in object_strings:
        try:
            # Try to parse the individual object
            # Try both json and ast approaches
            try:
                obj = json.loads(obj_str, strict=False)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(obj_str)
                except:
                    continue # Discard this object
            
            # 💥 Key modification: keep any dict, no longer check for specific keys like 'entity'
            if isinstance(obj, dict):
                final_records.append(obj)
                
        except Exception:
            continue
    
    if final_records:
        print(f"✅ Robust extraction successfully recovered {len(final_records)} records.")
        return final_records
    
    # Complete failure
    print(f"❌ Unable to parse JSON. Raw content snippet: {json_string[:100]}...")
    return default
    

def apply_genealogical_penalty(entities: list, dist_matrix: np.ndarray, penalty_value: float = 10.0) -> np.ndarray:
    """
    Apply a 'genealogical conflict penalty' to the distance matrix.
    If two entities have explicitly conflicting generational suffixes (e.g., Sr. vs Jr., II vs III),
    forcefully set their distance to penalty_value (increase distance, prevent clustering).
    
    Args:
        entities: List of entity names (corresponding to rows/columns of dist_matrix)
        dist_matrix: Pre‑computed cosine distance matrix (N x N)
        penalty_value: Penalty value, typically set to 10.0 or larger to exceed any eps
    
    Returns:
        Modified dist_matrix (in‑place modification)
    """
    
    # 1. Define suffix mapping table (map different spellings to a common generational ID)
    # Same ID means same generation (no conflict), different IDs mean explicit conflict
    SUFFIX_MAP = {
        # son / second
        'jr': 1, 'junior': 1, 'ii': 1, '2nd': 1,
        # father / first / senior
        'sr': 2, 'senior': 2, 'i': 2, '1st': 2,
        # third
        'iii': 3, '3rd': 3,
        # fourth
        'iv': 4, '4th': 4,
        # fifth
        'v': 5, '5th': 5
    }

    # 2. Compile regex: match complete words at the end of the string, case‑insensitive, allowing optional dot
    # e.g. match: "Ed Wood Jr", "King George V", "John Smith, 3rd"
    # \b ensures word boundary, \.? allows dot or no dot
    pattern_str = r'\b(' + '|'.join(SUFFIX_MAP.keys()) + r')\.?$'
    regex = re.compile(pattern_str, re.IGNORECASE)

    n = len(entities)
    
    # Pre‑extract generational ID for each entity (None if no suffix)
    generations = []
    for name in entities:
        # Trim extra spaces, convert to lowercase for matching
        clean_name = name.strip().lower()
        match = regex.search(clean_name)
        if match:
            # Extract matched suffix (strip possible dot)
            suffix = match.group(1).replace('.', '')
            gen_id = SUFFIX_MAP.get(suffix)
            generations.append(gen_id)
        else:
            generations.append(None)

    # 3. Iterate through matrix and apply penalty
    # Only treat as conflict when both persons have suffixes and the suffix IDs differ
    # Note: if one has a suffix (Jr) and the other has none (no suffix), it is NOT treated as a conflict (could be an abbreviation)
    for i in range(n):
        gen_i = generations[i]
        if gen_i is None: continue # If i has no suffix, skip
        
        for j in range(i + 1, n):
            gen_j = generations[j]
            
            # Only compare if both have suffixes
            if gen_j is not None:
                if gen_i != gen_j:
                    # 💥 Conflict detected (e.g., Jr vs Sr), apply penalty
                    dist_matrix[i, j] = penalty_value
                    dist_matrix[j, i] = penalty_value
                    
                    # (Optional) print debug info to see who is being separated
                    # print(f"🔨 Force separating: '{entities[i]}' vs '{entities[j]}'")

    return dist_matrix

def clean_entity(entity_name):
    """
    Clean an entity name: remove titles, common suffixes, non‑alphanumeric characters, and all spaces.
    """
    if pd.isna(entity_name):
        return ""
    entity_name = str(entity_name).strip()
    
    # Remove common titles and suffixes
    entity_name = re.sub(r'\s+(Jr|Sr|Dr|Prof|King|Sultan|President|Queen|Princess)\b\.?', '', entity_name, flags=re.IGNORECASE).strip()
    
    # Remove all non‑alphanumeric characters (e.g., commas, quotes)
    entity_name = re.sub(r'[^a-zA-Z0-9\s]', '', entity_name, flags=re.IGNORECASE) 
    
    # Replace multiple spaces with a single space
    entity_name = re.sub(r'\s+', ' ', entity_name).strip()
    
    # =======================================================
    # 🚨 Key fix: remove all spaces for substring matching
    # =======================================================
    cleaned_no_space = entity_name.replace(' ', '')
    
    return cleaned_no_space.lower() # Convert to lowercase for case‑insensitivity

def post_process_person_entities(df_standardization_map):
    """
    Post‑process Person category entities, merging full names / abbreviations based on substring containment.
    
    Args:
        df_standardization_map (pd.DataFrame): Mapping table containing Original_Entity, Standard_Entity, context_id.
        
    Returns:
        pd.DataFrame: Updated mapping table.
    """
    print("\n⏳ Starting post‑processing of 'Person' type full names / abbreviations...")
    
    # Process only Person entities
    person_map = df_standardization_map[df_standardization_map['category'] == 'Person'].copy()
    
    # Store updates to be applied (Original_Entity -> New_Standard_Entity)
    updates = {}
    
    # 1. Add a cleaned entity name column to the current mapping table
    person_map['Clean_Entity'] = person_map['Original_Entity'].apply(clean_entity)
    
    # 2. Iterate over each context_id
    for context_id, group in person_map.groupby('context_id'):
        unique_entities = group['Original_Entity'].unique()
        
        # Map (Entity, Clean_Entity, Current_Standard) for this context
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
                
                # If already merged into the same Standard_Entity by DBSCAN, skip
                if std_a == std_b:
                    continue

                # Heuristic: check substring relation (assuming already cleaned)
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
                    # Goal: unify all involved entities (including their original Standard_Entity) under the new Standard_Entity
                    
                    # Ensure the new Standard Entity is the selected one
                    updates[(context_id, shorter_entity)] = new_standard
                    
                    # Also, point the current Std_A and Std_B (if they are different) to the new Standard
                    # Since we operate at the Original_Entity level, we only update Original_Entity to New_Standard here
                    updates[(context_id, entity_a)] = new_standard
                    updates[(context_id, entity_b)] = new_standard

    # 5. Apply updates to person_map
    for (cid, orig_ent), new_std in updates.items():
        # Find and update within the group
        # Note: need to match exactly context_id and Original_Entity
        mask = (person_map['context_id'] == cid) & (person_map['Original_Entity'] == orig_ent)
        person_map.loc[mask, 'Standard_Entity'] = new_std

    # 6. Merge the updated Person mapping back into the original mapping table
    
    # Remove Person category from original mapping
    df_without_person = df_standardization_map[df_standardization_map['category'] != 'Person']
    
    # Concatenate
    df_updated_map = pd.concat([df_without_person, person_map], ignore_index=True)
    print(f"✅ Post‑processing complete. Updated {len(updates)} mapping relationships.")
    return df_updated_map
