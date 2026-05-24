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
from langchain_core.messages import BaseMessage 


# ==========================================
# Parallel executor
# ==========================================
def parallel_llm_processor(
    dataframe: pd.DataFrame,
    processing_func: Callable[[int, pd.Series], Tuple[Any, Any]],
    start_message: str,
    max_workers: int = 15,
    max_retries: int = 8,
    initial_delay: int = 2,
    task_timeout: int = 120 # The mandatory timeout time for a single task
) -> List[Tuple[Any, Any]]:
    """
    Use tqdm to display the progress.
    """
    print(start_message)
    results_list: List[Tuple[Any, Any]] = []
    
    # Internal retry logic
    def _run_with_retry(i: int, row: pd.Series) -> Tuple[int, Any, Any]:
        identifier = row.get('chunk_id', i)
        for attempt in range(max_retries):
            try:
                # Warning: The processing_func called here should also have a timeout setting for requests
                current_identifier, result = processing_func(i, row)
                return (i, current_identifier, result)
            except Exception as e:
                error_msg = str(e)
                # Simple error classification log
                if attempt + 1 == max_retries:
                    # Only print ERROR on the final attempt to avoid spamming the console with retries
                    print(f"\n🛑 ID {identifier}最终失败: {error_msg[:100]}...")
                    return (i, identifier, None)

                # Exponential backoff
                delay = initial_delay * (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(delay)
        return (i, identifier, None)

    # ---Manually create the executor, without using the with context ---
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}

    try:
        # 1. Submit all tasks
        print(f"🚀 Submitting {len(dataframe)} tasks to the thread pool")
        for i, row in dataframe.iterrows():
            future = executor.submit(_run_with_retry, i, row)
            futures[future] = row.get('chunk_id', i)

        # 2. Use tqdm to display progress
        print(f"⏳ Starting parallel processing...")
        # total=len(futures) 让进度条知道总数
        with tqdm(total=len(futures), unit="chunk") as pbar:
            for future in as_completed(futures):
                identifier = futures[future]
                try:
                    # Wait for the result, setting a timeout to prevent the main thread from being permanently blocked by a single task
                    original_index, current_id, result = future.result(timeout=task_timeout)
                    
                    if result is not None:
                        results_list.append((current_id, result))
                    
                except TimeoutError:
                    print(f"\n⏰ Task ID {identifier}: Retrieval result timed out (the thread might be stuck), skipping.")
                except Exception as e:
                    print(f"\n🔴 Task ID {identifier} threw an unknown exception: {e}")
                finally:
                    pbar.update(1)

    finally:
        # When wait=False: It means not to wait for the threads that are still running (stuck), directly close the entry point, and let the main program continue to execute further. 
        # In Python 3.9 and above, you can also add cancel_futures=True.
        print("\n🧹 Cleaning up the thread pool (discarding the stuck threads) ...")
        executor.shutdown(wait=False)
        print(f"✅ Processing completed. Successfully retrieved results: {len(results_list)}/{len(dataframe)}")

    return results_list

def get_unique_chunks(chunk_ids):
    if pd.isna(chunk_ids) or not chunk_ids:
        return set()
    return {c.strip() for c in str(chunk_ids).split(',') if c.strip().isdigit()}


def safe_json_parse(json_string: str, default: List = None) -> Any:
    """
     Attempt to safely parse the JSON string, and address the parsing failures caused by invisible characters, truncation, missing delimiters, or single quotes.
    """
    if default is None:
        default = []
    
    if not json_string or not isinstance(json_string, str):
        return default

    # 1. Ultimate cleanup: Remove BOM, invisible characters, and strip Markdown
    cleaned_string = re.sub(r'[\ufeff\u200b\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', json_string).strip()
    
    # Remove Markdown code blocks (```json ... ```)
    # Optimize regex: Non-greedy matching from the start until a newline or end is found
    cleaned_string = re.sub(r'^```(?:json)?\s*', '', cleaned_string, flags=re.IGNORECASE)
    cleaned_string = re.sub(r'\s*```$', '', cleaned_string)
    cleaned_string = cleaned_string.strip()

    # 2. Boundary trimming: Ensure the string starts with [ or { and ends with ] or }
    # This step is to remove any extraneous text before or after the JSON
    match = re.search(r'(\[|\{).*(\]|\})', cleaned_string, re.DOTALL)
    if match:
        cleaned_string = match.group(0)
    else:
        # If no matching brackets are found, try to find any possible starting point (fault tolerance)
        start_match = re.search(r'[\{\[]', cleaned_string)
        if start_match:
            cleaned_string = cleaned_string[start_match.start():]
        else:
            return default

    # 3. Attempt Standard Parsing (JSON)
    try:
        # strict=False allows control characters (such as newline characters) to be present in the string
        result = json.loads(cleaned_string, strict=False)
        if isinstance(result, dict):
            return [result]
        return result
    except json.JSONDecodeError:
        pass 

    # 4. Attempt Python literal parsing (handling single-quoted JSON)
    # Many LLMs prefer to output Python dict format {'key': 'value'} instead of {"key": "value"}
    try:
        result = ast.literal_eval(cleaned_string)
        if isinstance(result, (list, dict)):
            if isinstance(result, dict):
                return [result]
            return result
    except (ValueError, SyntaxError):
        pass 

    # 5. Robust Fix: Missing Comma Correction
    # Common Error: No commas between objects, such as }{ or ] [
    try:
        repaired_string = re.sub(r'\}\s*\{', '},{', cleaned_string)
        repaired_string = re.sub(r'\]\s*\[', '],[', repaired_string)
        result = json.loads(repaired_string, strict=False)
        if isinstance(result, dict):
            return [result]
        return result
    except json.JSONDecodeError:
        pass

    # 6. Ultimate Backup: Regular Expression for Brutal Extraction
    # If the overall structure is flawed (for example, lacking a closing ']'), try extracting the complete {...} within. }
    print(f"⚠️ Overall parsing failed. Attempting to extract objects one by one...")
    final_records: List[Dict] = []
    
    # Regular Expression Explanation: Non-greedy Matching { ... } }
    # Note: If the value contains '}', this simple regular expression might truncate it.
    # For simple graph relationships (flat structure), this regular expression is usually sufficient.
    object_pattern = re.compile(r'\{[^{}]+\}', re.DOTALL) 
    # If you need to handle nested structures (values contain {}), you would need more complex logic, but that's not required for the current scenario.
    
    object_strings = object_pattern.findall(cleaned_string)

    for obj_str in object_strings:
        try:
            try:
                obj = json.loads(obj_str, strict=False)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(obj_str)
                except:
                    continue # Give up this object
            
            # 💥 Key Modification: Keep any dictionary, no longer check for specific key like 'entity'
            if isinstance(obj, dict):
                final_records.append(obj)
                
        except Exception:
            continue
    
    if final_records:
        print(f"✅ Robust extraction successfully retrieved {len(final_records)} records.")
        return final_records
    
    print(f"❌ Unable to parse JSON. Original content snippet: {json_string[:100]}...")
    return default
    

def apply_genealogical_penalty(entities: list, dist_matrix: np.ndarray, penalty_value: float = 10.0) -> np.ndarray:
    """
    Apply the "phylogenetic conflict penalty" to the distance matrix.
    If it is detected that two entities have an explicitly conflicting generational suffix (such as Sr. vs Jr., II vs III),
    then force their distance to be set to penalty_value (increase the distance to prevent clustering).
    
    Args:
        entities: List of entity names (corresponding to the rows and columns of dist_matrix)
        dist_matrix: Precomputed cosine distance matrix (N x N)
        penalty_value: Penalty value, usually set to 10.0 or larger to ensure it exceeds any eps 
    Returns:
        Modified dist_matrix (In-place modification)
    """
    # 1. Define the suffix mapping table (mapping different notations to a unified generation ID)
    # If the IDs are the same, it indicates the same generation (no conflict); if the IDs are different, it indicates an explicit conflict
    SUFFIX_MAP = {
        # Son / Second Generation
        'jr': 1, 'junior': 1, 'ii': 1, '2nd': 1,
        # Father / First Generation
        'sr': 2, 'senior': 2, 'i': 2, '1st': 2,
        # Third Generation
        'iii': 3, '3rd': 3,
        # Fourth Generation
        'iv': 4, '4th': 4,
        # Fifth Generation
        'v': 5, '5th': 5
    }

    # 2. Compile the regex: match complete words at the end of the string, ignoring case and periods
    # For example, match: "Ed Wood Jr", "King George V", "John Smith, 3rd"
    # \b ensures word boundaries, \.? allows for an optional period
    pattern_str = r'\b(' + '|'.join(SUFFIX_MAP.keys()) + r')\.?$'
    regex = re.compile(pattern_str, re.IGNORECASE)

    n = len(entities)
    
    # Pre-extract the generation IDs for each entity (None if no suffix)
    generations = []
    for name in entities:
        # Clean extra spaces and convert to lowercase for matching
        clean_name = name.strip().lower()
        match = regex.search(clean_name)
        if match:
            # Extract the matched suffix (remove any potential periods)
            suffix = match.group(1).replace('.', '')
            gen_id = SUFFIX_MAP.get(suffix)
            generations.append(gen_id)
        else:
            generations.append(None)

    # 3. Apply penalty by traversing the matrix
    # Only when both individuals have suffixes and their suffix IDs are different, it is considered a conflict
    # Note: If one person has a suffix (e.g., Jr) and the other person has no suffix (no), it is usually **not** regarded as a conflict (possibly a shortened form)
    for i in range(n):
        gen_i = generations[i]
        if gen_i is None: continue # If i has no suffix, skip
        
        for j in range(i + 1, n):
            gen_j = generations[j]
            
            # Only compare if both have suffixes
            if gen_j is not None:
                if gen_i != gen_j:
                    # 💥 When a conflict is identified (such as Jr vs Sr), apply the penalty* as the conflict (it might be abbreviated as "conflict")
                    dist_matrix[i, j] = penalty_value
                    dist_matrix[j, i] = penalty_value
                    
                    #(Optional) Print debugging information to facilitate identifying which users were forcibly separated.
                    # print(f"🔨 Forced separation: '{entities[i]}' vs '{entities[j]}'")

    return dist_matrix

def clean_entity(entity_name):
    """
    Remove the entity names, eliminate titles, common suffixes, non-alphanumeric characters and all spaces.
    """
    if pd.isna(entity_name):
        return ""
    entity_name = str(entity_name).strip()
    
    # Remove common titles and suffixes
    entity_name = re.sub(r'\s+(Jr|Sr|Dr|Prof|King|Sultan|President|Queen|Princess)\b\.?', '', entity_name, flags=re.IGNORECASE).strip()
    
    # Remove all non-alphanumeric characters (such as commas, quotes, etc.)
    entity_name = re.sub(r'[^a-zA-Z0-9\s]', '', entity_name, flags=re.IGNORECASE) 
    
    # Replace multiple spaces with a single space
    entity_name = re.sub(r'\s+', ' ', entity_name).strip()
    
    #  Remove all spaces and use for substring matching
    cleaned_no_space = entity_name.replace(' ', '')
    
    return cleaned_no_space.lower() #Convert to lowercase uniformly and ensure that case sensitivity is not considered.
def post_process_person_entities(df_standardization_map):
    """
    Perform post-processing on the "Person" category and merge the full name/synonym based on the string containment relationship.
    
    Args:
        df_standardization_map (pd.DataFrame): INCLUDE  Original_Entity, Standard_Entity, context_id mapping table.
        
    Returns:
        pd.DataFrame: The updated mapping table.
    """
    print("\n⏳ Starting post-processing for 'Person' type full names/synonyms...")
    
    # Only handle entities of the Person type
    person_map = df_standardization_map[df_standardization_map['category'] == 'Person'].copy()
    
    # Store the mappings that need to be updated (Original_Entity -> New_Standard_Entity)
    updates = {}
    
    # 1. Add a clean entity name column to the current mapping table
    person_map['Clean_Entity'] = person_map['Original_Entity'].apply(clean_entity)
    
    # 2. Iterate over each context_id
    for context_id, group in person_map.groupby('context_id'):
        unique_entities = group['Original_Entity'].unique()
        
        # Convert the current context's (Entity, Clean_Entity, Current_Standard) mappings to a dictionary
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
                
                # Check all entities. If they have been merged into the same Standard_Entity, then skip them.
                if std_a == std_b:
                    continue

                # Heuristic rule: Check substring relationship (assuming already cleaned)
                is_sub_a = clean_a in clean_b and clean_a != clean_b
                is_sub_b = clean_b in clean_a and clean_b != clean_a

                if is_sub_a or is_sub_b:
                    
                    # Identify longer entities as the new Standard_Entity
                    if len(entity_a) > len(entity_b):
                        new_standard = entity_a
                        shorter_entity = entity_b
                    else:
                        new_standard = entity_b
                        shorter_entity = entity_a
                        
                    # 4. Update mapping
                    # Target: Unify all involved entities (including their original Standard_Entity) to the new Standard_Entity
                    
                    # Ensure the new Standard Entity is the selected one
                    updates[(context_id, shorter_entity)] = new_standard
                    
                    # Additionally, we need to point the entire cluster represented by the current Std_A and Std_B (if they are different) to the new Standard
                    updates[(context_id, entity_a)] = new_standard
                    updates[(context_id, entity_b)] = new_standard

    # 5.  Apply the update to person_map
    for (cid, orig_ent), new_std in updates.items():
        mask = (person_map['context_id'] == cid) & (person_map['Original_Entity'] == orig_ent)
        person_map.loc[mask, 'Standard_Entity'] = new_std

    # 6. Merge the updated Person mapping back into the original mapping table
    
    # Remove the "Person" category from the original mapping
    df_without_person = df_standardization_map[df_standardization_map['category'] != 'Person']
    
    #merge
    df_updated_map = pd.concat([df_without_person, person_map], ignore_index=True)
    print(f"✅ Post-processing is complete. A total of {len(updates)} mapping relationships have been updated.")
    return df_updated_map
