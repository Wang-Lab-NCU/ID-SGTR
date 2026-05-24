import os
import json
import pandas as pd
import numpy as np
import re
import ast
import sys
from pathlib import Path
from tqdm import tqdm
# from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)
from helper import apply_genealogical_penalty, post_process_person_entities
from df_helpers import merge_concepts, build_enriched_name_text
from utils import get_embeddings_model 

class KGPipeline:
    def __init__(self, output_dir, model_dir="../models", embedding_dim=1024):
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.embedding_dim = embedding_dim
        self.os_makedirs()
        
    def os_makedirs(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

    def _get_tokenizer(self, model_name="bert-base-uncased"):
        local_path = os.path.join(self.model_dir, model_name)
        try:
            return AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        except:
            print(f"Downloading tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(local_path)
            return tokenizer

    # =================================================================
    # 1. Core Embedding Logic (private methods)
    # =================================================================
    def _batch_embed(self, texts: list, batch_size=64, desc="Embedding"):
        """Internal batch embedding method"""
        model = get_embeddings_model(dimensions=self.embedding_dim)
        # Clean empty texts
        texts = [str(t).strip() if str(t).strip() else " " for t in texts]
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[i : i + batch_size]
            try:
                batch_emb = model.embed_documents(batch)
                all_embeddings.extend(batch_emb)
            except Exception as e:
                print(f"❌ Batch embedding failed: {e}")
                # Fill with None or zero vectors; we choose None and handle later
                all_embeddings.extend([None] * len(batch))
        return [np.array(e) if e is not None else np.zeros(self.embedding_dim) for e in all_embeddings]

    def compute_embeddings(self, df: pd.DataFrame, text_col: str, id_col: str, file_name: str):
        """General embedding computation and saving a single file"""
        cache_file = self.output_dir / file_name
        emb_col_name = 'entity_embedding' if 'entity' in file_name.lower() else 'embedding'

        if cache_file.exists():
            print(f"🔍 [Pipeline] Loading embedding cache: {cache_file}")
            try:
                df_cached = pd.read_parquet(cache_file)
                if emb_col_name in df_cached.columns:
                     df_cached[emb_col_name] = df_cached[emb_col_name].apply(lambda x: np.array(x) if isinstance(x, list) else x)
                return df_cached
            except Exception:
                pass

        print(f"⏳ [Pipeline] Computing embeddings for {len(df)} items...")
        embeddings = self._batch_embed(df[text_col].tolist(), desc="Computing Embeddings")
        df[emb_col_name] = pd.Series(embeddings, index=df.index)
        
        # Sort before saving
        if 'chunk_id' in df.columns:
            df = df.sort_values(by=['chunk_id'])
            
        # Save
        df_to_save = df.copy()
        df_to_save[emb_col_name] = df_to_save[emb_col_name].apply(lambda x: x.tolist())
        df_to_save.to_parquet(cache_file, index=False)
        print(f"✅ [Pipeline] Saved to {cache_file}")
        return df

    # =================================================================
    # 2. Entity Aggregation and Dual Embedding (Merge Logic)
    # =================================================================
    def merge_and_embed_concepts(self, df_concepts: pd.DataFrame = None):
        """
        Aggregate entities -> generate merge_entity.csv
        Compute vectors -> generate concepts_merged_with_vectors.parquet (contains vec_entity and vec_desc)
        """
        cache_file = self.output_dir / "concepts_merged_with_vectors.parquet"
        merge_csv_file = self.output_dir / "merge_entity.csv"
        
        if cache_file.exists():
            print(f"🔍 [Pipeline] Loading aggregated entity cache: {cache_file}")
            df_merged = pd.read_parquet(cache_file)
            # Restore vector format
            for col in ['vec_entity', 'vec_desc']:
                if col in df_merged.columns:
                    df_merged[col] = df_merged[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)
            return df_merged

        if df_concepts is None:
             # Try loading from previous step file
             prev_file = self.output_dir / "dp_extracted_concepts.csv"
             if not prev_file.exists():
                 raise FileNotFoundError("Missing input for merge step.")
             df_concepts = pd.read_csv(prev_file, sep="|")

        # 1. Aggregation
        print("🚀 [Pipeline] Starting entity aggregation...")
        df_merged = merge_concepts(df_concepts)
        # Sort before saving merge_entity.csv
        df_merged = df_merged.sort_values(by=['chunk_id']).reset_index(drop=True)
        df_merged.to_csv(merge_csv_file, sep="|", index=False)
        
        # 2. Text enrichment
        print("🛠️ [Pipeline] Building enriched entity texts...")
        enriched_texts = df_merged.apply(build_enriched_name_text, axis=1).tolist()
        
        # 3. Compute vectors (dual: Name+Synonyms and Description)
        print("🚀 [Pipeline] Computing enriched entity vectors...")
        df_merged['vec_entity'] = self._batch_embed(enriched_texts, desc="Vec: Entity")
        
        print("🚀 [Pipeline] Computing description vectors...")
        desc_texts = df_merged['description'].fillna("").astype(str).tolist()
        df_merged['vec_desc'] = self._batch_embed(desc_texts, desc="Vec: Desc")
        
        # 4. Save
        print(f"💾 [Pipeline] Saving aggregated vector table to: {cache_file}")
        df_to_save = df_merged.copy()
        # Ensure final output is ordered
        df_to_save = df_to_save.sort_values(by=['chunk_id']).reset_index(drop=True)
        df_to_save['vec_entity'] = df_to_save['vec_entity'].apply(lambda x: x.tolist())
        df_to_save['vec_desc'] = df_to_save['vec_desc'].apply(lambda x: x.tolist())
        df_to_save.to_parquet(cache_file, index=False)
        
        return df_merged

    # =================================================================
    # 3. Other existing pipelines (Splitting, Standardization, etc.)
    # =================================================================
    def load_and_split_data(self, json_path, max_contexts=None, chunk_size=300, chunk_overlap=50):
        cache_file = self.output_dir / "chunk.csv"
        print(f"🚀 [Pipeline] Loading 2WikiMultiHopQA data: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        global_chunk_id = 0
        limit = max_contexts if max_contexts else len(data)

        # Iterate over each QA pair (Item)
        for i, item in enumerate(data[:limit]):
            # The context field in 2Wiki is a list containing multiple paragraphs
            # Format: [ ["Title1", ["Sent1", "Sent2"]], ["Title2", ["Sent3"]] ]
            context_list = item.get("context", [])
            
            for paragraph in context_list:
                # Error tolerance: sometimes format could be [Title, [Sentences]]
                if len(paragraph) < 2: continue
                
                title = paragraph[0]
                sentences = paragraph[1]
                
                # Combine the list of sentences into a single text
                # Sentences in 2Wiki are usually already split, just concatenate them
                paragraph_text = "".join(sentences)
                
                # If the text is too long, we can further split using self.splitter
                # But generally 2Wiki paragraphs are not long, use directly as a chunk
                
                # Build chunk
                chunks.append({
                    "context_id": i,          # Relates to the i-th question
                    "chunk_id": global_chunk_id,
                    "title": title,
                    "text": f"{title} information: {paragraph_text}" # Add title for semantic enhancement
                })
                global_chunk_id += 1
        
        df_chunks = pd.DataFrame(chunks)
        df_chunks.to_csv(cache_file, sep="|", index=False)
        print(f"✅ Loaded {len(df_chunks)} chunks from 2WikiMultiHopQA.")
        return df_chunks

    def standardize_entities(self, df_concepts_flat: pd.DataFrame):
        """
        Core function:
        1. Parse and aggregate entity synonyms (Entity + Synonyms) -> Rich Input Text
        2. Compute entity embeddings
        3. Run DBSCAN clustering
        4. Generate standard name mapping
        """
        print("🚀 [Pipeline] Starting entity standardization process...")
        
        # --- 3.1 Data preprocessing and parsing ---
        df_concepts_flat = df_concepts_flat.copy()
        df_concepts_flat['Entity'] = df_concepts_flat['Entity'].astype(str).str.strip()
        df_concepts_flat['category'] = df_concepts_flat['category'].astype(str).str.strip()

        # Helper parsing function: handle string list like "['a', 'b']" after CSV reading
        def parse_synonyms_col(x):
            if isinstance(x, list): return x
            if isinstance(x, str):
                try: 
                    val = ast.literal_eval(x)
                    return val if isinstance(val, list) else []
                except: 
                    return []
            return []

        # Parse synonyms column
        df_concepts_flat['synonyms'] = df_concepts_flat['synonyms'].apply(parse_synonyms_col)
        
        # Add the 'Entity' (original mention) to the synonyms list
        # This implements the merge logic: ensuring Rich_Input_Text contains the entity itself
        # New code: add and deduplicate with set(), then convert back to sorted list
        df_concepts_flat['synonyms'] = df_concepts_flat.apply(
            lambda row: sorted(list(set(row['synonyms'] + [str(row['Entity'])]))), axis=1
        )
        print("First 5 rows expanded:", df_concepts_flat.head(5))
        # --- 3.2 Aggregate to generate Rich Input Text ---
        # Groupby aggregation logic: flatten and deduplicate list of lists -> set -> string
        grouped = df_concepts_flat.groupby(['context_id', 'Entity', 'category'])['synonyms'].apply(
            lambda x: ' | '.join(sorted(list(set([item for sublist in x for item in sublist if item]))))
        ).reset_index()
        
        grouped.rename(columns={'synonyms': 'Aggregated_Synonyms'}, inplace=True)
        
        # Build Rich Input Text
        grouped['Rich_Input_Text'] = grouped.apply(
            lambda row: f"Entity: {row['Entity']}. synonyms: {row['Aggregated_Synonyms']}", axis=1
        )
        
        print("First 5 rows after grouping:", grouped.head(5))
        # --- 3.3 Compute entity embeddings (reuse general function) ---
        df_embedded = self.compute_embeddings(
            df=grouped, 
            text_col='Rich_Input_Text', 
            id_col='Entity', 
            file_name="entity_embeddings.parquet"
        )

        # --- 3.4 DBSCAN clustering ---
        df_map = self._perform_clustering(df_embedded)
        
        # --- 3.5 Apply mapping ---
        return self._apply_mapping(df_concepts_flat, df_map)

    def _perform_clustering(self, df_linked: pd.DataFrame):
        EPS_CONFIG = {
            # --- Original categories ---
            'Person': 0.05,       # Name variations (e.g., J. Biden vs Joe Biden)
            'Organization': 0.08, # Full name vs abbreviation (e.g., NASA vs National Aeronautics...)
            'Location': 0.05,     # Place names relatively fixed, but may have "City of X" vs "X"
            'Structure': 0.05,    # Building names relatively fixed
            'Natural': 0.08,      # Species/chemicals may have aliases
            'Work': 0.05,         # Book/movie titles usually specific
            'Event': 0.05,        # Event names
            'Time': 0.01,         # Time should be highly normalized, strict match
            'Quantity': 0.01,     # Numeric values should be strict match
            'Product': 0.05,      # Products/vehicles (e.g., Boeing 747 vs 747)
            'Award': 0.05,        # Awards (e.g., Oscar vs Academy Award)
            'Role': 0.08,         # Roles/positions (e.g., CEO vs Chief Executive Officer) - semantically close
            'Concept': 0.08,      # Concepts/disciplines (e.g., AI vs Artificial Intelligence) - larger semantic span
            'Group': 0.08         # Groups/nationalities (e.g., American vs Americans)
        }
        
        DEFAULT_EPS = 0.10
        final_entity_map = []
        context_cluster_max_id = {}
        
        print(f"⏳ [Pipeline] Executing intelligent grouping clustering (HAC + Genealogy Penalty)...")        # Group by context_id and category to keep them independent
        for (context_id, category), group in tqdm(df_linked.groupby(['context_id', 'category']), desc="Clustering"):
            current_max = context_cluster_max_id.get(context_id, -1)
            eps = EPS_CONFIG.get(category, DEFAULT_EPS)
            
            # 1. Extract unique entities (deduplicate)
            unique_group = group[['Entity', 'entity_embedding']].drop_duplicates(subset=['Entity']).reset_index(drop=True)
            entities = unique_group['Entity'].tolist()
            
            # Case A: Only one entity, no clustering needed, it is its own cluster
            if len(unique_group) < 2:
                unique_group['cluster_id'] = current_max + 1
                current_max += 1 
            
            # Case B: Multiple entities, perform advanced clustering
            else:
                # Stack embeddings
                matrix = np.vstack(unique_group['entity_embedding'].values)
                
                # --- Step 1: Compute base cosine distance matrix ---
                # cosine_distances = 1 - cosine_similarity (range 0~2, smaller means closer)
                dist_matrix = cosine_distances(matrix)
                
                # --- Step 2: (For persons only) Apply genealogy penalty ---
                if category == 'Person':
                    dist_matrix = apply_genealogical_penalty(entities, dist_matrix)

                # --- Step 3: Hierarchical Agglomerative Clustering (HAC) ---
                # metric='precomputed': tell the algorithm we pass a distance matrix, not raw vectors
                # linkage='single': single-link strategy, allows chaining A->B->C (key to solving long/short name problems)
                # distance_threshold=eps: merge if distance is less than this value
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    metric='precomputed', 
                    linkage='single', 
                    distance_threshold=eps
                )
                
                # Fit
                clusters = clustering.fit_predict(dist_matrix)
                
                # Assign cluster IDs (add offset for current context to avoid conflict)
                unique_group['cluster_id'] = clusters + (current_max + 1)
                
                if len(clusters) > 0:
                    current_max = unique_group['cluster_id'].max()

            # --- Standard name generation ---
            # Strategy: choose the longest name within the cluster as Standard Entity (usually full name contains most information)
            def get_standard(sub_df):
                valid = sub_df[sub_df['Entity'].str.len() > 0]
                if valid.empty: return sub_df['Entity'].iloc[0] if not sub_df.empty else ""
                return valid.loc[valid['Entity'].str.len().idxmax(), 'Entity']
            
            cluster_standards = unique_group.groupby('cluster_id').apply(get_standard, include_groups=False).to_dict()
            unique_group['Standard_Entity'] = unique_group['cluster_id'].map(cluster_standards)
            
            # Update context state
            context_cluster_max_id[context_id] = current_max
            unique_group['context_id'] = context_id
            unique_group['category'] = category
            final_entity_map.append(unique_group)
            
        # Concatenate all grouped results
        df_map = pd.concat(final_entity_map, ignore_index=True)
        df_map.rename(columns={'Entity': 'Original_Entity'}, inplace=True)
        
        # Call existing postprocessing function (if any)
        df_map = post_process_person_entities(df_map)
        
        return df_map

    def _apply_mapping(self, df_concepts, df_map):
        # Important: When setting index we must include 'category' to prevent conflicts between entities with same name but different types
        # df_map contains columns: 'Original_Entity', corresponding to 'Entity' in df_concepts
        
        # 1. Build composite key dictionaries
        std_lookup = df_map.set_index(['context_id', 'Original_Entity', 'category'])['Standard_Entity'].to_dict()
        cls_lookup = df_map.set_index(['context_id', 'Original_Entity', 'category'])['cluster_id'].to_dict()
        
        def lookup(row, lookup_dict, default_col=None):
            # Important: look up must also include category
            key = (row['context_id'], row['Entity'], row['category'])
            val = lookup_dict.get(key)
            if val is not None: return val
            
            # Fallback: if exact match fails (rare), try fallback to entity-only match (optional, depending on case)
            # For rigor, we directly return default
            return row[default_col] if default_col else -1

        print("🚀 [Pipeline] Applying entity mapping (Key: Context + Entity + Category)...")
        df_concepts['Standard_Entity'] = df_concepts.apply(lambda r: lookup(r, std_lookup, 'Entity'), axis=1)
        df_concepts['cluster_id'] = df_concepts.apply(lambda r: lookup(r, cls_lookup), axis=1)
        
        # Ensure output is ordered
        df_concepts = df_concepts.sort_values(by=['chunk_id']).reset_index(drop=True)
        
        save_path = self.output_dir / "dp_extracted_concepts.csv"
        df_concepts.to_csv(save_path, sep="|", index=False)
        print(f"✅ [Pipeline] Standardization completed.")
        return df_concepts
        
    def generate_entity_map_for_graph(self, df_concepts):
        valid = df_concepts[df_concepts['Standard_Entity'].notna() & (df_concepts['Standard_Entity'] != "")]
        return valid.groupby(['context_id', 'chunk_id']).apply(
            lambda x: dict(zip(x['Entity'], x['Standard_Entity']))
        ).to_dict()
    
    def extract_qa_pairs(self, json_path, max_contexts=None):
        print(f"🚀 [Pipeline] Extracting 2WikiMultiHopQA QA pairs...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        limit = max_contexts if max_contexts else len(data)
        relations = []
        
        for i, item in enumerate(data[:limit]):
            question_text = item.get("question", "").strip()
            answer_text = item.get("answer", "")
            
            # In 2Wiki the answer is usually a string directly, but convert to string to be safe
            if not isinstance(answer_text, str):
                answer_text = str(answer_text)

            relations.append({
                "question": question_text,
                "answer": answer_text,
                "context_id": i
            })
            
        df_qa = pd.DataFrame(relations)
        df_qa.to_csv(self.output_dir / "qa.csv", sep="|", index=False)
        return df_qa
