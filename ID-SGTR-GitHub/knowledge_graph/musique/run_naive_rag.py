import torch
import numpy as np
import pandas as pd
import ast
import re
import os
import sys
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 0. Helper class: console color output
# ==========================================
class Colors:
    HEADER = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# ==========================================
# 1. Environment and path configuration
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import your model loading utilities
from utils import get_embeddings_model, get_chat_model
from helper import parallel_llm_processor

# ==========================================
# 2. Core engine: Naive Vector RAG (Dense-Only, Global Mode)
# ==========================================
class Naive_Vector_RAG_Engine:
    def __init__(self, chunk_df, device=None):
        """
        Initialize pure vector retrieval engine
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Initializing Naive RAG engine (Device: {self.device})...")

        self.gpu_lock = threading.Lock()

        # -------------------------------------------------------
        # Module 1: Data preprocessing (only chunk vector logic)
        # -------------------------------------------------------
        print("📦 [1/3] Data preprocessing and context index construction...")
        self.chunk_df = chunk_df.copy()
        if 'chunk_id' in self.chunk_df.columns:
            self.chunk_df['chunk_id'] = self.chunk_df['chunk_id'].astype(str)
            self.chunk_df.set_index('chunk_id', inplace=True)
            
        self.chunk_df.index = self.chunk_df.index.astype(str)

        def parse_vec_safe(x):
            if isinstance(x, np.ndarray): return x.astype(np.float32)
            if isinstance(x, list): return np.array(x, dtype=np.float32)
            if isinstance(x, str):
                try:
                    if x.strip().startswith('['):
                        return np.array(ast.literal_eval(x), dtype=np.float32)
                except:
                    return None
            return None

        # Ensure embedding column is numpy array
        if 'embedding_np' not in self.chunk_df.columns:
            tqdm.pandas(desc="Parsing Vectors")
            self.chunk_df['embedding_np'] = self.chunk_df['embedding'].progress_apply(parse_vec_safe)

        # Build context_id to chunk_id index (as a fallback for local retrieval)
        self.chunk_dict_by_ctx = {}
        if 'context_id' in self.chunk_df.columns:
            self.chunk_df['context_id'] = self.chunk_df['context_id'].astype(str)
            self.chunk_dict_by_ctx = self.chunk_df.groupby('context_id')['text'].apply(lambda x: x.index.tolist()).to_dict()

        # -------------------------------------------------------
        # Module 2: Load embedding model (BGE-M3)
        # -------------------------------------------------------
        print("📥 [2/3] Loading Embedding model (BGE-M3)...")
        self.embed_model = get_embeddings_model(dimensions=1024)

        # -------------------------------------------------------
        # Module 3: Load reasoning LLM (Qwen3-8B)
        # -------------------------------------------------------
        print("🤖 [3/3] Initializing reasoning LLM (Qwen3-8B)...")
        self.llm = get_chat_model(task_type="kg_query")

    def _get_top_chunks_dense_only(self, candidate_cids, query_vec, top_k=3):
        """
        Pure dense vector similarity computation on the whole corpus
        """
        valid_cids = [c for c in candidate_cids if c in self.chunk_df.index]
        if not valid_cids or query_vec is None: 
            return valid_cids[:top_k]

        try:
            # 1. Extract content vector matrix (when retrieving from whole corpus, this extracts vectors of all chunks)
            content_matrix = np.stack(self.chunk_df.loc[valid_cids, 'embedding_np'].values)
            
            # 2. Compute cosine similarity
            q_norm = np.linalg.norm(query_vec)
            c_norms = np.linalg.norm(content_matrix, axis=1)
            cosine_sims = (content_matrix @ query_vec) / (c_norms * q_norm + 1e-9)
            
            # 3. Sort and take Top-K
            sorted_indices = np.argsort(cosine_sims)[::-1]
            return [valid_cids[i] for i in sorted_indices[:top_k]]

        except Exception as e:
            print(f"⚠️ Vector retrieval error: {e}")
            return valid_cids[:top_k]

    def _get_chunk_text(self, chunk_id):
        try:
            return str(self.chunk_df.loc[str(chunk_id), 'text']).replace('\n', ' ')
        except: 
            return ""

    def solve(self, query, context_id=None, verbose=False, top_k=3):
        """
        End-to-end retrieval and generation (seamlessly supports Global or Local)
        """
        if verbose:
            print(f"{Colors.HEADER}\n=================================================={Colors.ENDC}")
            print(f"{Colors.BOLD}🔍 [Naive RAG] Query: {query}{Colors.ENDC}")

        # 1. Vectorize query (with lock to prevent CUDA crashes)
        query_vec = None
        with self.gpu_lock:
            try:
                query_vec = self.embed_model.embed_query(query)
                query_vec = np.array(query_vec, dtype=np.float32)
            except Exception as e: 
                print(f"⚠️ Query Embedding Failed: {e}")
                return "Error: Embedding failed", "Naive-Vector"

        # 2. ✅ [Core modification]: Get candidate chunk IDs, completely remove isolation!
        candidate_cids = []
        if context_id is None or str(context_id).lower() == 'global':
            # 🌍 Global mode: take all chunk IDs in the knowledge base to search everywhere
            candidate_cids = self.chunk_df.index.tolist()
        elif str(context_id) in self.chunk_dict_by_ctx:
            # 🏠 Local mode (only for backward compatibility)
            candidate_cids = self.chunk_dict_by_ctx[str(context_id)]
        else:
            # Fallback to whole corpus
            candidate_cids = self.chunk_df.index.tolist()

        # 3. Pure vector retrieval Top-K comparison on the whole corpus (default 3)
        top_cids = self._get_top_chunks_dense_only(candidate_cids, query_vec, top_k=top_k)
        
        # 4. Assemble prompt text
        top_chunks_text = []
        for cid in top_cids:
            txt = self._get_chunk_text(cid)
            clean_txt = txt[:1000].replace('\n', ' ')  # limit per chunk length
            top_chunks_text.append(f"[Ref {cid}] {clean_txt}")
            if verbose:
                print(f"{Colors.CYAN}  -> Selected Chunk [{cid}]{Colors.ENDC}")

        context_str = "\n".join(top_chunks_text) if top_chunks_text else "No specific context found."

        # 5. Build prompt (strictly restrict answer based on reference material)
        prompt = f"""
You are a high-precision QA system answering a complex question. You must answer based DIRECTLY on the provided Reference Text.

### User Query
"{query}"

### Reference Context (Primary Evidence)
{context_str}

### Task
Answer the query using ONLY the information above. Read very carefully, watching out for distractor entities with similar names.

### Strict Rules
1. **Format**:
   - If the answer is explicitly in the text, extract the exact entity/value.
   - If it's a Yes/No question, answer "Yes" or "No".
2. **Output**:  
   - Keep output minimal. 
   - `Final Answer: [Clean Entity Name / Yes / No / data / etc.]`

### ⛔ OUTPUT RESTRICTIONS
- **NO** sentences or paragraphs.
- **NO** explanations.
"""
        
        # 6. Call LLM
        resp = self.llm.invoke(prompt).content.strip()
        if "Final Answer:" in resp:
            resp = resp.split("Final Answer:")[-1].strip()
        
        if verbose:
            print(f"{Colors.GREEN}💡 [LLM Answer]: {resp}{Colors.ENDC}")

        return resp, "Naive-Vector-Global-Top3"


# ==========================================
# 3. Main program entry (multi-threaded parallel processing)
# ==========================================
if __name__ == "__main__":
    # Configure your data paths
    DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\musique\ds1000"
    
    QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
    CHUNK_EMB_FILE = os.path.join(DATA_ROOT, "chunks_with_embeddings.parquet")
    CHUNK_RAW_FILE = os.path.join(DATA_ROOT, "chunk.csv")

    # 1. Data loading
    try:
        df_qa = pd.read_csv(QA_FILE, sep="|")
        
        if os.path.exists(CHUNK_EMB_FILE):
            print(f"📦 Loading Chunk data with vectors: {CHUNK_EMB_FILE}")
            df_chunk = pd.read_parquet(CHUNK_EMB_FILE)
        else:
            print(f"⚠️ Embedding Parquet not found, loading raw CSV: {CHUNK_RAW_FILE}")
            df_chunk = pd.read_csv(CHUNK_RAW_FILE, sep="|")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        sys.exit(1)

    # 2. Initialize pure vector engine
    engine = Naive_Vector_RAG_Engine(chunk_df=df_chunk)

    # 3. Select test data range
    target_data = df_qa.head(1000)
    print(f"\n📝 Starting concurrent processing of {len(target_data)} queries (pure vector global retrieval mode)...")
    
    def process_query_wrapper(i: int, row: pd.Series):
        q = row['question']
        ctx = row['context_id'] # Keep for logging and comparison
        gold = row['answer']
        
        # ✅ [Core modification]: Force pass 'global', completely abandon the original context isolation!
        pred_answer, strategy = engine.solve(q, context_id="global", verbose=False, top_k=3)
                
        return i, {
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "strategy": strategy,
            "context_id": ctx # Keep original context_id so evaluation script can compute scores
        }
        
    # 4. Start multi-threading
    processed_results = parallel_llm_processor(
        dataframe=target_data,
        processing_func=process_query_wrapper,
        start_message="🚀 Starting multi-threaded pure vector inference...",
        max_workers=5,
        max_retries=6,
        initial_delay=2
    )

    # 5. Organize and save results
    processed_results.sort(key=lambda x: x[0])
    final_data = [item[1] for item in processed_results]
    
    output_path = os.path.join(current_dir, "query_results_naive_rag_dense_only_GLOBAL_Qwen3-8B_top3.csv")
    pd.DataFrame(final_data).to_csv(output_path, index=False, sep="|")
    print(f"\n✅ Processing completed! Global Naive RAG results saved to: {output_path}")
