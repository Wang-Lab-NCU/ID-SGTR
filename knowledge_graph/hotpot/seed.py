import pandas as pd
import numpy as np
import os
import sys
import ast
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi  # [New] Import BM25

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import get_embeddings_model

class SemanticMatcher:
    def __init__(self, parquet_path):
        """
        Initialize the matcher: load Parquet and build in-memory matrices.
        """
        self.parquet_path = Path(parquet_path)
        print(f"🔄 Loading anchor database: {self.parquet_path.name} ...")
        
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"File not found: {self.parquet_path}")
            
        # 1. Read data
        self.df = pd.read_parquet(self.parquet_path)
        
        # 2. Preprocess: parse synonyms column
        print("⚙️ Parsing synonyms...")
        def parse_synonyms(x):
            if isinstance(x, (list, np.ndarray)):
                return list(x)
            if isinstance(x, str):
                try:
                    val = ast.literal_eval(x)
                    return val if isinstance(val, list) else []
                except:
                    return []
            return []
            
        self.df['synonyms_list'] = self.df['synonyms'].apply(parse_synonyms)

        # 3. Preprocess: build numpy matrices
        print("⚙️ Building vector matrices...")
        
        self.matrix_entity = np.stack(self.df['vec_entity'].values)
        
        # Defensive handling for empty description vectors
        if self.df['vec_desc'].isnull().any():
             print("⚠️ Warning: Found empty description vectors; will fill with zero vectors.")
             dim = self.matrix_entity.shape[1]
             self.df['vec_desc'] = self.df['vec_desc'].apply(
                 lambda x: x if x is not None else np.zeros(dim)
             )
        self.matrix_desc = np.stack(self.df['vec_desc'].values)
        
        # 4. Initialize embedding model
        self.embed_model = get_embeddings_model(dimensions=1024)
        print(f"✅ Initialization complete. In-memory contains {len(self.df)} aggregated entities.")

    def get_query_vector(self, query):
        """Call API to get query vector"""
        vec = self.embed_model.embed_query(query)
        return np.array(vec).reshape(1, -1)

    def _extract_keywords_for_vector(self, query):
        """
        [Key optimization] Query purification for description vector retrieval.
        """
        # 1. Prefer quoted content (strong intent)
        quotes = re.findall(r'"([^"]*)"', query)
        if quotes: return " ".join(quotes)

        # 2. Simple part-of-speech filtering (keep only nouns and capitalized words)
        stopwords = {
            'what', 'which', 'who', 'where', 'when', 'how', 'is', 'are', 'was', 'were', 
            'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about',
            'this', 'that', 'these', 'those', 'does', 'did', 'can', 'could',
        }
        
        words = query.split()
        keywords = [w for w in words if w[0].isupper() or (w.lower() not in stopwords and len(w) > 2)]
        
        return " ".join(keywords) if keywords else query

    def _calculate_exact_match_score(self, candidate_df, query):
        """
        [Renamed] Calculate absolute exact substring match score.
        Still retained because for simple entities (e.g., years or short names), exact match has the highest confidence.
        """
        query_lower = query.lower()
        scores = []
        
        for idx, row in candidate_df.iterrows():
            match_score = 0.0
            
            # Check standard name
            std_name = str(row['Standard_Entity']).lower().strip()
            if len(std_name) >= 2 and std_name in query_lower:
                match_score = 1.0
            else:
                # Check synonyms
                syns = row.get('synonyms_list', [])
                if syns:
                    for s in syns:
                        s_str = str(s).lower().strip()
                        if len(s_str) >= 2 and s_str in query_lower:
                            match_score = 1.0
                            break
            scores.append(match_score)
            
        return np.array(scores)

    def _calculate_bm25_score(self, candidate_df, query):
        """
        [New core] Dynamically build BM25 index for the current context, compute sparse similarity.
        Combine entity name + synonyms + description into a "document", use TF-IDF logic to lock low-frequency domain terms.
        """
        tokenized_corpus = []
        
        for idx, row in candidate_df.iterrows():
            # Concatenate all possible entity names and description
            text_parts = [str(row.get('Standard_Entity', ''))]
            text_parts.extend([str(s) for s in row.get('synonyms_list', [])])
            text_parts.append(str(row.get('description', '')))
            
            full_text = " ".join(text_parts).lower()
            # Simple tokenization (keep letters and numbers)
            tokens = re.findall(r'\w+', full_text)
            tokenized_corpus.append(tokens)
            
        if not tokenized_corpus:
            return np.zeros(len(candidate_df))
            
        # Instantiate BM25
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = re.findall(r'\w+', query.lower())
        
        # Compute scores
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Min-Max normalize to [0,1]
        max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        return np.array([s / max_score for s in bm25_scores])

    def link(self, query, context_id, top_k=10, lambda_weights=(0.25, 0.40, 0.15, 0.20)):
        """
        Core linking function (BM25 + Dense ultimate form)
        :param lambda_weights: (w_name, w_desc, w_exact, w_bm25) four recall weights
        """
        w_name, w_desc, w_exact, w_bm25 = lambda_weights
        
        # === Modification point: intelligent global/local mode detection ===
        if context_id is None or str(context_id).lower() == 'global':
            # 🌍 Global mode: allow all, mask all True
            mask = np.ones(len(self.df), dtype=bool)
        else:
            # 🏠 Local mode: strictly check context_id
            mask = self.df['context_id'].astype(str) == str(context_id)
            
        if not mask.any(): 
            return pd.DataFrame()
        # ==================================

        # --- 1. Dual query vectorization (Dense) ---
        v_q_name = self.get_query_vector(query)
        clean_query = self._extract_keywords_for_vector(query)
        v_q_desc = self.get_query_vector(clean_query)

        # --- 2. Matrix slicing ---
        sub_matrix_entity = self.matrix_entity[mask]
        sub_matrix_desc = self.matrix_desc[mask]
        candidate_rows = self.df[mask].copy()

        # --- 3. Vector computation (Dense Scores) ---
        s_name = cosine_similarity(v_q_name, sub_matrix_entity)[0]
        s_desc = cosine_similarity(v_q_desc, sub_matrix_desc)[0]
        
        # --- 4. Lexical matching and BM25 (Sparse/Lexical Scores) ---
        s_exact = self._calculate_exact_match_score(candidate_rows, query)
        s_bm25 = self._calculate_bm25_score(candidate_rows, query)  # [New]

        # --- 5. Final scoring (4D Fusion) ---
        final_scores = (w_name * s_name) + (w_desc * s_desc) + (w_exact * s_exact) + (w_bm25 * s_bm25)
        
        candidate_rows['Score'] = final_scores
        candidate_rows['S_name'] = s_name
        candidate_rows['S_desc'] = s_desc
        candidate_rows['S_exact'] = s_exact
        candidate_rows['S_bm25'] = s_bm25  # Keep for debugging
        
        # Return results
        cols = ['Standard_Entity', 'context_id', 'category', 'Score', 'S_name', 'S_desc', 'S_exact', 'S_bm25', 'synonyms']
        return candidate_rows.sort_values(by='Score', ascending=False).head(top_k)[cols]
            
# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # --- Path configuration ---
    # Please modify the following paths according to your actual directory
    BASE_DIR = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\hotpot\ds1000_2"
    PARQUET_FILE = os.path.join(BASE_DIR, "concepts_merged_with_vectors.parquet")
    QA_FILE = os.path.join(BASE_DIR, "qa.csv")
    
    # 1. Instantiate (load once, keep in memory)
    try:
        linker = SemanticMatcher(PARQUET_FILE)
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)
    
    # 2. Read QA file for testing
    if os.path.exists(QA_FILE):
        df_qa = pd.read_csv(QA_FILE, sep="|")
        
        print("\n🚀 Starting entity linking test (Top 5 Questions)...")
        print("="*80)
        
        # Hybrid weights: (Vector_Name, Vector_Desc, Lexical_Match)
        # Here 0.3 Lexical weight makes exact-matching entities score significantly higher
        HYBRID_WEIGHTS = (0.25, 0.40, 0.15, 0.20) 

        # Test first 10 questions
        for i, row in df_qa.head(10).iterrows():
            q = row['question']
            ctx = row['context_id']
            
            print(f"\n❓ [Q{i+1}] Context: {ctx}")
            print(f"   Query: {q}")
            
            # Call linker
            df_res = linker.link(q, context_id=None, top_k=15, lambda_weights=HYBRID_WEIGHTS)
            
            if not df_res.empty:
                print(f"✅ Found {len(df_res)} candidate entities (sorted by Score descending):")
                # Print all score details
                # 🌍 Update printed columns: add S_exact and S_bm25
                print(df_res[['Standard_Entity', 'context_id','category', 'Score', 'S_name', 'S_desc', 'S_exact', 'S_bm25', 'synonyms']].to_string(
                    index=False, 
                    formatters={
                        'Score': '{:.3f}'.format,
                        'S_name': '{:.3f}'.format,
                        'S_desc': '{:.3f}'.format,
                        'S_exact': '{:.3f}'.format,
                        'S_bm25': '{:.3f}'.format
                    }
                ))
            else:
                print("❌ No matching entities found")
            print("-" * 80)
    else:
        print(f"❌ QA file not found: {QA_FILE}")
