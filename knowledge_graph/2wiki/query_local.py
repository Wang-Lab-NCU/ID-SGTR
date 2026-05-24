import torch
import time
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import json
import re
from tqdm import tqdm
import os
import sys
import ast
import threading
from rank_bm25 import BM25Okapi
import difflib
from typing import List, Tuple, Callable, Any, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 0. Helper class: Console color output (for debugging log distinction)
# ==========================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ==========================================
# 1. Token optimization configuration class
# Controls the context window size sent to LLM and similarity truncation to prevent token explosion or noise
# ==========================================
class TokenConfig:
    # --- Stage 0: Initial definition check ---
    STAGE0_ADD_CHUNKS = True          # Whether to forcibly attach related text chunks in entity definition stage
    STAGE0_MAX_CHUNKS = 3             # Maximum number of chunks to pass in the initial stage
    
    # --- Stage N: Path expansion ---
    TOP_K_NEIGHBORS = 12              # Maximum number of neighbor nodes to explore per step in the graph
    CHUNK_SIM_THRESHOLD_STRICT = 0.35 # Cosine similarity threshold for semantic supplement chunks (high threshold to avoid noise)
    CHUNK_SIM_THRESHOLD_LOOSE = 0.25  # Cosine similarity threshold for structurally associated chunks (lower threshold because graph edges guarantee relevance)
    MIN_EDGE_SCORE = 0.05             # Minimum comprehensive edge weight; edges below this are considered disconnected
    
    # --- General text limits ---
    MAX_CANDIDATE_POOL = 15           # Maximum number of Next Hop candidates sent to LLM
    CHUNK_CHAR_LIMIT = 1000           # Character truncation length for a single text chunk (prevents overly long texts)
    MAX_CHUNKS_IN_PROMPT = 3          # Maximum total chunks allowed when assembling each prompt

# ==========================================
# 2. Environment and path configuration
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from adapt.adapt import IntentClassifier, dynamic_weight_modulation, INPUT_DIM, HIDDEN_DIM
from utils import get_embeddings_model, get_llm_model, get_chat_model
from seed import SemanticMatcher 
from helper import parallel_llm_processor

# ==========================================
# 3. Core engine class: ID-SGTR (supports multi‑threading and Agent reasoning)
# ==========================================
class ID_SGTR_Reasoning_Engine:
    def __init__(self, 
                 intent_model_path, 
                 parquet_path, 
                 graph_df, 
                 chunk_df, 
                 proximity_df=None, 
                 device=None,
                 edge_mask_ratio=0.0,
                 random_seed=42):
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_mask_ratio = edge_mask_ratio
        self.random_seed = random_seed
        print(f"🔧 Initializing engine (Device: {self.device})...")

        # [Core mechanism] Thread lock: prevents concurrent calls to local GPU models (e.g., Embedding/classification network) from causing CUDA errors
        self.gpu_lock = threading.Lock()

        # --- Module 1: Load intent classification network ---
        print("📥 [1/4] Loading intent classification network...")
        self.intent_model = IntentClassifier(INPUT_DIM, HIDDEN_DIM).to(self.device)
        try:
            if os.path.exists(intent_model_path):
                self.intent_model.load_state_dict(torch.load(intent_model_path, map_location=self.device))
                self.intent_model.eval()
            else:
                print(f"⚠️ Intent model file not found: {intent_model_path}")
        except Exception as e:
            print(f"❌ Failed to load intent model: {e}")
        
        # --- Module 2: Load semantic matching module (entity linking) ---
        print("📥 [2/4] Loading semantic anchor database...")
        self.matcher = SemanticMatcher(parquet_path)

        if hasattr(self.matcher, 'embed_model'):
            self.graph_embed_model = self.matcher.embed_model
            print("✅ Reusing SemanticMatcher's Embedding model")
        else:
            print("⚠️ Creating new Embedding model for graph reasoning")
            self.graph_embed_model = get_embeddings_model(dimensions=1024)

        # --- Module 3: Data preprocessing and knowledge graph construction ---
        print("🕸️ [3/4] Data preprocessing and graph construction (enforcing defensive ID string conversion)...")
        self.chunk_df = chunk_df.copy()
        
        # ✅ Defensive programming: force convert all input IDs to strings
        if 'context_id' in self.chunk_df.columns:
            self.chunk_df['context_id'] = self.chunk_df['context_id'].astype(str)
        if 'chunk_id' in self.chunk_df.columns:
            self.chunk_df['chunk_id'] = self.chunk_df['chunk_id'].astype(str)
            self.chunk_df.set_index('chunk_id', inplace=True)
            
        self.chunk_df.index = self.chunk_df.index.astype(str)
        
        def parse_vec_safe(x):
            """Safely parse vectors: compatible with numpy, list, and string representations of arrays"""
            if isinstance(x, np.ndarray): return x.astype(np.float32)
            if isinstance(x, list): return np.array(x, dtype=np.float32)
            if isinstance(x, str):
                try:
                    if x.strip().startswith('['):
                        return np.array(ast.literal_eval(x), dtype=np.float32)
                except: return None
            return None

        if 'embedding_np' not in self.chunk_df.columns:
             tqdm.pandas(desc="Parsing Vectors")
             self.chunk_df['embedding_np'] = self.chunk_df['embedding'].progress_apply(parse_vec_safe)
             
        if 'title_embedding' in self.chunk_df.columns:
            print("   ✅ Detect pre-computed 'title_embedding', loading...")
            self.chunk_df['title_embedding_np'] = self.chunk_df['title_embedding'].apply(parse_vec_safe)
        
        # Build inverted index from context_id to chunk_ids
        self.chunk_dict_by_ctx = {}
        if 'context_id' in self.chunk_df.columns:
            print("   ✅ Building Context-to-Chunk Index...")
            self.chunk_dict_by_ctx = self.chunk_df.groupby('context_id')['text'].apply(lambda x: x.index.tolist()).to_dict()        
        
        print("   -> Building Node-to-Matrix Index...")
        self.node_to_vec_idx = {
            str(name): idx for idx, name in enumerate(self.matcher.df['Standard_Entity'])
        }

        # Build NetworkX graph
        self.G = self._build_hybrid_graph(graph_df, proximity_df)
        
        # --- Module 4: Load large language model for reasoning ---
        print("🤖 [4/4] Initializing reasoning LLM...")
        self.llm_filter = get_chat_model(task_type="reasoning")
        self.llm = get_chat_model(task_type="kg_query")        

    def _build_hybrid_graph(self, graph_df, proximity_df):
        """Build hybrid graph (explicit + implicit edges)"""
        G = nx.Graph()
        
        # ✅ Defensive programming: convert all IDs in graph input to strings as well
        df_g = graph_df.copy()
        df_g['node_1'] = df_g['node_1'].astype(str)
        df_g['node_2'] = df_g['node_2'].astype(str)
        if 'chunk_id' in df_g.columns:
            df_g['chunk_id'] = df_g['chunk_id'].astype(str)
            
        has_ctx = 'context_id' in df_g.columns 
        if has_ctx:
            df_g['context_id'] = df_g['context_id'].astype(str)
        
        if self.edge_mask_ratio > 0:
            np.random.seed(self.random_seed)
            
        dropped_count = 0
        total_explicit = len(df_g)
        
        for row in tqdm(df_g.itertuples(index=False), total=total_explicit, desc="Graph Nodes"):
            if self.edge_mask_ratio > 0.0:
                if np.random.rand() < self.edge_mask_ratio:
                    dropped_count += 1
                    continue 
                
            u, v = row.node_1, row.node_2
            ctx_id = str(row.context_id) if has_ctx and pd.notnull(row.context_id) else "-1"
            
            if G.has_edge(u, v):
                G[u][v]['context_ids'].add(ctx_id)
                G[u][v]['chunk_ids'].append(row.chunk_id)
            else:
                G.add_edge(u, v, 
                          type='explicit', 
                          relation=row.edge, 
                          chunk_ids=[row.chunk_id],
                          context_ids={ctx_id})
                          
        if self.edge_mask_ratio > 0.0:
            print(f"\n⚠️ [Ablation Study] Experiment triggered: randomly dropped {dropped_count}/{total_explicit} ({dropped_count/total_explicit*100:.1f}%) explicit edges!\n")

        # 2. Attach implicit relationships (Implicit Edges - co-occurrence)
        if proximity_df is not None and not proximity_df.empty:
            df_p = proximity_df.copy()
            df_p['node_1'] = df_p['node_1'].astype(str)
            df_p['node_2'] = df_p['node_2'].astype(str)
            
            # ✅ New: recognize context_id passed from upstream
            has_implicit_ctx = 'context_id' in df_p.columns
            if has_implicit_ctx:
                df_p['context_id'] = df_p['context_id'].astype(str)
            
            max_count = df_p['count'].max() + 1e-5
            for row in df_p.itertuples(index=False):
                u, v = row.node_1, row.node_2
                norm_count = np.log1p(row.count) / np.log1p(max_count)
                
                # ✅ Extract the context_id specific to this implicit edge
                ctx_id = row.context_id if has_implicit_ctx and pd.notnull(row.context_id) else "-1"
                
                # ✅ Parse comma‑separated chunk_id list spliced by upstream lambda (e.g., "1037,1038")
                chunks = []
                if hasattr(row, 'chunk_id') and pd.notnull(row.chunk_id):
                    chunks = [c.strip() for c in str(row.chunk_id).split(',') if c.strip()]
                
                if G.has_edge(u, v):
                    # If edge already exists (either an explicit edge previously attached, or an implicit edge built by another context)
                    # Implicit score takes the maximum of the two
                    G[u][v]['implicit_score'] = max(G[u][v].get('implicit_score', 0.0), norm_count)
                    G[u][v]['has_implicit'] = True
                    G[u][v]['context_ids'].add(ctx_id) # 👈 Core: inject the current context_id into the edge's pass
                    if chunks:
                        G[u][v].setdefault('chunk_ids', []).extend(chunks)
                else:
                    # Brand new edge
                    G.add_edge(u, v, 
                               type='implicit', 
                               implicit_score=norm_count,
                               has_implicit=True,
                               relation="co-occurs with",
                               context_ids={ctx_id}, # 👈 Core: assign exclusive context_id at initialization
                               chunk_ids=chunks)
        
        print("   ✅ Graph topology built (All IDs unified to string).")   
        return G
    
    def _get_top_chunks(self, candidate_cids, query_vec, top_k=5, min_score=0.25):
        """Filter and rank candidate text chunks by cosine similarity"""
        valid_cids = [str(c) for c in candidate_cids if str(c) in self.chunk_df.index]
        valid_cids = list(dict.fromkeys(valid_cids))
        
        if not valid_cids or query_vec is None: 
            return valid_cids[:top_k]

        try:
            content_matrix = np.stack(self.chunk_df.loc[valid_cids, 'embedding_np'].values)
            
            if 'title_embedding_np' in self.chunk_df.columns:
                title_matrix = np.stack(self.chunk_df.loc[valid_cids, 'title_embedding_np'].values)
                t_norms = np.linalg.norm(title_matrix, axis=1)
                q_norm = np.linalg.norm(query_vec)
                sim_title = (title_matrix @ query_vec) / (t_norms * q_norm + 1e-9)
            else:
                sim_title = 0.0
            
            q_norm = np.linalg.norm(query_vec)
            c_norms = np.linalg.norm(content_matrix, axis=1)
            sim_content = (content_matrix @ query_vec) / (c_norms * q_norm + 1e-9)
            
            final_scores = 0.4 * sim_title + 0.6 * sim_content
            sorted_indices = np.argsort(final_scores)[::-1]
            passing_indices = np.where(final_scores >= min_score)[0]
            
            if len(passing_indices) == 0:
                return []
            
            sorted_passing = [i for i in sorted_indices if i in passing_indices]
            return [valid_cids[i] for i in sorted_passing[:top_k]]
            
        except Exception:
            return valid_cids[:top_k]

    def step1_analyze_intent(self, query, query_vec):
        """Intent analysis: reuse the global query vector"""
        with self.gpu_lock:
            try:
                if query_vec is None:
                    raise ValueError("Query vector is None!")
                
                emb_tensor = torch.tensor(np.array([query_vec]), dtype=torch.float32).to(self.device)
                probs = self.intent_model.predict_proba(emb_tensor, [query])
                weights, strategy = dynamic_weight_modulation(probs, query)
                return weights, strategy
            except Exception as e:
                print(f"⚠️ Intent analysis error: {e}")
                return [0.15, 0.40, 0.45], "Default (Error Fallback)" 

    def step2_semantic_anchoring(self, query, context_id, query_vec=None, top_k=10):
        """Semantic anchoring: find entities and starting points relevant to the query"""
        with self.gpu_lock:
            df = self.matcher.link(query, str(context_id), top_k=top_k)
            raw_seeds = df['Standard_Entity'].tolist() if not df.empty else []
        
        if not raw_seeds: return []

        details, _ = self._get_node_details(raw_seeds, context_id, query_vec=None, add_chunks=False)        
        if not details: return raw_seeds

        indexed_candidates = []
        candidate_names = list(details.keys()) 
        
        for i, name in enumerate(candidate_names):
            desc = details[name].replace('\n', ' ')
            indexed_candidates.append(f"ID {i}: {name} (Info: {desc})")
        
        candidates_txt = "\n".join(indexed_candidates)
        
        prompt = f"""
        You are an expert Entity Linker for a Knowledge Graph Reasoning system.
        Your goal is to identify all **Relevant Entities** that could serve as starting points or key evidence to answer the query.

        Query: "{query}"

        Candidate Entities (with definitions):
        {candidates_txt}
        ### Task
        Select entities that are **Useful Starting Points** to navigate the graph and answer the query.
        
        ### ⚖️ Selection Criteria 
        1. **Direct Matches**: The entity appears in the query or is a synonym (KEEP).
        2. **Relevant Concepts**: Entities that serve as starting points or key evidence to answer the query. (KEEP). 
        3. **Key Concepts**: Entities that are central to the topic and  help to answer the query (KEEP).
        ### Output Format:
        Return ONLY the IDs of the selected entities, separated by commas. Do not explain.
        Example: 0, 2, 5
        """

        resp = self.llm_filter.invoke(prompt).content.strip()
        selected_indices = [int(x) for x in re.findall(r'\d+', resp)]
            
        filtered_seeds = []
        for idx in selected_indices:
            if 0 <= idx < len(candidate_names):
                filtered_seeds.append(candidate_names[idx])
            
        return filtered_seeds if filtered_seeds else raw_seeds[:5]

    def step3_iterative_agent_reasoning(self, seeds, query, context_id, intent_weights, query_vec, max_hops=3, verbose=True, max_prompt_chunks=TokenConfig.MAX_CHUNKS_IN_PROMPT):
        """Core reasoning Agent: iteratively walk on the knowledge graph, collect evidence, until exit condition is met"""
        def log(msg, color=Colors.ENDC):
            if verbose: print(f"{color}{msg}{Colors.ENDC}")

        # ✅ Defense: convert context_id to string for usage
        ctx_id_str = str(context_id)

        relevant_entities = set()       
        accumulated_chunk_texts = set() 
        history_facts = set()           
        visited_nodes = set()           
        entity_memory = {} 

        if verbose:
            log(f"\n{'='*60}", Colors.HEADER)
            log(f"🧠 [Agent Start] Query: {query}", Colors.BOLD)

        # --- Stage 0: Initial verification layer ---
        log(f"📍 [Stage 0] Analyzing Initial Seeds...", Colors.BLUE)
        seed_infos, seed_chunks = self._get_node_details(seeds, ctx_id_str, query_vec=query_vec, add_chunks=True)
        for node, desc in seed_infos.items():
            entity_memory[node] = desc
        
        if verbose:
            for n, desc in seed_infos.items():
                log(f"   - Entity: {n} | {desc[:150]}...", Colors.CYAN)
                
        for txt in seed_chunks: accumulated_chunk_texts.add(txt)

        prompt_0 = self._build_agent_prompt(
            query=query, stage="checking_seeds", known_evidence=list(relevant_entities), 
            current_focus_content=seed_infos, related_chunks=seed_chunks, valid_next_hops=seeds
        )
        decision_0 = self.llm.invoke(prompt_0).content
        parsed_0 = self._parse_llm_decision(decision_0, valid_scope=None) 
        log(f" 💭 [LLM Decision]: {parsed_0}", Colors.GREEN)

        if parsed_0['is_final']:
            return parsed_0['answer'], "Agent-Zero-Shot"

        relevant_nodes_step = parsed_0['relevant_nodes']
        if not relevant_nodes_step: relevant_nodes_step = seeds 
        relevant_entities.update(relevant_nodes_step)
        
        active_nodes = []
        for n in parsed_0['next_nodes']:
            if n in self.G: active_nodes.append(n)
        if not active_nodes:
            active_nodes = [n for n in relevant_nodes_step if n in self.G]

        # --- Stage N: Multi‑hop reasoning layer (Graph Reasoning) ---
        for hop in range(1, max_hops + 1):
            visited_nodes.update(active_nodes)
            log(f"\n📍 [Stage {hop}] Expanding from {len(active_nodes)} nodes...", Colors.BLUE)
            
            if not active_nodes: break

            candidate_paths = self._expand_neighbors(
                active_nodes, query_vec, ctx_id_str, intent_weights, 
                top_k_per_node=TokenConfig.TOP_K_NEIGHBORS, 
                verbose=verbose,
                visited_set=visited_nodes
            )

            if not candidate_paths:
                log("   🛑 No neighbors found.", Colors.WARNING)
                break

            path_strings = [f"{p['u']} --[{p['rel']}]--> {p['v']}" for p in candidate_paths]
            valid_next_hop_candidates = list(set([p['v'] for p in candidate_paths if p['v'] not in visited_nodes]))

            structure_chunk_ids = set()
            for p in candidate_paths:
                if self.G.has_edge(p['u'], p['v']):
                    edge_data = self.G[p['u']][p['v']]
                    edge_ctxs = edge_data.get('context_ids', set())
                    if ctx_id_str in edge_ctxs:
                        if 'chunk_ids' in edge_data:
                            structure_chunk_ids.update(edge_data['chunk_ids'])
                        
            struct_limit = min(3, max_prompt_chunks)
            if struct_limit > 0:
                filtered_struct_cids = self._get_top_chunks(
                    list(structure_chunk_ids), 
                    query_vec, 
                    top_k=struct_limit, 
                    min_score=TokenConfig.CHUNK_SIM_THRESHOLD_LOOSE
                )
            else:
                filtered_struct_cids = []

            current_focus_nodes = set(relevant_entities) | set(valid_next_hop_candidates) | set(active_nodes)
            semantic_pool_ids = set()
            for node in current_focus_nodes:
                if node in self.G:
                    for nbr in self.G.neighbors(node):
                        edge_data = self.G[node][nbr] 
                        edge_ctxs = edge_data.get('context_ids', set())
                        if ctx_id_str in edge_ctxs:
                            semantic_pool_ids.update(edge_data.get('chunk_ids', []))
            
            # ✅ Defense: double set deduplication mechanism
            semantic_pool_ids = {str(c) for c in semantic_pool_ids}
            struct_cids_set = {str(c) for c in filtered_struct_cids}
            semantic_pool_ids = semantic_pool_ids - struct_cids_set
            
            remaining_slots = max_prompt_chunks - len(filtered_struct_cids)            
            filtered_sem_cids = []
            if remaining_slots > 0 and semantic_pool_ids:
                filtered_sem_cids = self._get_top_chunks(
                    list(semantic_pool_ids), 
                    query_vec, 
                    top_k=remaining_slots,
                    min_score=TokenConfig.CHUNK_SIM_THRESHOLD_STRICT 
                )

            final_chunks_for_prompt = []
            used_cids_in_prompt = set()
            
            for cid in filtered_struct_cids:
                cid_str = str(cid)
                if cid_str not in used_cids_in_prompt:
                    used_cids_in_prompt.add(cid_str)
                    txt = self._get_chunk_text(cid_str)
                    if txt:
                        clean_txt = txt[:TokenConfig.CHUNK_CHAR_LIMIT].replace('\n', ' ')
                        final_chunks_for_prompt.append(f"[Path Evidence {cid_str}] {clean_txt}...")
                        accumulated_chunk_texts.add(f"[Ref {cid_str}] {clean_txt}...")

            for cid in filtered_sem_cids:
                cid_str = str(cid)
                if cid_str not in used_cids_in_prompt:
                    used_cids_in_prompt.add(cid_str)
                    txt = self._get_chunk_text(cid_str)
                    if txt:
                        clean_txt = txt[:TokenConfig.CHUNK_CHAR_LIMIT].replace('\n', ' ')
                        final_chunks_for_prompt.append(f"[Context {cid_str}] {clean_txt}...")
                        accumulated_chunk_texts.add(f"[Ref {cid_str}] {clean_txt}...")

            if verbose:
                log(f"   📝 Context: {len(filtered_struct_cids)} Path Chunks, {len(filtered_sem_cids)} Context Chunks added.")

            nodes_to_display = set(relevant_entities) | set(active_nodes)
            evidence_with_desc = []
            for node in nodes_to_display:
                if node in entity_memory:
                    evidence_with_desc.append(f"**{node}**: {entity_memory[node]}...") 
                else:
                    evidence_with_desc.append(node)

            # ✅ Fix: inject previously discovered historical paths into memory
            prompt_n = self._build_agent_prompt(
                query=query, stage="stage_n", known_evidence=evidence_with_desc,
                current_focus_content=path_strings + list(history_facts),
                related_chunks=final_chunks_for_prompt,
                valid_next_hops=valid_next_hop_candidates
            )
            
            decision_n = self.llm.invoke(prompt_n).content
            parsed_n = self._parse_llm_decision(decision_n, valid_scope=valid_next_hop_candidates)
            log(f" 💭 [LLM Decision]: {parsed_n}", Colors.GREEN)

            if parsed_n['is_final']:
                return parsed_n['answer'], f"Agent-Hop-{hop}"

            relevant_entities.update(parsed_n['relevant_nodes'])
            nodes_to_check = set(parsed_n['relevant_nodes']) | set(parsed_n['next_nodes'])
            unknown_nodes = [n for n in nodes_to_check if n not in entity_memory]
            if unknown_nodes:
                new_defs, _ = self._get_node_details(unknown_nodes, ctx_id_str, query_vec=query_vec, add_chunks=False)
                for node_name, node_desc in new_defs.items():
                    entity_memory[node_name] = node_desc

            for p in candidate_paths:
                if p['v'] in parsed_n['relevant_nodes']:
                    history_facts.add(f"{p['u']} {p['rel']} {p['v']}")
            
            next_targets = []
            for n in parsed_n['next_nodes']:
                if n in valid_next_hop_candidates or (n in self.G and n not in visited_nodes):
                    next_targets.append(n)
            
            if not next_targets and valid_next_hop_candidates:
                 next_targets = valid_next_hop_candidates[:2]
            
            active_nodes = next_targets
            log(f"   📌 Relevant Update: {list(relevant_entities)}", Colors.CYAN)
            log(f"   🚀 Next Hop: {active_nodes}", Colors.WARNING)

        return self._fallback_answer(query, context_id=ctx_id_str, query_vec=query_vec, seed_infos=seed_infos, verbose=verbose), "Fallback"

    def _build_agent_prompt(self, query, stage, known_evidence, current_focus_content, related_chunks, valid_next_hops):
        """Format Agent instruction prompt"""
        evidence_str = "\n".join(known_evidence) if known_evidence else "None"
        chunks_str = "\n".join(related_chunks) if related_chunks else "None"
        
        limit_pool = TokenConfig.MAX_CANDIDATE_POOL
        valid_hops_str = ", ".join(valid_next_hops[:limit_pool]) 
        if len(valid_next_hops) > limit_pool: valid_hops_str += ", ..."

        if stage == "checking_seeds":
            def_str = "\n".join([f"- **{k}**: {v}" for k, v in current_focus_content.items()])
            prompt = f"""You are a Fact-Checking & Answer Extraction Agent. Your goal is to answer the query IMMEDIATELY if the information exists in the definitions or Context.

### User Query
"{query}"

### 1. Entity Definitions
{def_str}

### 2. Source Context
{chunks_str}

### 3. Valid Next Hops
[{valid_hops_str}]

### 🧠 DECISION LOGIC (STRICT)
1. **DEDUCE**: Can the answer be derived from the Evidence?
2. **DECIDE**:
- **YES** ->select Scenario A to Output `Final Answer`. IMMEDIATELY
- **NO** ->select Scenario B to find the answer in the graph. **NEVER say "Not Found".**
    
### Output Format
Scenario A. If Answerable (Answer Found):
`Final Answer: [Clean Entity Name / Yes / No / data / etc.]` (Precise and Concise)

Scenario B. If Not Answerable (Answer not Found):
`Relevant Nodes: [...]` (Select useful entities found in Entity Definitions,separate with semicolons e.g., EntityA; EntityB)
`Next Hop: [...]` (Select 1-3 useful nodes from 'Valid Next Hops' to explore graph,separate with semicolons e.g., EntityA; EntityB)

### ✅ POSITIVE INSTRUCTIONS
- **ALWAYS** output ONLY in one of the two specified formats: Scenario A or Scenario B
- **ALWAYS** keep output minimal - just the required lines with no explanations

### ⛔ OUTPUT RESTRICTIONS
- **NO** sentences or paragraphs
- **NO** explanations or reasoning
"""
            # print("checking_seeds:", prompt)
            return prompt

        else:
            focus_str = "\n".join([f"- {s}" for s in current_focus_content])
            prompt = f"""You are an intelligent Graph Reasoning Agent.

### User Query
"{query}"

### 1. Entity Definitions (Secondary Source)
{evidence_str}

### 2. New Graph Paths & Historical Facts
{focus_str}

### 3. Context (PRIMARY SOURCE - Check First!)
{chunks_str}

### 4. Valid Candidates for Next Hop
[{valid_hops_str}]

### 🧠 DECISION LOGIC
1. **DEDUCE**: Can the answer be fully derived from the Evidence?
2. **DECIDE**:
- **YES** -> select Scenario A to Output `Final Answer`. IMMEDIATELY
- **NO** -> select Scenario B to continue searching. **NEVER say "Not Found".**
    
### Output Format
Scenario A. If Answerable (Answer Found):
`Final Answer: [Clean Entity Name / Yes / No / data / etc.]` (Precise and Concise)

Scenario B. If Not Answerable (Answer not Found):
`Relevant Nodes: [...]` (Select useful entities found in Entity Definitions,separate with semicolons e.g., EntityA; EntityB)
`Next Hop: [...]` (Select 1-3 useful nodes from 'Valid Next Hops' to explore graph,separate with semicolons e.g., EntityA; EntityB)

### ✅ POSITIVE INSTRUCTIONS
- **ALWAYS** output ONLY in one of the two specified formats: Scenario A or Scenario B
- **ALWAYS** keep output minimal - just the required lines with no explanations

### ⛔ OUTPUT RESTRICTIONS
- **NO** full sentences or paragraphs
- **NO** explanations or reasoning
"""
            # print("stage_n:", prompt)
            return prompt

    def _parse_llm_decision(self, text, valid_scope=None):
        """Robust regex parsing and format fallback, including false positive answer interception"""
        text = str(text).strip()
        result = {"is_final": False, "answer": "", "relevant_nodes": [], "next_nodes": []}

        def extract_list_robust(label):
            candidates = []
            pattern_strict = re.search(fr"{label}\s*\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
            pattern_loose = re.search(fr"{label}\s*(.+?)(\n|$|Relevant|Next|Final)", text, re.IGNORECASE)

            content = ""
            if pattern_strict:
                content = pattern_strict.group(1)
            elif pattern_loose:
                content = pattern_loose.group(1)
            
            if content:
                content = content.replace('[', '').replace(']', '')
                raw_items = re.split(r'[;\n]', content)                
                for x in raw_items:
                    clean = x.strip().strip("'").strip('"').strip('-').strip()
                    if clean: candidates.append(clean)
            return candidates

        raw_relevant = extract_list_robust("Relevant Nodes:")
        raw_next = extract_list_robust("Next Hop:")

        if "Final Answer:" in text:
            raw_ans = text.split("Final Answer:")[-1].strip()
            stop_tokens = ["\n\n", "Relevant Nodes:", "Next Hop:", "If Not Answerable", "###"]
            for token in stop_tokens:
                if token in raw_ans:
                    raw_ans = raw_ans.split(token)[0]
            
            clean_ans = raw_ans.replace("**", "").replace("__", "").strip().strip('`').strip('"').strip("'")
            if clean_ans.endswith('.'): clean_ans = clean_ans[:-1]
            
            negative_patterns = [
                "not found", "no information", "information is missing", 
                "cannot answer", "unable to answer", "doesn't mention", 
                "not provided", "n/a", "cannot", "not specify", "no specify", "not specified"
            ]
            is_negative = any(pat in clean_ans.lower() for pat in negative_patterns)
            
            if not is_negative:
                lower_ans = clean_ans.lower()
                if lower_ans.startswith("yes") and (len(lower_ans) == 3 or not lower_ans[3].isalnum()):
                    clean_ans = "yes"
                elif lower_ans.startswith("no") and (len(lower_ans) == 2 or not lower_ans[2].isalnum()):
                    clean_ans = "no"

                result["is_final"] = True
                result["answer"] = clean_ans
                return result
            else:
                result["is_final"] = False
                if valid_scope:
                    if not result["relevant_nodes"]:
                         result["relevant_nodes"] = valid_scope[:]
                    if not result["next_nodes"]:
                         result["next_nodes"] = valid_scope[:3]
        
        def validate_and_correct(raw_nodes, scope):
            if not scope: return raw_nodes 
            validated = []
            scope_map = {s.lower(): s for s in scope}
            for node in raw_nodes:
                if node in scope:
                    validated.append(node)
                    continue
                if node.lower() in scope_map:
                    validated.append(scope_map[node.lower()])
                    continue
                matches = difflib.get_close_matches(node, scope, n=1, cutoff=0.7)
                if matches:
                    validated.append(matches[0])
            return list(set(validated))

        result["relevant_nodes"] = raw_relevant 
        result["next_nodes"] = validate_and_correct(raw_next, valid_scope)

        if not result["is_final"] and not result["next_nodes"] and valid_scope:
            result["next_nodes"] = valid_scope[:3]

        return result
    
    def _fallback_answer(self, query, context_id, query_vec, seed_infos={}, verbose=False):
        """Industrial‑grade hybrid retrieval fallback layer: when graph reasoning breaks, use BM25 (lexical) + dense vectors to compute hybrid scores for extraction"""
        if verbose:
            print(f"{Colors.WARNING}⚠️ [Fallback] Switching to BM25+Vector Hybrid RAG...{Colors.ENDC}")

        candidate_cids = []
        ctx_id_str = str(context_id)
        if ctx_id_str in self.chunk_dict_by_ctx:
            candidate_cids = self.chunk_dict_by_ctx[ctx_id_str]
        
        top_chunks_text = []
        if candidate_cids:
            vector_top_cids = self._get_top_chunks(
                candidate_cids, query_vec, top_k=len(candidate_cids), min_score=0.15 
            )
            vec_score_map = {cid: (len(vector_top_cids) - idx) for idx, cid in enumerate(vector_top_cids)}
            
            corpus_cids = []
            tokenized_corpus = []
            
            for cid in candidate_cids:
                txt = self._get_chunk_text(cid)
                if txt:
                    corpus_cids.append(cid)
                    tokens = re.findall(r'\w+', txt.lower())
                    tokenized_corpus.append(tokens)
                    
            if tokenized_corpus:
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = re.findall(r'\w+', query.lower())
                bm25_scores = bm25.get_scores(tokenized_query)
                
                max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
                norm_bm25_scores = [s / max_bm25 for s in bm25_scores]
                
                hybrid_scores = []
                for idx, cid in enumerate(corpus_cids):
                    s_bm25 = norm_bm25_scores[idx]
                    s_vec = vec_score_map.get(cid, 0) / (len(candidate_cids) + 1e-5)
                    final_score = (0.3 * s_bm25) + (0.7 * s_vec)
                    hybrid_scores.append((cid, final_score))
                
                hybrid_scores.sort(key=lambda x: x[1], reverse=True)
                for cid, score in hybrid_scores[:3]:
                    txt = self._get_chunk_text(cid)
                    clean_txt = txt[:TokenConfig.CHUNK_CHAR_LIMIT].replace('\n', ' ')
                    top_chunks_text.append(f"[Ref {cid}] {clean_txt}")
                
        seeds_str = "\n".join([f"- **{k}**: {v}" for k, v in seed_infos.items()])
        context_str = "\n".join(top_chunks_text) if top_chunks_text else "No specific context found."

        prompt = f"""You are a high-precision QA system answering a complex question. The primary reasoning path was broken, so you must answer based DIRECTLY on the provided Reference Text and Entity Definitions.

### User Query
"{query}"

### 1. Key Entity Definitions (Background Info)
{seeds_str}

### 2. Reference Context (Primary Evidence)
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
        resp = self.llm.invoke(prompt).content.strip()
        if "Final Answer:" in resp:
            resp = resp.split("Final Answer:")[-1].strip()
        
        return resp

    def _expand_neighbors(self, source_nodes, query_vec, context_id, intent_weights, top_k_per_node=None, verbose=False, visited_set=None):
        """Look outward at neighbor nodes, comprehensively consider edge type, implicit score, and semantic vectors, score and return Top K optimal hops"""
        w_f, w_s, w_e = intent_weights
        k = top_k_per_node if top_k_per_node else TokenConfig.TOP_K_NEIGHBORS
        query_norm = np.linalg.norm(query_vec) if query_vec is not None else 1.0

        all_potential_neighbors = set()
        for u in source_nodes:
            if u in self.G:
                neighbors = [v for v in self.G.neighbors(u) if not (visited_set and v in visited_set)]
                all_potential_neighbors.update(neighbors)
        
        unique_v_list = list(all_potential_neighbors)
        sim_map = {} 

        if unique_v_list and query_vec is not None:
            valid_vecs_ent = []
            valid_vecs_desc = []
            valid_v_names = []
            
            for v in unique_v_list:
                if v in self.node_to_vec_idx:
                    idx = self.node_to_vec_idx[v]
                    
                    vec_e = self.matcher.matrix_entity[idx]
                    vec_d = self.matcher.matrix_desc[idx]
                    
                    valid_vecs_ent.append(vec_e)
                    valid_vecs_desc.append(vec_d) 
                    valid_v_names.append(v)
            
            if valid_vecs_ent:
                vec_matrix_ent = np.stack(valid_vecs_ent)  
                dot_products_ent = vec_matrix_ent @ query_vec
                norms_ent = np.linalg.norm(vec_matrix_ent, axis=1)
                sim_entity = dot_products_ent / (norms_ent * query_norm + 1e-9)
                
                vec_matrix_desc = np.stack(valid_vecs_desc)
                dot_products_desc = vec_matrix_desc @ query_vec
                norms_desc = np.linalg.norm(vec_matrix_desc, axis=1)
                sim_desc = dot_products_desc / (norms_desc * query_norm + 1e-9)
                
                cosine_sims = (0.4 * sim_entity) + (0.6 * sim_desc)
                sim_map = dict(zip(valid_v_names, np.maximum(0, cosine_sims)))

        candidate_paths = []
        ctx_id_str = str(context_id) # ✅ defensive string conversion
        
        for u in source_nodes:
            if u not in self.G: continue
            neighbors_scores = []
            
            for v in self.G.neighbors(u):
                if visited_set and v in visited_set: continue
                
                data = self.G[u][v]
                s_sem = sim_map.get(v, 0.0) 
                s_imp = data.get('implicit_score', 0.0)
                is_explicit = (data.get('type') == 'explicit')
                W = 0.0
                
                if is_explicit:
                    W = w_e * 0.5 + (w_s * s_sem * 1.0) 
                else:
                    if w_f == 0.0 and w_s == 0.0:
                        W = 0.0
                    elif s_sem > 0.60 or (s_imp > 0.35 and s_sem > 0.35):
                        base_leap = w_e * 0.30
                        W = base_leap + (w_f * s_imp) + (w_s * s_sem)
                    else:
                        W = 0.0 
                        
                edge_ctxs = data.get('context_ids', set())
                in_context = ctx_id_str in edge_ctxs # ✅ use string matching
                final_score = W * (1.0 if in_context else 0.0)

                if final_score < TokenConfig.MIN_EDGE_SCORE: continue
                neighbors_scores.append((v, final_score, data))

            neighbors_scores.sort(key=lambda x: x[1], reverse=True)
            for v, score, data in neighbors_scores[:k]:
                candidate_paths.append({
                    'u': u, 'v': v, 
                    'rel': data.get('relation', 'related_to'),
                    'score': score
                })
        
        return candidate_paths
        
    def _get_node_details(self, nodes, context_id, query_vec=None, add_chunks=True):
        details = {}
        all_candidate_chunks = set()
        entity_df = self.matcher.df.set_index('Standard_Entity')
        
        ctx_id_str = str(context_id) # ✅ safely convert to string
            
        for n in nodes:
            has_def = False
            if n in entity_df.index:
                try:
                    row = entity_df.loc[n]
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    raw_desc = str(row.get('description', '')).replace('\n', ' ')
                    category = str(row.get('category', 'N/A')).replace('\n', ' ')
                    synonyms = str(row.get('synonyms', '')).replace('\n', ' ')
                    details[n] = f"{raw_desc}, [Category: {category}], [Synonyms: {synonyms}]"
                    if len(raw_desc.strip()) > 0: has_def = True
                except: pass

            if add_chunks and ((not has_def) or TokenConfig.STAGE0_ADD_CHUNKS):
                if n in self.G:
                    for nbr in self.G.neighbors(n):
                        edge_data = self.G[n][nbr]
                        edge_ctxs = edge_data.get('context_ids', set())
                        
                        if ctx_id_str in edge_ctxs: # ✅ use string matching
                            all_candidate_chunks.update(edge_data.get('chunk_ids', []))

        chunks = []
        if add_chunks and all_candidate_chunks:
            limit = TokenConfig.STAGE0_MAX_CHUNKS
            
            best_cids = self._get_top_chunks(
                list(all_candidate_chunks), 
                query_vec, 
                top_k=limit, 
                min_score=TokenConfig.CHUNK_SIM_THRESHOLD_LOOSE
            )
            
            for cid in best_cids:
                txt = self._get_chunk_text(cid)
                if txt:
                    clean_txt = txt[:TokenConfig.CHUNK_CHAR_LIMIT]
                    chunks.append(f"[Ref {cid}] {clean_txt}")
            
        return details, chunks
    
    def _get_chunk_text(self, chunk_id):
        try:
            res = self.chunk_df.loc[str(chunk_id), 'text']
            if isinstance(res, pd.Series):
                res = res.iloc[0]
            return str(res).replace('\n', ' ')
        except: 
            return ""
    
    def solve(self, query, context_id, verbose=True, mode='full'):
        """Main entry point exposed to external callers"""
        stage_n_chunks = TokenConfig.MAX_CHUNKS_IN_PROMPT
        query_vec = None
        ctx_id_str = str(context_id)

        with self.gpu_lock:
            try:
                query_vec = self.graph_embed_model.embed_query(query)
                query_vec = np.array(query_vec, dtype=np.float32)
            except Exception as e: 
                print(f"⚠️ Query embedding failed: {e}")
                
        if mode == 'vector_only':
            return self._fallback_answer(query, context_id, query_vec, verbose=verbose), "Vector-RAG"

        weights, strategy = self.step1_analyze_intent(query, query_vec)
        seeds = self.step2_semantic_anchoring(query, context_id) 
        
        if not seeds: 
            return "Sorry, no relevant entities were found in the knowledge graph.", strategy

        if mode == 'explicit_only':
            weights = [0.0, 0.0, 1.0] 
        
        answer, final_stage_tag = self.step3_iterative_agent_reasoning(
            seeds, query, ctx_id_str, 
            intent_weights=weights, 
            max_hops=3, 
            verbose=verbose,
            query_vec=query_vec,
            max_prompt_chunks=stage_n_chunks
        )
        
        return answer, f"{strategy} -> {final_stage_tag}"
    
# ==========================================
# 4. Main program entry (testing and execution)
# ==========================================
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data_output", "dataset", "2wiki", "ds1000")
    ADAPT_ROOT = os.path.join(PROJECT_ROOT, "adapt")
    
    QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
    GRAPH_FILE = os.path.join(DATA_ROOT, "graph.csv")
    CHUNK_EMB_FILE = os.path.join(DATA_ROOT, "chunks_with_embeddings.parquet")
    CHUNK_RAW_FILE = os.path.join(DATA_ROOT, "chunk.csv")
    PARQUET_FILE = os.path.join(DATA_ROOT, "concepts_merged_with_vectors.parquet")
    PROX_FILE = os.path.join(DATA_ROOT, "contextual_proximity.csv")
    MODEL_FILE = os.path.join(ADAPT_ROOT, "intent_classifier_struct.pth")
    
    # 1. Data loading
    try:
        df_qa = pd.read_csv(QA_FILE, sep="|")
        df_graph = pd.read_csv(GRAPH_FILE, sep="|")
        
        if os.path.exists(CHUNK_EMB_FILE):
            print(f"📦 Loading chunk data with vectors: {CHUNK_EMB_FILE}")
            df_chunk = pd.read_parquet(CHUNK_EMB_FILE)
        else:
            print(f"⚠️ Embedding Parquet not found, loading raw CSV: {CHUNK_RAW_FILE}")
            df_chunk = pd.read_csv(CHUNK_RAW_FILE, sep="|")
        
        if os.path.exists(PROX_FILE):
            df_prox = pd.read_csv(PROX_FILE, sep="|")
        else:
            df_prox = None
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        sys.exit(1)

    # 2. Initialize engine
    print("🚀 Initializing ID-SGTR engine...")
    MASK_RATIO = 0
    engine = ID_SGTR_Reasoning_Engine(
        intent_model_path=MODEL_FILE,
        parquet_path=PARQUET_FILE,
        graph_df=df_graph,
        chunk_df=df_chunk,
        proximity_df=df_prox,
        edge_mask_ratio=MASK_RATIO  # [pass ratio]
    )


    # --- Mode B: Multi‑threaded batch processing (logs will be messy, suggest verbose=False) ---
    # target_data = df_qa.iloc[[9,42,44,53,59,61,67,70,71,75,78,85]]
    # target_data = df_qa.iloc[[13,20,21,22,26,32,35,39,45,47,55,59,60,61,64,68,70,71,72,77,82,84,87,91,92]]
    target_data = df_qa.head(1000)
    # target_data = df_qa.sample(5)


    
    print(f"\n📝 Starting concurrent processing of {len(target_data)} queries...")
    
    def process_query_wrapper(i: int, row: pd.Series) -> Tuple[int, Any]:
        # [New] force flush print so you can see debug info
        q = row['question']
        ctx = row['context_id']
        gold = row['answer']
        
        pred_answer, strategy = engine.solve(q, ctx, verbose=False, mode='vector_only')
                
        return i, {
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "strategy": strategy,
            "context_id": ctx
        }
        
    processed_results = parallel_llm_processor(
        dataframe=target_data,
        processing_func=process_query_wrapper,
        start_message="Starting multi‑threaded reasoning...",
        max_workers=5,
        max_retries=6,
        initial_delay=2
    )

    processed_results.sort(key=lambda x: x[0])
    final_data = [item[1] for item in processed_results]
    output_path = os.path.join(current_dir, "query_results_agent_1000Qwen3-8B_3_20_B_top3.csv")
    # output_path = os.path.join(current_dir, "query_results_agent_100glm46_2_1.csv")
    # output_path = os.path.join(current_dir, "test.csv")
    pd.DataFrame(final_data).to_csv(output_path, index=False, sep="|")
    print(f"\n✅ Processing completed, results saved to: {output_path}")







# ==========================================
# 4. Main program entry (automated batch run version)
# ==========================================
# if __name__ == "__main__":
#     import gc # For garbage collection to prevent memory leaks
    
#     DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\2wiki\ds1000"
#     ADAPT_ROOT = r"D:\Code\jupyter\knowledge_graph\adapt"
    
#     QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
#     GRAPH_FILE = os.path.join(DATA_ROOT, "graph.csv")
#     CHUNK_EMB_FILE = os.path.join(DATA_ROOT, "chunks_with_embeddings.parquet")
#     CHUNK_RAW_FILE = os.path.join(DATA_ROOT, "chunk.csv")
#     PARQUET_FILE = os.path.join(DATA_ROOT, "concepts_merged_with_vectors.parquet")
#     PROX_FILE = os.path.join(DATA_ROOT, "contextual_proximity.csv")
#     MODEL_FILE = os.path.join(ADAPT_ROOT, "intent_classifier_struct.pth")

#     # 1. Base data loading (only once)
#     try:
#         df_qa = pd.read_csv(QA_FILE, sep="|")
#         df_graph = pd.read_csv(GRAPH_FILE, sep="|")
        
#         if os.path.exists(CHUNK_EMB_FILE):
#             print(f"📦 Loading chunk data with vectors: {CHUNK_EMB_FILE}")
#             df_chunk = pd.read_parquet(CHUNK_EMB_FILE)
#         else:
#             print(f"⚠️ Embedding Parquet not found, loading raw CSV: {CHUNK_RAW_FILE}")
#             df_chunk = pd.read_csv(CHUNK_RAW_FILE, sep="|")
        
#         if os.path.exists(PROX_FILE):
#             df_prox = pd.read_csv(PROX_FILE, sep="|")
#         else:
#             df_prox = None
#     except Exception as e:
#         print(f"❌ Data loading failed: {e}")
#         sys.exit(1)

#     # 2. Base engine initialization (loads LLM and embedding model, only once)
#     print("🚀 Initializing core components of ID-SGTR engine...")
#     engine = ID_SGTR_Reasoning_Engine(
#         intent_model_path=MODEL_FILE,
#         parquet_path=PARQUET_FILE,
#         graph_df=df_graph,
#         chunk_df=df_chunk,
#         proximity_df=df_prox,
#         edge_mask_ratio=0.0  # initially set to 0
#     )

#     # =====================================================================
#     # 🌟 Automated evaluation configuration area
#     # =====================================================================
#     target_data = df_qa.head(1000)  # test dataset range
    
#     # Remember to run two sets of comparative experiments!
#     # First run: set to 'explicit_only' (test Baseline)
#     # Second run: set to 'full' (test our Hybrid algorithm)
#     RUN_MODE = 'explicit_only' 
    
#     # Automatically iterate over 5 mask ratios
#     mask_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
#     # =====================================================================

#     # 3. Start multi‑round automated testing
#     for mask_ratio in mask_ratios:
#         mask_pct = int(mask_ratio * 100)
#         print(f"\n\n{'='*80}")
#         print(f"🔥 [Start round {mask_ratios.index(mask_ratio) + 1}/5] Testing MASK_RATIO = {mask_ratio} ({mask_pct}%) | Mode: {RUN_MODE}")
#         print(f"{'='*80}")
        
#         # [Key optimization] Dynamically rebuild the graph (no need to reload large models, very fast)
#         print("🕸️ Applying MASK to regenerate graph structure...")
#         engine.edge_mask_ratio = mask_ratio
#         engine.G = engine._build_hybrid_graph(df_graph, df_prox)
        
#         def process_query_wrapper(i: int, row: pd.Series) -> Tuple[int, Any]:
#             q = row['question']
#             ctx = row['context_id']
#             gold = row['answer']
            
#             # Use the configured RUN_MODE
#             pred_answer, strategy = engine.solve(q, ctx, verbose=False, mode=RUN_MODE)
                    
#             return i, {
#                 "question": q,
#                 "gold_answer": gold,
#                 "pred_answer": pred_answer,
#                 "strategy": strategy,
#                 "context_id": ctx
#             }
            
#         # Start multi‑threading
#         processed_results = parallel_llm_processor(
#             dataframe=target_data,
#             processing_func=process_query_wrapper,
#             start_message=f"Starting multi‑threaded reasoning (MASK={mask_pct}%)...",
#             max_workers=5,
#             max_retries=6,
#             initial_delay=2
#         )

#         # Assemble and save results
#         processed_results.sort(key=lambda x: x[0])
#         final_data = [item[1] for item in processed_results]
        
#         # Dynamically generate file name
#         file_name = f"query_results_agent_1000Qwen3-8B_3_20_A_{mask_pct}%.csv"
#         output_path = os.path.join(current_dir, file_name)
        
#         pd.DataFrame(final_data).to_csv(output_path, index=False, sep="|")
#         print(f"\n✅ [Round complete] Results saved to: {output_path}")
        
#         # Clean memory for next round
#         del processed_results
#         del final_data
#         gc.collect()

#     print("\n🎉🎉🎉 All 5 rounds of MASK testing finished, you can wrap up!")
