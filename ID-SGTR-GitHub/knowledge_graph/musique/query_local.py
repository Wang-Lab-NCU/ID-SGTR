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
from contextlib import nullcontext
from rank_bm25 import BM25Okapi
import difflib
from typing import List, Tuple, Callable, Any, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 0. Helper class: console color output (for debugging log differentiation)
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
# Controls the size of the context window sent to LLM and similarity cutoff to prevent token explosion or noise
# ==========================================
class TokenConfig:
    # --- Stage 0: initial entity definition checking ---
    STAGE0_ADD_CHUNKS = True          # Whether to forcibly attach relevant text chunks during entity definition stage
    STAGE0_MAX_CHUNKS = 3            # Maximum number of chunks to pass in the initial stage
    
    # --- Stage N: path expansion ---
    TOP_K_NEIGHBORS = 12              # Maximum number of neighbor nodes the agent explores per step in the graph
    CHUNK_SIM_THRESHOLD_STRICT = 0.35 # Cosine similarity threshold for semantically supplemented chunks (high threshold to prevent noise)
    CHUNK_SIM_THRESHOLD_LOOSE = 0.25  # Cosine similarity threshold for structurally associated chunks (lower threshold because graph edges guarantee relevance)
    MIN_EDGE_SCORE = 0.05             # Minimum comprehensive edge weight; edges below this are considered disconnected
    
    # --- General text limits ---
    MAX_CANDIDATE_POOL = 20           # Maximum number of candidate Next Hops to pass to LLM
    CHUNK_CHAR_LIMIT = 1000           # Character truncation length for a single text block (prevents overly long text)
    MAX_CHUNKS_IN_PROMPT = 3          # Maximum total number of chunks allowed when assembling each prompt

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
from experiments.stage0_replay import load_stage0_replay, replay_decision, replay_seeds
from experiments.telemetry import ContextTrackedChatModel, QueryTelemetry, bind_telemetry, record_candidates, record_evidence, record_hop_gate, record_runtime_execution, record_runtime_step, record_seed_selection, record_stage0_gate, record_stage0_policy, record_stage0_seeds, record_terminal_recovery
from experiments.evidence import EvidenceAssembler, EvidenceItem, EvidenceVariant
from experiments.stage0_gate import (
    Stage0GateResult,
    parse_stage0_confidence,
    parse_supporting_refs,
    verify_stage0_answer,
)
from experiments.query_reranker import (
    bind_rerank_query,
    current_rerank_query,
    get_query_reranker,
)
from experiments.routing import parse_selected_ids, rank_seed_candidates, seed_rerank_decision, seed_rerank_prompt
from experiments.terminal_recovery import build_terminal_prompt, build_terminal_query, deduplicate_chunk_ids_by_text, parse_terminal_response, select_terminal_chunks

# ==========================================
# 3. Core engine class: ID-SGTR (supports multi-threading and agent reasoning)
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
                 random_seed=42,
                 evidence_variant="topology_folding",
                 evidence_budget=TokenConfig.MAX_CHUNKS_IN_PROMPT,
                 reasoning_stress=False,
                 allow_stage0_exit=True):
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_mask_ratio = edge_mask_ratio
        self.random_seed = random_seed
        self.evidence_variant = EvidenceVariant(evidence_variant)
        self.evidence_assembler = EvidenceAssembler(budget=evidence_budget, random_seed=random_seed)
        self.reasoning_stress = bool(reasoning_stress)
        self.allow_stage0_exit = bool(allow_stage0_exit)
        self.stage0_replay = load_stage0_replay(
            os.getenv("ID_SGTR_STAGE0_REPLAY_FILE")
        )
        self.reasoning_beam_width = max(
            1, int(os.getenv("ID_SGTR_REASONING_BEAM_WIDTH", "6"))
        )
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
            print(f"� Failed to load intent model: {e}")
        
        # --- Module 2: Load semantic matching module (entity linking) ---
        print("📥 [2/4] Loading semantic anchor database...")
        self.matcher = SemanticMatcher(parquet_path)

        if hasattr(self.matcher, 'embed_model'):
            self.graph_embed_model = self.matcher.embed_model
            print("✅ Reusing SemanticMatcher's embedding model")
        else:
            print("⚠️ Creating new embedding model for graph reasoning")
            self.graph_embed_model = get_embeddings_model(dimensions=1024)

        # --- Module 3: Data preprocessing and knowledge graph construction ---
        print("🕸️ [3/4] Data preprocessing and graph construction (performing defensive ID string conversion)...")
        self.chunk_df = chunk_df.copy()
        
        # ✅ Defensive programming: force convert all input IDs to strings
        if 'context_id' in self.chunk_df.columns:
            self.chunk_df['context_id'] = self.chunk_df['context_id'].astype(str)
        if 'chunk_id' in self.chunk_df.columns:
            self.chunk_df['chunk_id'] = self.chunk_df['chunk_id'].astype(str)
            self.chunk_df.set_index('chunk_id', inplace=True)
            
        self.chunk_df.index = self.chunk_df.index.astype(str)
        
        def parse_vec_safe(x):
            """Safely parse vector: compatible with numpy, list, and string representations of arrays"""
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
            print("   ✅ Detected pre-computed 'title_embedding', loading...")
            self.chunk_df['title_embedding_np'] = self.chunk_df['title_embedding'].apply(parse_vec_safe)
        
        # Build inverted index from context_id to chunk_ids
        self.chunk_dict_by_ctx = {}
        if 'context_id' in self.chunk_df.columns:
            print("   ✅ Building Context-to-Chunk index...")
            self.chunk_dict_by_ctx = self.chunk_df.groupby('context_id')['text'].apply(lambda x: x.index.tolist()).to_dict()        
        
        print("   -> Building Node-to-Matrix index...")
        self.node_to_vec_idx = {
            str(name): idx for idx, name in enumerate(self.matcher.df['Standard_Entity'])
        }

        # Build NetworkX graph
        self.G = self._build_hybrid_graph(graph_df, proximity_df)
        self.query_reranker = get_query_reranker()
        
        # --- Module 4: Load reasoning LLM ---
        print("🤖 [4/4] Initializing reasoning LLM...")
        self.llm_filter = ContextTrackedChatModel(get_chat_model(task_type="reasoning"), role="auxiliary")
        self.llm = ContextTrackedChatModel(get_chat_model(task_type="kg_query"), role="answer")
        self.llm_final_reasoner = ContextTrackedChatModel(get_chat_model(task_type="final_reasoning"), role="answer")
        self.llm_terminal = ContextTrackedChatModel(get_chat_model(task_type="terminal_recovery"), role="answer")
        self.llm_final = ContextTrackedChatModel(get_chat_model(task_type="final_answer"), role="answer")

    def _embedding_guard(self):
        """Serialize local embedding models, but not remote embedding APIs."""
        provider = os.getenv("EMBEDDINGS_MODEL", "").strip().lower()
        return nullcontext() if provider in {"siliconflow", "zhipuai"} else self.gpu_lock

    def _build_hybrid_graph(self, graph_df, proximity_df):
        """Build hybrid graph"""
        G = nx.Graph()
        
        # ✅ Defensive programming: convert all graph input IDs to strings
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

      # 2. Mount implicit relationships (Implicit Edges - co-occurrence)
        if proximity_df is not None and not proximity_df.empty:
            df_p = proximity_df.copy()
            df_p['node_1'] = df_p['node_1'].astype(str)
            df_p['node_2'] = df_p['node_2'].astype(str)
            
            # ✅ New: identify context_id passed from upstream
            has_implicit_ctx = 'context_id' in df_p.columns
            if has_implicit_ctx:
                df_p['context_id'] = df_p['context_id'].astype(str)
            
            max_count = df_p['count'].max() + 1e-5
            for row in df_p.itertuples(index=False):
                u, v = row.node_1, row.node_2
                norm_count = np.log1p(row.count) / np.log1p(max_count)
                
                # ✅ Extract context_id specific to this implicit edge
                ctx_id = row.context_id if has_implicit_ctx and pd.notnull(row.context_id) else "-1"
                
                # ✅ Parse comma-separated chunk_id list concatenated by upstream lambda (e.g., "1037,1038")
                chunks = []
                if hasattr(row, 'chunk_id') and pd.notnull(row.chunk_id):
                    chunks = [c.strip() for c in str(row.chunk_id).split(',') if c.strip()]
                
                if G.has_edge(u, v):
                    # If edge already exists (maybe previously mounted explicit edge, or implicit edge built by another context)
                    # Implicit score takes the maximum of both
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
                               context_ids={ctx_id}, # 👈 Core: assign dedicated context_id at initialization
                               chunk_ids=chunks)
        
        print("   ✅ Graph topology built (All IDs unified to string).")   
        return G
    
    def _rank_chunks(
        self, candidate_cids, query_vec, top_k=5, min_score=0.25,
        rerank_query=None, rerank_base_scores=None, rerank_score_cache=None,
        rerank_cross_weight=None,
    ):
        """Return ``(chunk_id, cosine_score)`` pairs in retrieval-score order."""
        valid_cids = [str(c) for c in candidate_cids if str(c) in self.chunk_df.index]
        valid_cids = list(dict.fromkeys(valid_cids))
        if not valid_cids:
            return []
        if query_vec is None:
            return [(cid, 0.0) for cid in valid_cids[:top_k]]

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

            if self.query_reranker.enabled:
                documents = [self._get_chunk_text(cid) for cid in valid_cids]
                ranking_query = rerank_query or current_rerank_query()
                base_scores = [
                    float(rerank_base_scores.get(cid, final_scores[index]))
                    if rerank_base_scores is not None
                    else float(final_scores[index])
                    for index, cid in enumerate(valid_cids)
                ]
                reranked = self.query_reranker.rank(
                    ranking_query,
                    valid_cids,
                    documents,
                    base_scores=base_scores,
                    cross_weight=rerank_cross_weight,
                )
                if rerank_score_cache is not None:
                    rerank_score_cache.update({
                        item.item_id: float(item.score) for item in reranked
                    })
                return [
                    (item.item_id, item.score)
                    for item in reranked[:top_k]
                ]

            sorted_indices = np.argsort(final_scores)[::-1]
            passing_indices = np.where(final_scores >= min_score)[0]
            
            if len(passing_indices) == 0:
                return []
            
            sorted_passing = [i for i in sorted_indices if i in passing_indices]
            return [(valid_cids[i], float(final_scores[i])) for i in sorted_passing[:top_k]]
            
        except Exception:
            return [(cid, 0.0) for cid in valid_cids[:top_k]]

    def _get_top_chunks(
        self, candidate_cids, query_vec, top_k=5, min_score=0.25,
        rerank_query=None, rerank_base_scores=None, rerank_score_cache=None,
        rerank_cross_weight=None,
    ):
        return [chunk_id for chunk_id, _ in self._rank_chunks(
            candidate_cids, query_vec, top_k, min_score,
            rerank_query=rerank_query,
            rerank_base_scores=rerank_base_scores,
            rerank_score_cache=rerank_score_cache,
            rerank_cross_weight=rerank_cross_weight,
        )]

    def step1_analyze_intent(self, query, query_vec):
        """Intent analysis: reuses global query vector"""
        with self.gpu_lock:
            try:
                if query_vec is None:
                    raise ValueError("Query vector is None!")
                
                emb_tensor = torch.tensor(np.array([query_vec]), dtype=torch.float32).to(self.device)
                probs = self.intent_model.predict_proba(emb_tensor, [query])
                weights, strategy = dynamic_weight_modulation(
                    probs, query, dataset="musique"
                )
                return weights, strategy
            except Exception as e:
                print(f"⚠️ Intent analysis error: {e}")
                return [0.15, 0.40, 0.45], "Default (Error Fallback)" 

    def step2_semantic_anchoring(self, query, context_id, query_vec=None, top_k=20):
        """Shared Top-20 -> adaptive Top-5 semantic seed selection."""
        candidate_limit = max(
            1, int(os.getenv("ID_SGTR_SEED_CANDIDATE_LIMIT", "20"))
        )
        seed_limit = max(1, int(os.getenv("ID_SGTR_SEED_LIMIT", "5")))
        with self._embedding_guard():
            df = self.matcher.link(
                query,
                str(context_id),
                top_k=candidate_limit,
                query_vector=query_vec,
            )
        if df.empty:
            record_seed_selection(
                candidate_count=0, selected_count=0,
                policy="no_candidates", reason="no_candidates", margin=0.0,
                generic_hub_risk=False, reranker_used=False,
            )
            return []

        candidates = rank_seed_candidates(
            df, self.G, query=query, limit=candidate_limit,
            generic_degree_threshold=int(
                os.getenv("ID_SGTR_SEED_GENERIC_DEGREE_THRESHOLD", "50")
            ),
        )
        if not candidates:
            return []
        needs_llm, reason, seed_margin, generic_hub_risk = seed_rerank_decision(
            candidates,
            seed_limit=seed_limit,
            margin_threshold=float(
                os.getenv("ID_SGTR_SEED_MARGIN_THRESHOLD", "0.04")
            ),
        )
        llm_enabled = (
            os.getenv(
                "ID_SGTR_SEED_LLM_RERANK_ENABLED", "true"
            ).strip().lower() in {"1", "true", "yes", "on"}
        )
        selected = [candidate.name for candidate in candidates[:seed_limit]]
        reranker_used = False
        policy = (
            "deterministic_reasoning_stress"
            if self.reasoning_stress else "deterministic_top5"
        )
        if needs_llm and llm_enabled:
            names = [candidate.name for candidate in candidates]
            details, _ = self._get_node_details(
                names, context_id, query_vec=None, add_chunks=False
            )
            response = self.llm_filter.invoke(
                seed_rerank_prompt(
                    query, candidates, details, seed_limit=seed_limit
                )
            )
            selected_ids = parse_selected_ids(
                response, len(candidates), limit=seed_limit
            )
            if selected_ids:
                selected = [candidates[index].name for index in selected_ids]
                reranker_used = True
                policy = "llm_low_confidence"
            else:
                policy = "deterministic_parse_fallback"

        selected = list(dict.fromkeys(selected))[:seed_limit]
        record_seed_selection(
            candidate_count=len(candidates),
            selected_count=len(selected),
            policy=policy,
            reason=reason,
            margin=seed_margin,
            generic_hub_risk=generic_hub_risk,
            reranker_used=reranker_used,
        )
        return selected

    @staticmethod
    def _build_hop_rerank_query(
        query, hop, active_nodes, relevant_entities, history_facts,
        candidate_paths,
    ):
        current = ", ".join(map(str, active_nodes[:6])) or "none"
        relevant = ", ".join(sorted(map(str, relevant_entities))[:8]) or "none"
        history = "; ".join(map(str, history_facts[-8:])) or "none"
        relations = "; ".join(
            f"{path['u']} --[{path['rel']}]--> {path['v']}"
            for path in candidate_paths[:12]
        ) or "none"
        return (
            f"Original question: {query}\nCurrent reasoning hop: {hop}\n"
            f"Current frontier entities: {current}\n"
            f"Relevant entities already found: {relevant}\n"
            f"Known path facts: {history}\nCandidate next relations: {relations}\n"
            "Find the passage that supplies the next missing relation or the "
            "final answer. Prefer evidence that advances the current path, "
            "not passages that merely repeat an earlier hop."
        )

    def step3_iterative_agent_reasoning(self, seeds, query, context_id, intent_weights, query_vec, max_hops=4, verbose=True, max_prompt_chunks=TokenConfig.MAX_CHUNKS_IN_PROMPT, gold_evidence=None, query_id=None):
        """Core reasoning agent: iteratively walks on the knowledge graph, collects evidence until exit condition is met"""
        gold_evidence = {str(value) for value in (gold_evidence or [])}
        def log(msg, color=Colors.ENDC):
            if verbose: print(f"{color}{msg}{Colors.ENDC}")

        # ✅ Defensive: convert context_id to string for use
        ctx_id_str = str(context_id)

        relevant_entities = set()       
        accumulated_chunk_texts = set() 
        history_facts = []
        history_fact_seen = set()
        folded_evidence_items = []
        last_folded_chunks = []
        last_selected_chunk_ids = []
        provisional_answer = ""
        visited_nodes = set()           
        entity_memory = {} 

        if verbose:
            log(f"\n{'='*60}", Colors.HEADER)
            log(f"🧠 [Agent Start] Query: {query}", Colors.BOLD)

        # --- Stage 0: Initial verification layer ---
        log(f"📍 [Stage 0] Analyzing Initial Seeds...", Colors.BLUE)
        original_rerank_scores = {}
        seed_infos, seed_chunks = self._get_node_details(
            seeds, ctx_id_str, query_vec=query_vec,
            add_chunks=not self.reasoning_stress,
            rerank_score_cache=original_rerank_scores,
        )
        for node, desc in seed_infos.items():
            entity_memory[node] = desc
        
        if verbose:
            for n, desc in seed_infos.items():
                log(f"   - Entity: {n} | {desc[:150]}...", Colors.CYAN)
                
        for txt in seed_chunks: accumulated_chunk_texts.add(txt)
        stage0_items = []
        for position, chunk in enumerate(seed_chunks, start=1):
            match = re.match(r"\[Ref ([^\]]+)\]\s*(.*)", chunk, flags=re.DOTALL)
            if match:
                chunk_id, source_text = match.groups()
                stage0_items.append(EvidenceItem(
                    chunk_id=str(chunk_id),
                    text=str(source_text),
                    score=float(len(seed_chunks) - position + 1),
                    path_position=position,
                    triple="",
                    is_gold=str(chunk_id) in gold_evidence,
                ))
        record_candidates(stage0_items, hop=0)
        record_evidence([item.chunk_id for item in stage0_items])

        prompt_0 = self._build_agent_prompt(
            query=query, stage="checking_seeds", known_evidence=list(relevant_entities), 
            current_focus_content=seed_infos, related_chunks=seed_chunks, valid_next_hops=seeds
        )
        decision_0 = self.llm.invoke(prompt_0).content
        parsed_0 = self._parse_llm_decision(decision_0, valid_scope=None)
        parsed_0, stage0_replay_applied = replay_decision(
            query, parsed_0, self.stage0_replay, query_id=query_id
        )
        log(f" 💭 [LLM Decision]: {parsed_0}", Colors.GREEN)

        record_stage0_policy(
            parsed_0.get('answer', ''),
            proposed_final=parsed_0['is_final'],
            allow_early_exit=self.allow_stage0_exit,
            reasoning_stress=self.reasoning_stress,
            decision=parsed_0,
            replay_applied=stage0_replay_applied,
        )
        if (
            parsed_0['is_final']
            and self.allow_stage0_exit
            and not self.reasoning_stress
        ):
            stage0_gate = verify_stage0_answer(
                query=query,
                answer=parsed_0['answer'],
                cited_refs=parsed_0.get("supporting_refs", []),
                stage0_items=stage0_items,
                graph=self.G,
                confidence=parsed_0.get("confidence", ""),
            )
            record_stage0_gate(parsed_0['answer'], stage0_gate)
            if stage0_gate.accepted:
                suffix = (
                    "Verified"
                    if stage0_gate.policy == "evidence_verified"
                    else "Legacy"
                )
                return parsed_0['answer'], f"Agent-Zero-Shot-{suffix}"
        else:
            reason = (
                "model_not_final"
                if not parsed_0['is_final']
                else "stage0_exit_disabled"
            )
            record_stage0_gate("", Stage0GateResult(
                accepted=False,
                policy=os.getenv("ID_SGTR_STAGE0_POLICY", "legacy"),
                rejection_reason=reason,
            ))

        if self.reasoning_stress:
            relevant_nodes_step = list(dict.fromkeys(seeds))
        else:
            relevant_nodes_step = parsed_0['relevant_nodes']
            if not relevant_nodes_step: relevant_nodes_step = seeds
        relevant_entities.update(relevant_nodes_step)
        
        active_nodes = []
        stage0_next_nodes = (
            relevant_nodes_step if self.reasoning_stress else parsed_0['next_nodes']
        )
        for n in stage0_next_nodes:
            if n in self.G: active_nodes.append(n)
        if not active_nodes:
            active_nodes = [n for n in relevant_nodes_step if n in self.G]

        # --- Stage N: multi-hop reasoning layer (Graph Reasoning) ---
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
            hop_rerank_query = self._build_hop_rerank_query(
                query, hop, active_nodes, relevant_entities, history_facts,
                candidate_paths,
            )
            original_weight = max(0.0, float(
                os.getenv("ID_SGTR_HOP_ORIGINAL_WEIGHT", "0.35")
            ))
            hop_weight = max(0.0, float(
                os.getenv("ID_SGTR_HOP_DYNAMIC_WEIGHT", "0.45")
            ))
            hop_cross_weight = (
                hop_weight / (original_weight + hop_weight)
                if original_weight + hop_weight > 0 else 0.5
            )
            valid_next_hop_candidates = list(dict.fromkeys(
                p['v'] for p in candidate_paths if p['v'] not in visited_nodes
            ))

            structure_chunk_ids = []
            structure_chunk_seen = set()
            for p in candidate_paths:
                if self.G.has_edge(p['u'], p['v']):
                    edge_data = self.G[p['u']][p['v']]
                    edge_ctxs = edge_data.get('context_ids', set())
                    if ctx_id_str in edge_ctxs:
                        if 'chunk_ids' in edge_data:
                            for chunk_id in self._context_chunk_ids(edge_data, ctx_id_str):
                                value = str(chunk_id)
                                if value not in structure_chunk_seen:
                                    structure_chunk_ids.append(value)
                                    structure_chunk_seen.add(value)
                        
            # Build a candidate pool first; the EvidenceAssembler applies the
            # final dynamic slot budget shared by all controlled variants.
            struct_limit = min(
                len(structure_chunk_ids), TokenConfig.MAX_CANDIDATE_POOL
            )
            if struct_limit > 0 and not self.query_reranker.enabled:
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
                            semantic_pool_ids.update(self._context_chunk_ids(edge_data, ctx_id_str))
            
            # ✅ Defensive: double set deduplication
            semantic_pool_ids = {str(c) for c in semantic_pool_ids}
            struct_cids_set = {str(c) for c in filtered_struct_cids}
            semantic_pool_ids = semantic_pool_ids - struct_cids_set

            if self.query_reranker.enabled:
                filtered_struct_cids = []
                semantic_pool_ids = {
                    str(cid)
                    for cid in self.chunk_dict_by_ctx.get(ctx_id_str, [])
                }
            
            remaining_slots = max(
                0, TokenConfig.MAX_CANDIDATE_POOL - len(filtered_struct_cids)
            )
            filtered_sem_cids = []
            hop_rerank_scores = {}
            if remaining_slots > 0 and semantic_pool_ids:
                filtered_sem_cids = self._get_top_chunks(
                    list(semantic_pool_ids), 
                    query_vec, 
                    top_k=remaining_slots,
                    min_score=TokenConfig.CHUNK_SIM_THRESHOLD_STRICT,
                    rerank_query=hop_rerank_query,
                    rerank_base_scores=original_rerank_scores,
                    rerank_score_cache=hop_rerank_scores,
                    rerank_cross_weight=hop_cross_weight,
                )

            path_metadata = {}
            for position, path_item in enumerate(candidate_paths, start=1):
                if self.G.has_edge(path_item['u'], path_item['v']):
                    triple = f"{path_item['u']} --[{path_item['rel']}]--> {path_item['v']}"
                    for chunk_id in self._context_chunk_ids(
                        self.G[path_item['u']][path_item['v']], ctx_id_str
                    ):
                        trace = (position, triple)
                        traces = path_metadata.setdefault(str(chunk_id), [])
                        if trace not in traces:
                            traces.append(trace)

            ordered_ids = [str(cid) for cid in filtered_struct_cids + filtered_sem_cids]
            if self.query_reranker.enabled:
                chunk_score_map = {
                    chunk_id: float(hop_rerank_scores.get(chunk_id, 0.0))
                    for chunk_id in ordered_ids
                }
            else:
                chunk_score_map = dict(self._rank_chunks(
                    ordered_ids, query_vec, top_k=len(ordered_ids),
                    min_score=float("-inf")
                ))
            evidence_items = []
            for rank, chunk_id in enumerate(ordered_ids):
                traces = path_metadata.get(chunk_id, [])
                if traces:
                    position, triple = traces[0]
                else:
                    position, triple = len(candidate_paths) + rank + 1, ""
                evidence_items.append(EvidenceItem(
                    chunk_id=chunk_id,
                    text=self._get_chunk_text(chunk_id),
                    score=float(chunk_score_map.get(chunk_id, 0.0)),
                    path_position=position,
                    triple=triple,
                    is_gold=chunk_id in gold_evidence,
                    hop=hop,
                    topology_trace=tuple(
                        (hop, trace_position, trace_triple)
                        for trace_position, trace_triple in traces
                    ),
                ))

            # Fold evidence accumulated across all visited hops into one bounded
            # window.  Earlier source passages are no longer discarded at the
            # next graph expansion step.
            folded_evidence_items.extend(evidence_items)
            final_chunks_for_prompt, selected_chunk_ids = self.evidence_assembler.render(
                folded_evidence_items,
                self.evidence_variant,
                query_id=query,
                char_limit=TokenConfig.CHUNK_CHAR_LIMIT,
            )
            last_folded_chunks = list(final_chunks_for_prompt)
            last_selected_chunk_ids = list(selected_chunk_ids)
            record_candidates(evidence_items, hop)
            record_evidence(selected_chunk_ids)
            for chunk_id in selected_chunk_ids:
                clean_txt = self._get_chunk_text(chunk_id)[:TokenConfig.CHUNK_CHAR_LIMIT].replace('\n', ' ')
                accumulated_chunk_texts.add(f"[Ref {chunk_id}] {clean_txt}...")

            if verbose:
                log(f"   📝 Context: {len(filtered_struct_cids)} Path Chunks, {len(filtered_sem_cids)} Context Chunks added.")

            nodes_to_display = set(relevant_entities) | set(active_nodes)
            evidence_with_desc = []
            for node in nodes_to_display:
                if node in entity_memory:
                    evidence_with_desc.append(f"**{node}**: {entity_memory[node]}...") 
                else:
                    evidence_with_desc.append(node)
            
            # ✅ Fix: inject historical paths previously discovered by LLM into its memory
            prompt_n = self._build_agent_prompt(
                query=query, stage="stage_n", known_evidence=evidence_with_desc,
                current_focus_content=path_strings + history_facts,
                related_chunks=final_chunks_for_prompt,
                valid_next_hops=valid_next_hop_candidates
            )
            
            decision_n = self.llm.invoke(prompt_n).content
            parsed_n = self._parse_llm_decision(decision_n, valid_scope=valid_next_hop_candidates)
            decision_relevant_nodes = list(parsed_n.get('relevant_nodes') or [])
            decision_next_nodes = list(parsed_n.get('next_nodes') or [])
            chosen_targets = set(decision_relevant_nodes) | set(decision_next_nodes)
            chosen_runtime_edges = []
            for position, path_item in enumerate(candidate_paths, start=1):
                if path_item['v'] not in chosen_targets:
                    continue
                edge_data = self.G[path_item['u']][path_item['v']]
                chunk_ids = self._context_chunk_ids(edge_data, ctx_id_str)
                if chunk_ids:
                    chosen_runtime_edges.append({
                        'source_entity': path_item['u'],
                        'relation': path_item['rel'],
                        'target_entity': path_item['v'],
                        'chunk_ids': list(chunk_ids),
                        'path_position': position,
                    })
            record_runtime_step(
                hop, active_nodes, decision_relevant_nodes,
                decision_next_nodes, chosen_runtime_edges,
            )
            log(f" 💭 [LLM Decision]: {parsed_n}", Colors.GREEN)

            if (
                parsed_n['is_final']
                and os.getenv("ID_SGTR_HOP_CONFIDENCE_GATE", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            ):
                selected_set = {str(value) for value in selected_chunk_ids}
                visible_items = []
                visible_seen = set()
                for item in reversed(folded_evidence_items):
                    chunk_id = str(item.chunk_id)
                    if chunk_id in selected_set and chunk_id not in visible_seen:
                        visible_items.append(item)
                        visible_seen.add(chunk_id)
                visible_items.reverse()
                hop_gate = verify_stage0_answer(
                    query=query,
                    answer=parsed_n['answer'],
                    cited_refs=parsed_n.get("supporting_refs", []),
                    stage0_items=visible_items,
                    graph=self.G,
                    confidence=parsed_n.get("confidence", ""),
                )
                record_hop_gate(hop, hop_gate)
                if not hop_gate.accepted:
                    parsed_n['is_final'] = False

            if parsed_n['is_final']:
                should_continue = (
                    self.reasoning_stress
                    and hop < max_hops
                    and bool(valid_next_hop_candidates)
                )
                if not should_continue:
                    return parsed_n['answer'], f"Agent-Hop-{hop}"
                provisional_answer = str(parsed_n.get('answer') or '').strip()
                parsed_n['is_final'] = False
                frontier = valid_next_hop_candidates[:self.reasoning_beam_width]
                parsed_n['relevant_nodes'] = frontier
                parsed_n['next_nodes'] = frontier

            relevant_entities.update(parsed_n['relevant_nodes'])
            nodes_to_check = set(parsed_n['relevant_nodes']) | set(parsed_n['next_nodes'])
            unknown_nodes = [n for n in nodes_to_check if n not in entity_memory]
            if unknown_nodes:
                new_defs, _ = self._get_node_details(unknown_nodes, ctx_id_str, query_vec=query_vec, add_chunks=False)
                for node_name, node_desc in new_defs.items():
                    entity_memory[node_name] = node_desc

            for p in candidate_paths:
                if p['v'] in parsed_n['relevant_nodes']:
                    fact = f"{p['u']} {p['rel']} {p['v']}"
                    if fact not in history_fact_seen:
                        history_facts.append(fact)
                        history_fact_seen.add(fact)
            
            next_targets = []
            for n in parsed_n['next_nodes']:
                if n in valid_next_hop_candidates or (n in self.G and n not in visited_nodes):
                    next_targets.append(n)

            if self.reasoning_stress:
                for n in valid_next_hop_candidates[:self.reasoning_beam_width]:
                    if n not in next_targets:
                        next_targets.append(n)
            
            if not next_targets and valid_next_hop_candidates:
                 next_targets = valid_next_hop_candidates[:self.reasoning_beam_width]

            executed_target_set = set(next_targets)
            executed_runtime_edges = []
            for position, path_item in enumerate(candidate_paths, start=1):
                if path_item['v'] not in executed_target_set:
                    continue
                edge_data = self.G[path_item['u']][path_item['v']]
                chunk_ids = self._context_chunk_ids(edge_data, ctx_id_str)
                if chunk_ids:
                    executed_runtime_edges.append({
                        'source_entity': path_item['u'],
                        'relation': path_item['rel'],
                        'target_entity': path_item['v'],
                        'chunk_ids': list(chunk_ids),
                        'path_position': position,
                    })
            record_runtime_execution(hop, next_targets, executed_runtime_edges)
            
            active_nodes = next_targets
            log(f"   📌 Relevant Update: {list(relevant_entities)}", Colors.CYAN)
            log(f"   🚀 Next Hop: {active_nodes}", Colors.WARNING)

        final_policy = os.getenv(
            "ID_SGTR_MUSIQUE_FINAL_POLICY", "topology"
        ).strip().lower()
        if final_policy in {"terminal", "terminal_v21", "terminal-recovery-v2.1"}:
            answer = self._terminal_recovery_answer(
                query, ctx_id_str, query_vec, last_selected_chunk_ids,
                relevant_entities, history_facts, provisional_answer,
            )
            return (answer or "No answer"), "Terminal-Recovery-v2.1"

        # Default: retain the original topology-aligned final synthesis.
        answer = self._answer_from_folded_context(query, last_folded_chunks)
        return (answer or provisional_answer or "No answer"), "Topology-Final-Synthesis"

    def _build_agent_prompt(self, query, stage, known_evidence, current_focus_content, related_chunks, valid_next_hops):
        """Format agent instruction prompt"""
        evidence_str = "\n".join(known_evidence) if known_evidence else "None"
        chunks_str = "\n".join(related_chunks) if related_chunks else "None"
        confidence_gate = (
            os.getenv("ID_SGTR_STAGE0_POLICY", "legacy").strip().lower()
            in {"confidence_grounded", "confidence_coverage"}
            and (
                stage == "checking_seeds"
                or os.getenv("ID_SGTR_HOP_CONFIDENCE_GATE", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            )
        )
        stage0_refs_line = (
            "`Supporting Refs: [comma-separated Ref IDs that directly support the answer]`"
            if confidence_gate else ""
        )
        stage0_confidence_line = (
            "`Confidence: HIGH` only when those refs directly support the final "
            "answer; otherwise use Scenario B."
            if confidence_gate else ""
        )
        
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
- **YES** -> select Scenario A to output `Final Answer`. IMMEDIATELY
- **NO** -> select Scenario B to find the answer in the graph. **NEVER say "Not Found".**
    
### Output Format
Scenario A. If Answerable (Answer Found):
`Final Answer: [Clean Entity Name / Yes / No / data / etc.]` (Precise and Concise)
{stage0_refs_line}
{stage0_confidence_line}

Scenario B. If Not Answerable (Answer not Found):
`Relevant Nodes: [...]` (Select useful entities found in Entity Definitions, separate with semicolons e.g., EntityA; EntityB)
`Next Hop: [...]` (Select 1-3 useful nodes from 'Valid Next Hops' to explore graph, separate with semicolons e.g., EntityA; EntityB)

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
- **YES** -> select Scenario A to output `Final Answer`. IMMEDIATELY
- **NO** -> select Scenario B to continue searching. **NEVER say "Not Found".**
    
### Output Format
Scenario A. If Answerable (Answer Found):
`Final Answer: [Clean Entity Name / Yes / No / data / etc.]` (Precise and Concise)
{stage0_refs_line}
{stage0_confidence_line}

Scenario B. If Not Answerable (Answer not Found):
`Relevant Nodes: [...]` (Select useful entities found in Entity Definitions, separate with semicolons e.g., EntityA; EntityB)
`Next Hop: [...]` (Select 1-3 useful nodes from 'Valid Next Hops' to explore graph, separate with semicolons e.g., EntityA; EntityB)

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
        result = {"is_final": False, "answer": "", "confidence": "", "supporting_refs": [], "relevant_nodes": [], "next_nodes": []}
        result["confidence"] = parse_stage0_confidence(text)
        result["supporting_refs"] = parse_supporting_refs(text)

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
            stop_tokens = ["\n\n", "Confidence:", "Supporting Refs:", "Supporting Ref:", "Relevant Nodes:", "Next Hop:", "If Not Answerable", "###"]
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
    
    def _oracle_answer(self, query, gold_evidence):
        items = [
            EvidenceItem(
                chunk_id=str(chunk_id),
                text=self._get_chunk_text(chunk_id),
                path_position=rank,
                is_gold=True,
            )
            for rank, chunk_id in enumerate(gold_evidence, start=1)
            if str(chunk_id) in self.chunk_df.index
        ]
        references, selected_chunk_ids = self.evidence_assembler.render(
            items,
            EvidenceVariant.ORACLE,
            query_id=query,
            char_limit=TokenConfig.CHUNK_CHAR_LIMIT,
        )
        record_evidence(selected_chunk_ids)
        prompt = f"""Answer the query using ONLY the gold supporting passages below.

Query: {query}

Gold supporting passages:
{chr(10).join(references)}

Return only: Final Answer: [answer]"""
        response = self.llm.invoke(prompt).content.strip()
        return response.split("Final Answer:")[-1].strip() if "Final Answer:" in response else response

    def _entity_to_chunk_answer(self, query, context_id, seeds, query_vec):
        candidate_ids = set()
        ctx_id_str = str(context_id)
        for seed in seeds:
            if seed not in self.G:
                continue
            for neighbor in self.G.neighbors(seed):
                edge = self.G[seed][neighbor]
                edge_contexts = {str(value) for value in edge.get('context_ids', set())}
                if ctx_id_str in edge_contexts:
                    candidate_ids.update(str(value) for value in edge.get('chunk_ids', []))
        ranked_ids = self._get_top_chunks(
            list(candidate_ids),
            query_vec,
            top_k=self.evidence_assembler.budget,
            min_score=TokenConfig.CHUNK_SIM_THRESHOLD_LOOSE,
        )
        items = [
            EvidenceItem(chunk_id=str(chunk_id), text=self._get_chunk_text(chunk_id), score=len(ranked_ids) - rank)
            for rank, chunk_id in enumerate(ranked_ids)
        ]
        references, selected_chunk_ids = self.evidence_assembler.render(
            items,
            EvidenceVariant.ENTITY_TO_CHUNK,
            query_id=query,
            char_limit=TokenConfig.CHUNK_CHAR_LIMIT,
        )
        record_evidence(selected_chunk_ids)
        prompt = f"""Answer the query using ONLY the source passages below.

Query: {query}

Source passages:
{chr(10).join(references) if references else 'No source passage found.'}

Return only: Final Answer: [answer]"""
        response = self.llm.invoke(prompt).content.strip()
        return response.split("Final Answer:")[-1].strip() if "Final Answer:" in response else response

    def _fallback_answer(self, query, context_id, query_vec, seed_infos={}, verbose=False):
        """Industrial-grade hybrid retrieval fallback layer: when graph reasoning breaks, use BM25 (lexical matching) + dense vector hybrid scoring to extract answer"""
        if verbose:
            print(f"{Colors.WARNING}⚠️ [Fallback] Switching to BM25+Vector Hybrid RAG...{Colors.ENDC}")

        candidate_cids = []
        ctx_id_str = str(context_id)
        if ctx_id_str in self.chunk_dict_by_ctx:
            candidate_cids = self.chunk_dict_by_ctx[ctx_id_str]
        
        top_chunks_text = []
        selected_fallback_ids = []
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
                    selected_fallback_ids.append(str(cid))

        record_evidence(selected_fallback_ids)
                
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

    def _terminal_recovery_answer(
        self, query, context_id, query_vec, prior_chunk_ids,
        entities, facts, provisional_answer="",
    ):
        """Re-rank the complete local context and perform one grounded synthesis."""
        context_key = str(context_id)
        candidate_ids = [
            str(value)
            for value in self.chunk_dict_by_ctx.get(context_key, [])
        ]
        raw_candidate_count = len(candidate_ids)
        candidate_ids = deduplicate_chunk_ids_by_text(
            candidate_ids, self._get_chunk_text,
            preferred_ids=prior_chunk_ids,
        )
        budget = max(
            1, int(os.getenv("ID_SGTR_TERMINAL_EVIDENCE_BUDGET", "5"))
        )
        terminal_query = build_terminal_query(
            query, entities=entities, facts=facts
        )
        ranked_ids = self._get_top_chunks(
            candidate_ids,
            query_vec,
            top_k=len(candidate_ids),
            min_score=-1.0,
            rerank_query=terminal_query,
        )
        selected_ids = select_terminal_chunks(
            ranked_ids, prior_chunk_ids, budget=budget,
            text_getter=self._get_chunk_text,
        )
        evidence_items = [
            EvidenceItem(
                chunk_id=chunk_id,
                text=self._get_chunk_text(chunk_id),
                score=float(len(selected_ids) - position),
                path_position=position + 1,
                triple="",
            )
            for position, chunk_id in enumerate(selected_ids)
        ]
        evidence_lines = [
            f"[Ref {item.chunk_id}] "
            f"{item.text[:TokenConfig.CHUNK_CHAR_LIMIT].replace(chr(10), ' ')}"
            for item in evidence_items
        ]
        record_evidence(selected_ids)
        terminal_started = time.perf_counter()
        response = self.llm_terminal.invoke(
            build_terminal_prompt(query, evidence_lines)
        )
        draft = parse_terminal_response(response)
        gate = verify_stage0_answer(
            query=query,
            answer=draft.answer,
            cited_refs=draft.supporting_refs,
            stage0_items=evidence_items,
            graph=self.G,
            confidence=draft.confidence,
        )
        chosen_answer = draft.answer
        chosen_gate = gate
        if not gate.accepted and provisional_answer:
            provisional_gate = verify_stage0_answer(
                query=query,
                answer=provisional_answer,
                cited_refs=[],
                stage0_items=evidence_items,
                graph=self.G,
                confidence="high",
            )
            if provisional_gate.accepted:
                chosen_answer = str(provisional_answer).strip()
                chosen_gate = provisional_gate
        record_terminal_recovery(
            contexts=[context_key],
            raw_candidate_count=raw_candidate_count,
            candidate_count=len(candidate_ids),
            selected_count=len(selected_ids),
            elapsed_s=time.perf_counter() - terminal_started,
            result=chosen_gate,
        )
        if chosen_answer and chosen_answer.strip().casefold() not in {
            "unknown", "no answer", "not found",
        }:
            return chosen_answer.strip()
        return str(provisional_answer or "").strip()


    def _answer_from_folded_context(self, query, folded_chunks):
        """Two-stage constant-call synthesis over the final folding window."""
        evidence = chr(10).join(folded_chunks)
        reasoning_prompt = f"""Solve the multi-hop question using ONLY the topology-aligned source evidence below.

Follow every relation required by the question. For comparisons, resolve the
requested attribute for both alternatives. Think carefully and end with a
short Final Answer.

Question: {query}

Topology-aligned evidence:
{evidence}

End with: Final Answer: <shortest answer span>
"""
        reasoning_response = self.llm_final_reasoner.invoke(reasoning_prompt)
        content = str(getattr(reasoning_response, "content", "") or "").strip()
        extra = getattr(reasoning_response, "additional_kwargs", {}) or {}
        metadata = getattr(reasoning_response, "response_metadata", {}) or {}
        reasoning = str(
            extra.get("reasoning") or extra.get("reasoning_content")
            or metadata.get("reasoning") or metadata.get("reasoning_content")
            or getattr(reasoning_response, "reasoning", "") or ""
        ).strip()
        draft = "\n".join(part for part in (reasoning, content) if part).strip()

        formatter_prompt = f"""Normalize the answer to the question using the evidence and reasoning draft.

Question: {query}

Evidence:
{evidence}

Reasoning draft:
{draft or 'No reasoning draft was returned.'}

Final Answer: <shortest answer span>

The answer span must be only the entity, place, date, number, yes, or no.
Return exactly one line. Do not return reasoning, a sentence, an intermediate
entity, or a different attribute.
"""
        response = self.llm_final.invoke(formatter_prompt).content
        text = str(response or "").strip()
        if "Final Answer:" in text:
            text = text.split("Final Answer:")[-1].strip()
        text = text.splitlines()[0].strip() if text else ""
        text = text.strip().strip('`').strip('"').strip("'")
        binary = re.match(r"^(yes|no)\b", text, flags=re.IGNORECASE)
        return binary.group(1).lower() if binary else text

    def _expand_neighbors(self, source_nodes, query_vec, context_id, intent_weights, top_k_per_node=None, verbose=False, visited_set=None):
        """Look at K neighbor nodes outward, comprehensively consider edge type, implicit score, and semantic vector, score and return top K optimal hops"""
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
        
    def _get_node_details(
        self, nodes, context_id, query_vec=None, add_chunks=True,
        rerank_score_cache=None,
    ):
        details = {}
        all_candidate_chunks = set()
        entity_df = self.matcher.df.set_index('Standard_Entity')
        
        ctx_id_str = str(context_id) # ✅ safe string conversion
            
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
                            all_candidate_chunks.update(
                                self._context_chunk_ids(edge_data, ctx_id_str)
                            )

        chunks = []
        if add_chunks and self.query_reranker.enabled:
            all_candidate_chunks = set(
                str(value)
                for value in self.chunk_dict_by_ctx.get(ctx_id_str, [])
            )
        if add_chunks and all_candidate_chunks:
            limit = TokenConfig.STAGE0_MAX_CHUNKS
            from experiments.pcef_v2 import enabled as pcef_v2_enabled, stage0_select
            
            ranked_stage0 = self._rank_chunks(
                list(all_candidate_chunks), 
                query_vec, 
                top_k=len(all_candidate_chunks) if pcef_v2_enabled() else limit,
                min_score=TokenConfig.CHUNK_SIM_THRESHOLD_LOOSE,
                rerank_score_cache=rerank_score_cache,
            )
            best_cids = (
                stage0_select(self, nodes, ctx_id_str, ranked_stage0, limit, current_rerank_query())
                if pcef_v2_enabled() else [cid for cid, _ in ranked_stage0]
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

    def _context_chunk_ids(self, edge_data, context_id):
        """Return only edge chunks that belong to the current QA context.

        The undirected graph historically unions ``chunk_ids`` when the same
        entity pair occurs in several contexts.  ``context_ids`` alone cannot
        preserve the chunk-to-context association, so filter against chunk_df.
        """
        ctx = str(context_id)
        values = []
        seen = set()
        for raw_chunk_id in edge_data.get('chunk_ids', []):
            chunk_id = str(raw_chunk_id)
            if chunk_id in seen or chunk_id not in self.chunk_df.index:
                continue
            try:
                row_context = self.chunk_df.loc[chunk_id, 'context_id']
                if isinstance(row_context, pd.Series):
                    belongs = row_context.astype(str).eq(ctx).any()
                else:
                    belongs = str(row_context) == ctx
            except Exception:
                belongs = False
            if belongs:
                values.append(chunk_id)
                seen.add(chunk_id)
        return values
    
    def _solve_impl(self, query, context_id, verbose=True, mode='full', gold_evidence=None, query_id=None):
        """Main entry function exposed to external callers"""
        stage_n_chunks = TokenConfig.MAX_CHUNKS_IN_PROMPT
        query_vec = None
        
        # ✅ Force conversion at entry to protect the entire engine
        ctx_id_str = str(context_id)
        
        with self._embedding_guard():
            try:
                query_vec = self.graph_embed_model.embed_query(query)
                query_vec = np.array(query_vec, dtype=np.float32)
            except Exception as e: 
                print(f"⚠️ Query embedding failed: {e}")
                
        if mode == 'vector_only':
            return self._fallback_answer(query, ctx_id_str, query_vec, verbose=verbose), "Vector-RAG"

        if self.evidence_variant is EvidenceVariant.ORACLE:
            if not gold_evidence:
                raise ValueError("oracle evidence variant requires gold_evidence chunk IDs")
            return self._oracle_answer(query, gold_evidence), "Oracle-Supporting-Chunks"

        weights, strategy = self.step1_analyze_intent(query, query_vec)
        seeds = self.step2_semantic_anchoring(
            query,
            ctx_id_str,
            query_vec=query_vec,
        )
        seeds, _ = replay_seeds(
            query, seeds, self.stage0_replay, query_id=query_id
        )
        record_stage0_seeds(seeds)
        
        if not seeds: 
            return "Sorry, no relevant entities found in the knowledge graph.", strategy

        if mode == 'entity_to_chunk':
            return self._entity_to_chunk_answer(query, ctx_id_str, seeds, query_vec), "Entity-to-Chunk"

        if mode == 'explicit_only':
            weights = [0.0, 0.0, 1.0] 
        
        answer, final_stage_tag = self.step3_iterative_agent_reasoning(
            seeds, query, ctx_id_str, 
            intent_weights=weights, 
            max_hops=4, 
            verbose=verbose,
            query_vec=query_vec,
            max_prompt_chunks=stage_n_chunks,
            gold_evidence=gold_evidence,
            query_id=query_id,
        )
        
        return answer, f"{strategy} -> {final_stage_tag}"

    def solve(self, query, context_id, verbose=True, mode='full', return_trace=False, query_id=None, gold_evidence=None):
        telemetry = QueryTelemetry(query_id=str(query_id if query_id is not None else context_id or ""))
        telemetry.start()
        try:
            with bind_telemetry(telemetry), bind_rerank_query(query):
                answer, strategy = self._solve_impl(
                    query, context_id, verbose=verbose, mode=mode,
                    gold_evidence=gold_evidence,
                    query_id=query_id,
                )
        finally:
            telemetry.stop()

        hop_match = re.search(r"Agent-Hop-(\d+)", strategy)
        telemetry.retrieval_rounds = int(hop_match.group(1)) if hop_match else 0
        telemetry.fallback = "Fallback" in strategy
        if telemetry.fallback and telemetry.retrieval_rounds == 0:
            telemetry.retrieval_rounds = 3
        if return_trace:
            return answer, strategy, telemetry.to_dict()
        return answer, strategy
    
    
# ==========================================
# 4. Main program entry (Reasoning Setting P0)
# ==========================================
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data_output", "dataset", "musique", "ds1000")
    ADAPT_ROOT = os.path.join(PROJECT_ROOT, "adapt")
    
    QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
    GRAPH_FILE = os.path.join(DATA_ROOT, "graph.csv")
    CHUNK_EMB_FILE = os.path.join(DATA_ROOT, "chunks_with_embeddings.parquet")
    CHUNK_RAW_FILE = os.path.join(DATA_ROOT, "chunk.csv")
    PARQUET_FILE = os.path.join(DATA_ROOT, "concepts_merged_with_vectors.parquet")
    PROX_FILE = os.path.join(DATA_ROOT, "contextual_proximity.csv")
    MODEL_FILE = os.path.join(ADAPT_ROOT, "intent_classifier_struct.pth")

    try:
        df_qa = pd.read_csv(QA_FILE, sep="|")
        df_graph = pd.read_csv(GRAPH_FILE, sep="|")
        
        if os.path.exists(CHUNK_EMB_FILE):
            print(f"📦 Loading Chunk data with vectors: {CHUNK_EMB_FILE}")
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

    print("🚀 Initializing ID-SGTR engine...")
    os.environ.setdefault("ID_SGTR_TEMPERATURE", "0")
    os.environ.setdefault("ID_SGTR_MAX_TOKENS", "512")
    os.environ.setdefault("ID_SGTR_ENABLE_THINKING", "false")

    MASK_RATIO = 0
    EVIDENCE_VARIANT = os.getenv("ID_SGTR_EVIDENCE_VARIANT", "topology_folding")
    EVIDENCE_BUDGET = int(os.getenv("ID_SGTR_EVIDENCE_BUDGET", str(TokenConfig.MAX_CHUNKS_IN_PROMPT)))
    RUN_MODE = os.getenv("ID_SGTR_RUN_MODE", "full")
    REASONING_STRESS = os.getenv("ID_SGTR_REASONING_STRESS", "true").lower() in {"1", "true", "yes", "on"}
    ALLOW_STAGE0_EXIT = os.getenv(
        "ID_SGTR_ALLOW_STAGE0_EXIT", "true"
    ).lower() in {"1", "true", "yes", "on"}

    engine = ID_SGTR_Reasoning_Engine(
        intent_model_path=MODEL_FILE,
        parquet_path=PARQUET_FILE,
        graph_df=df_graph,
        chunk_df=df_chunk,
        proximity_df=df_prox,
        edge_mask_ratio=MASK_RATIO,
        evidence_variant=EVIDENCE_VARIANT,
        evidence_budget=EVIDENCE_BUDGET,
        reasoning_stress=REASONING_STRESS,
        allow_stage0_exit=ALLOW_STAGE0_EXIT,
    )

    sample_size = int(os.getenv("ID_SGTR_SAMPLE_SIZE", "200"))
    target_data = df_qa.head(sample_size)

    subset_file = os.getenv("ID_SGTR_SUBSET_FILE")
    if subset_file:
        target_data = pd.read_csv(subset_file, sep="|")
        required_subset_columns = {"question", "answer", "context_id", "query_id"}
        missing_subset_columns = required_subset_columns.difference(target_data.columns)
        if missing_subset_columns:
            raise ValueError(f"Subset file is missing required columns: {sorted(missing_subset_columns)}")
        target_data = target_data.head(sample_size)
    
    print(f"\n📝 Starting concurrent processing of {len(target_data)} queries...")

    def process_query_wrapper(i: int, row: pd.Series) -> Tuple[int, Any]:
        q = row["question"]
        ctx = row["context_id"]
        gold = row["answer"]
        stable_query_id = row.get("query_id", i)

        raw_gold_evidence = row.get("gold_evidence", [])
        if isinstance(raw_gold_evidence, str):
            try:
                raw_gold_evidence = ast.literal_eval(raw_gold_evidence)
            except (SyntaxError, ValueError):
                raw_gold_evidence = [
                    value.strip() for value in raw_gold_evidence.split(",") if value.strip()
                ]

        gold_evidence = (
            list(raw_gold_evidence)
            if isinstance(raw_gold_evidence, (list, tuple, set))
            else []
        )

        pred_answer, strategy, trace = engine.solve(
            q,
            ctx,
            verbose=False,
            mode=RUN_MODE,
            return_trace=True,
            query_id=stable_query_id,
            gold_evidence=gold_evidence,
        )
                
        result = {
            "query_id": stable_query_id,
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "strategy": strategy,
            "context_id": ctx,
            "gold_evidence": gold_evidence,
            "gold_mapping_complete": row.get("gold_mapping_complete", ""),
            "model": os.getenv("SILICONFLOW_MODEL", ""),
            "temperature": os.getenv("ID_SGTR_TEMPERATURE", "0"),
            "max_output_tokens": os.getenv("ID_SGTR_MAX_TOKENS", "512"),
            "evidence_variant": EVIDENCE_VARIANT,
            "evidence_budget": EVIDENCE_BUDGET,
            "run_mode": RUN_MODE,
            "reasoning_stress": REASONING_STRESS,
            "allow_stage0_exit": ALLOW_STAGE0_EXIT,
        }
        result.update(trace)
        return i, result
        
    processed_results = parallel_llm_processor(
        dataframe=target_data,
        processing_func=process_query_wrapper,
        start_message="Starting multi-threaded reasoning...",
        max_workers=int(os.getenv("ID_SGTR_MAX_WORKERS", "1")),
        max_retries=6,
        initial_delay=2,
    )

    processed_results.sort(key=lambda x: x[0])
    final_data = [item[1] for item in processed_results]

    output_path = os.getenv(
        "ID_SGTR_OUTPUT",
        os.path.join(current_dir, "query_results_agent_musique_local_p0.csv"),
    )
    pd.DataFrame(final_data).to_csv(output_path, index=False, sep="|")
    print(f"\n✅ Processing complete, results saved to: {output_path}")
