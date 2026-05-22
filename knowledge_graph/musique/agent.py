import torch
import time
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import json
import re
import os
import sys
import ast
import threading
import difflib
import operator
from tqdm import tqdm
from typing import List, Tuple, Callable, Any, Dict, Set, Annotated, Union, Literal
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangGraph 核心组件
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 0. 环境配置与自定义模块引入
# ==========================================
# 假设这些模块在你的项目目录中，保持引用不变
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 请确保这些文件在你本地存在
try:
    from adapt.adapt import IntentClassifier, dynamic_weight_modulation, INPUT_DIM, HIDDEN_DIM
    from utils import get_embeddings_model, get_chat_model  # 或 get_llm_model
    from seed import SemanticMatcher 
    from helper import parallel_llm_processor
except ImportError as e:
    print(f"❌ 导入错误: 请确保 adapt, utils, seed, helper 模块都在路径中. ({e})")
    sys.exit(1)

# ==========================================
# 1. 辅助类：控制台颜色与配置
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

class TokenConfig:
    """控制 Token 消耗与检索规模"""
    STAGE0_ADD_CHUNKS = True 
    STAGE0_MAX_CHUNKS = 3
    TOP_K_NEIGHBORS = 8
    MIN_EDGE_SCORE = 0.20
    MAX_CANDIDATE_POOL = 20
    CHUNK_CHAR_LIMIT = 1000
    MAX_CHUNKS_IN_PROMPT = 4

# ==========================================
# 2. 状态定义 (State)
# ==========================================
class AgentState(TypedDict):
    """LangGraph 状态定义"""
    # 输入
    query: str
    context_id: Any
    
    # 意图
    intent_weights: List[float]
    intent_strategy: str
    
    # 图推理状态
    seeds: List[str]
    relevant_entities: List[str]    # 累积的相关实体
    visited_nodes: List[str]        # 禁忌表
    active_nodes: List[str]         # 当前探索前沿
    
    # 证据与记忆
    history_facts: List[str]        # 路径 "A --rel--> B"
    accumulated_chunks: List[str]   # 文本块 "[Ref X] text..."
    entity_memory: Dict[str, str]   # 实体定义 {name: desc}
    
    # 控制流
    current_hop: int
    max_hops: int
    final_answer: str
    is_solved: bool

# ==========================================
# 3. 资源管理器 (核心逻辑封装)
# ==========================================
class ID_SGTR_Resources:
    """
    持有重型资源（模型、图数据）并提供核心算法方法。
    """
    def __init__(self, intent_model_path, parquet_path, graph_df, chunk_df, proximity_df=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 初始化资源 (Device: {self.device})...")

        # 互斥锁：防止多线程同时调用 Embedding 模型
        self.gpu_lock = threading.Lock()

        # 1. 加载意图模型
        self.intent_model = IntentClassifier(INPUT_DIM, HIDDEN_DIM).to(self.device)
        try:
            if os.path.exists(intent_model_path):
                self.intent_model.load_state_dict(torch.load(intent_model_path, map_location=self.device))
                self.intent_model.eval()
        except Exception as e:
            print(f"⚠️ 意图模型加载警告: {e}")
        
        self.intent_embed_model = get_embeddings_model(dimensions=512)

        # 2. 加载语义匹配器与 Embedding
        self.matcher = SemanticMatcher(parquet_path)
        if hasattr(self.matcher, 'embed_model'):
            self.graph_embed_model = self.matcher.embed_model
        else:
            self.graph_embed_model = get_embeddings_model(dimensions=1024)

        # 3. 加载 LLM
        self.llm = get_chat_model(task_type="kg_query")

        # 4. 数据预处理
        self.chunk_df = chunk_df.copy()
        if 'chunk_id' in self.chunk_df.columns:
            self.chunk_df.set_index('chunk_id', inplace=True)
        
        # 解析向量列
        if 'embedding_np' not in self.chunk_df.columns:
            tqdm.pandas(desc="Parsing Vectors")
            self.chunk_df['embedding_np'] = self.chunk_df['embedding'].progress_apply(self._parse_vec_safe)

        # 5. 构建图
        self.G = self._build_hybrid_graph(graph_df, proximity_df)
        print("✅ 资源加载完毕")

    def _parse_vec_safe(self, x):
        if isinstance(x, np.ndarray): return x.astype(np.float32)
        if isinstance(x, list): return np.array(x, dtype=np.float32)
        if isinstance(x, str):
            try:
                if x.strip().startswith('['):
                    return np.array(ast.literal_eval(x), dtype=np.float32)
            except: return None
        return None

    def _build_hybrid_graph(self, graph_df, proximity_df):
        G = nx.Graph()
        has_ctx = 'context_id' in graph_df.columns
        
        print("🕸️ 构建图结构...")
        for row in tqdm(graph_df.itertuples(index=False), total=len(graph_df), desc="Graph Nodes"):
            u, v = row.node_1, row.node_2
            ctx_id = row.context_id if has_ctx else -1
            
            if G.has_edge(u, v):
                G[u][v]['context_ids'].add(ctx_id)
                G[u][v]['chunk_ids'].append(row.chunk_id)
            else:
                G.add_edge(u, v, type='explicit', relation=row.edge, 
                           chunk_ids=[row.chunk_id], context_ids={ctx_id})

        if proximity_df is not None and not proximity_df.empty:
            max_count = proximity_df['count'].max() + 1e-5
            for row in proximity_df.itertuples(index=False):
                u, v = row.node_1, row.node_2
                norm_count = np.log1p(row.count) / np.log1p(max_count)
                if G.has_edge(u, v):
                    G[u][v]['implicit_score'] = norm_count
                    G[u][v]['has_implicit'] = True
                else:
                    G.add_edge(u, v, type='implicit', implicit_score=norm_count,
                               has_implicit=True, relation="co-occurs with", context_ids=set())
        
        # 注入向量
        entity_df = self.matcher.df.set_index('Standard_Entity')
        valid_nodes = set(G.nodes()) & set(entity_df.index)
        for node in valid_nodes:
            try:
                vec = entity_df.loc[node, 'vec_entity']
                if isinstance(vec, pd.Series): vec = vec.iloc[0]
                if isinstance(vec, list): vec = np.array(vec, dtype=np.float32)
                G.nodes[node]['vector'] = vec
            except: pass
        return G

    def get_query_vec(self, query):
        with self.gpu_lock:
            try:
                v = self.graph_embed_model.embed_documents([query])[0]
                return np.array(v, dtype=np.float32)
            except: return None

    def get_chunk_text(self, chunk_id):
        try:
            return str(self.chunk_df.loc[chunk_id, 'text']).replace('\n', ' ')
        except: return ""

    def _get_top_chunks(self, candidate_cids, query_vec, top_k=5):
        valid_cids = [c for c in candidate_cids if c in self.chunk_df.index]
        if not valid_cids or query_vec is None: return valid_cids[:top_k]
        try:
            chunk_matrix = np.stack(self.chunk_df.loc[valid_cids, 'embedding_np'].values)
            q_norm = np.linalg.norm(query_vec)
            c_norms = np.linalg.norm(chunk_matrix, axis=1)
            dot_products = chunk_matrix @ query_vec
            scores = dot_products / (c_norms * q_norm + 1e-9)
            sorted_indices = np.argsort(scores)[::-1][:top_k]
            return [valid_cids[i] for i in sorted_indices]
        except: return valid_cids[:top_k]

    def expand_neighbors(self, source_nodes, query_vec, context_id, intent_weights, visited_set, verbose=False):
        """核心扩展逻辑"""
        w_f, w_s, w_e = intent_weights
        candidate_paths = []
        raw_candidate_cids = set()
        query_norm = np.linalg.norm(query_vec) if query_vec is not None else 1.0

        if verbose:
            print(f"{Colors.BLUE}   🔎 Expanding from {len(source_nodes)} nodes (W_E={w_e:.2f}, W_I={w_f:.2f}, W_S={w_s:.2f}){Colors.ENDC}")

        for u in source_nodes:
            if u not in self.G: continue
            neighbors_scores = []
            
            for v in self.G.neighbors(u):
                if visited_set and v in visited_set: continue
                
                data = self.G[u][v]
                # 1. Semantic Score
                s_sem = 0.0
                if query_vec is not None:
                    vec_v = self.G.nodes[v].get('vector')
                    if vec_v is not None:
                        dot_val = np.dot(query_vec, vec_v)
                        norm_v = np.linalg.norm(vec_v)
                        s_sem = max(0.0, dot_val / (query_norm * norm_v + 1e-9))
                
                # 2. Implicit / Explicit Score
                s_imp = data.get('implicit_score', 0.0)
                is_explicit = 1.0 if data.get('type') == 'explicit' else 0.0
                s_exp = is_explicit * s_sem
                
                # 3. Weighted Sum
                W = (w_e * s_exp) + (w_f * s_imp) + (w_s * s_sem)
                
                # 4. Context Boost
                in_context = context_id in data.get('context_ids', set())
                final_score = W * (1.5 if in_context else 1.0)
                
                if final_score < TokenConfig.MIN_EDGE_SCORE: continue
                neighbors_scores.append((v, final_score, data))

            neighbors_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Top-K selection
            for v, score, data in neighbors_scores[:TokenConfig.TOP_K_NEIGHBORS]:
                candidate_paths.append({'u': u, 'v': v, 'rel': data.get('relation', 'related'), 'score': score})
                if 'chunk_ids' in data:
                    raw_candidate_cids.update(data['chunk_ids'])

        # Rerank chunks
        best_cids = []
        if raw_candidate_cids:
            best_cids = self._get_top_chunks(list(raw_candidate_cids), query_vec, top_k=TokenConfig.MAX_CHUNKS_IN_PROMPT)
            
        return candidate_paths, best_cids

    def get_node_details(self, nodes, context_id, query_vec=None):
        details = {}
        chunk_ids = set()
        entity_df = self.matcher.df.set_index('Standard_Entity')
        
        for n in nodes:
            has_def = False
            if n in entity_df.index:
                try:
                    row = entity_df.loc[n]
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    desc = str(row.get('description', '')).replace('\n', ' ')
                    details[n] = f"{desc}, [Cat: {row.get('category','N/A')}]"
                    if len(desc.strip()) > 0: has_def = True
                except: pass
            
            if (not has_def) or TokenConfig.STAGE0_ADD_CHUNKS:
                if n in self.G:
                    nbr_chunks = set()
                    for nbr in self.G.neighbors(n):
                        edge = self.G[n][nbr]
                        if context_id in edge.get('context_ids', set()):
                            nbr_chunks.update(edge.get('chunk_ids', []))
                    if nbr_chunks:
                        chunk_ids.update(self._get_top_chunks(list(nbr_chunks), query_vec, top_k=3))
                        
        chunks = []
        for cid in list(chunk_ids)[:TokenConfig.STAGE0_MAX_CHUNKS]:
            txt = self.get_chunk_text(cid)
            if txt: chunks.append(f"[Ref {cid}] {txt[:TokenConfig.CHUNK_CHAR_LIMIT]}...")
            
        return details, chunks

# ==========================================
# 4. LangGraph 节点定义 (Nodes)
# ==========================================

def _get_config_res(config):
    return config["configurable"]["resources"], config["configurable"].get("verbose", False)

def node_analyze_intent(state: AgentState, config):
    """Step 1: 意图分析"""
    res, verbose = _get_config_res(config)
    query = state["query"]
    
    with res.gpu_lock:
        try:
            emb = res.intent_embed_model.embed_documents([query])
            emb_t = torch.tensor(np.array(emb), dtype=torch.float32).to(res.device)
            probs = res.intent_model.predict_proba(emb_t, [query])
            weights, strategy = dynamic_weight_modulation(probs, query)
        except:
            weights, strategy = [0.2, 0.6, 0.2], "Default"
            
    if verbose:
        print(f"{Colors.HEADER}🧠 [Intent] Strategy: {strategy}, Weights: {weights}{Colors.ENDC}")
        
    return {
        "intent_weights": weights, "intent_strategy": strategy,
        "current_hop": 0, "is_solved": False, 
        "visited_nodes": [], "relevant_entities": [], 
        "history_facts": [], "accumulated_chunks": [], "entity_memory": {}
    }

def node_semantic_anchoring(state: AgentState, config):
    """Step 2: 锚点识别"""
    res, verbose = _get_config_res(config)
    query = state["query"]
    
    with res.gpu_lock:
        df = res.matcher.link(query, state["context_id"], top_k=10)
        seeds = df['Standard_Entity'].tolist() if not df.empty else []
    
    # 简单筛选前5个作为种子
    filtered = seeds[:8]
    if verbose:
        print(f"{Colors.BLUE}📍 [Anchoring] Seeds found: {filtered}{Colors.ENDC}")
        
    return {"seeds": filtered, "active_nodes": filtered}

def node_check_seeds(state: AgentState, config):
    """Stage 0: 定义检查"""
    res, verbose = _get_config_res(config)
    seeds = state["seeds"]
    if not seeds: return {"is_solved": False}
    
    q_vec = res.get_query_vec(state["query"])
    infos, chunks = res.get_node_details(seeds, state["context_id"], query_vec=q_vec)
    
    # 更新记忆
    mem = state["entity_memory"].copy()
    mem.update(infos)
    
    # Prompt
    def_str = "\n".join([f"- **{k}**: {v}" for k, v in infos.items()])
    chunk_str = "\n".join(chunks) if chunks else "None"
    prompt = f"""
    You are a Fact-Checking Agent.
    Query: "{state['query']}"
    
    Definitions:
    {def_str}
    
    Context:
    {chunk_str}
    
    Instruction: 
    1.  If the answer is deducible from all evidence, answer it.
    2. Output Format:
       - If Answerable: `Final Answer:  [Clean Entity Name / Yes / No / data / etc.]`(concise and directly)`
       - If Not: `Relevant Nodes: [list]`, `Next Hop: [list]`
    """
    
    print("node_check_seeds:",prompt)
    resp = res.llm.invoke(prompt).content
    parsed = parse_llm_decision(resp, valid_scope=seeds)
    
    if verbose:
        print(f"{Colors.GREEN}   💭 [Stage 0 LLM] {parsed}{Colors.ENDC}")

    updates = {
        "entity_memory": mem,
        "accumulated_chunks": list(set(state["accumulated_chunks"] + chunks)),
        "relevant_entities": list(set(state["relevant_entities"] + parsed["relevant_nodes"]))
    }
    
    if parsed["is_final"]:
        updates.update({"final_answer": parsed["answer"], "is_solved": True})
    else:
        # 下一跳逻辑
        next_n = [n for n in parsed["next_nodes"] if n in res.G]
        if not next_n and parsed["relevant_nodes"]:
            next_n = [n for n in parsed["relevant_nodes"] if n in res.G]
        updates["active_nodes"] = next_n
        
    return updates

def node_expand_graph(state: AgentState, config):
    """Stage N: 物理扩展"""
    res, verbose = _get_config_res(config)
    active = state["active_nodes"]
    visited = set(state["visited_nodes"])
    real_active = [n for n in active if n not in visited]
    
    if not real_active: return {"active_nodes": []}
    
    # 调用资源层扩展算法
    q_vec = res.get_query_vec(state["query"])
    paths, cids = res.expand_neighbors(real_active, q_vec, state["context_id"], 
                                      state["intent_weights"], visited, verbose=verbose)
    
    # 提取文本
    new_chunks = []
    for cid in cids:
        txt = res.get_chunk_text(cid)
        if txt: new_chunks.append(f"[Ref {cid}] {txt[:TokenConfig.CHUNK_CHAR_LIMIT]}...")
    
    path_strs = [f"{p['u']} --[{p['rel']}]--> {p['v']}" for p in paths]
    candidates = list(set([p['v'] for p in paths if p['v'] not in visited and p['v'] not in real_active]))
    
    return {
        "visited_nodes": list(visited | set(real_active)),
        "accumulated_chunks": list(set(state["accumulated_chunks"] + new_chunks)),
        "history_facts": list(set(state["history_facts"] + path_strs)),
        "active_nodes": candidates,
        "current_hop": state["current_hop"] + 1
    }

def node_reasoning(state: AgentState, config):
    """Stage N: 逻辑推理"""
    res, verbose = _get_config_res(config)
    
    # 构建 Prompt
    evidence = [f"**{n}**: {state['entity_memory'].get(n, '')[:100]}..." for n in state["relevant_entities"]]
    chunk_txt = "\n".join(state["accumulated_chunks"][-TokenConfig.MAX_CHUNKS_IN_PROMPT:])
    valid_hops = state["active_nodes"][:TokenConfig.MAX_CANDIDATE_POOL]
    
    prompt = f"""
    Graph Reasoning Agent. Hop {state['current_hop']}.
    Query: "{state['query']}"
    
    Known Evidence:
    {chr(10).join(evidence)}
    
    New Context:
    {chunk_txt}
    
    Graph Paths:
    {chr(10).join(state['history_facts'][-10:])}
    
    Valid Next Hops: {valid_hops}
    
    Output Format:
    - If Answerable: `Final Answer: ...`
    - Else: `Relevant Nodes: [...]`, `Next Hop: [...]` (Select from Valid Hops)
    """
    print("node_reasoning:",prompt)
    resp = res.llm.invoke(prompt).content
    parsed = parse_llm_decision(resp, valid_scope=valid_hops)
    
    if verbose:
        print(f"{Colors.GREEN}   💭 [Stage {state['current_hop']} LLM] {parsed}{Colors.ENDC}")
        
    updates = {"relevant_entities": list(set(state["relevant_entities"] + parsed["relevant_nodes"]))}
    
    if parsed["is_final"]:
        updates.update({"final_answer": parsed["answer"], "is_solved": True})
    else:
        next_n = [n for n in parsed["next_nodes"] if n in res.G]
        if not next_n and valid_hops: next_n = valid_hops[:3] # 兜底
        updates["active_nodes"] = next_n
        
    return updates

def node_fallback(state: AgentState, config):
    """兜底"""
    res, verbose = _get_config_res(config)
    prompt = f"""
    Based on logs, answer: "{state['query']}"
    Logs: {state['history_facts']}
    Final Answer:
    """
    ans = res.llm.invoke(prompt).content.replace("Final Answer:", "").strip()
    if verbose: print(f"{Colors.WARNING}⚠️ Fallback triggered.{Colors.ENDC}")
    return {"final_answer": ans, "is_solved": True}

# ==========================================
# 5. 辅助解析函数
# ==========================================
def parse_llm_decision(text, valid_scope=None):
    text = str(text).strip()
    result = {"is_final": False, "answer": "", "relevant_nodes": [], "next_nodes": []}
    
    # 1. 提取列表
    def extract_list(label):
        pattern = re.search(fr"{label}\s*\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
        if pattern:
            content = pattern.group(1).replace('[', '').replace(']', '')
            return [x.strip().strip("'").strip('"') for x in content.split(',') if x.strip()]
        return []

    result["relevant_nodes"] = extract_list("Relevant Nodes:")
    result["next_nodes"] = extract_list("Next Hop:")

    # 2. 提取 Final Answer
    if "Final Answer:" in text:
        raw = text.split("Final Answer:")[-1].strip()
        # 简单防伪
        if not any(x in raw.lower() for x in ["not found", "unknown", "cannot answer"]):
            result["is_final"] = True
            result["answer"] = raw.split('\n')[0].strip()
            return result
            
    # 3. 模糊匹配校正 (Scope Check)
    if valid_scope:
        validated = []
        for n in result["next_nodes"]:
            matches = difflib.get_close_matches(n, valid_scope, n=1, cutoff=0.7)
            if matches: validated.append(matches[0])
        result["next_nodes"] = validated
        
    return result

# ==========================================
# 6. 图构建 (Graph Builder)
# ==========================================
def build_agent_graph():
    def route_seeds(state):
        if state["is_solved"]: return END
        if not state["seeds"]: return "fallback"
        return "expand_graph"

    def route_reasoning(state):
        if state["is_solved"]: return END
        if state["current_hop"] >= state["max_hops"]: return "fallback"
        if not state["active_nodes"]: return "fallback"
        return "expand_graph"

    workflow = StateGraph(AgentState)
    
    workflow.add_node("intent_analysis", node_analyze_intent)
    workflow.add_node("semantic_anchoring", node_semantic_anchoring)
    workflow.add_node("check_seeds", node_check_seeds)
    workflow.add_node("expand_graph", node_expand_graph)
    workflow.add_node("reasoning", node_reasoning)
    workflow.add_node("fallback", node_fallback)
    
    workflow.add_edge(START, "intent_analysis")
    workflow.add_edge("intent_analysis", "semantic_anchoring")
    workflow.add_edge("semantic_anchoring", "check_seeds")
    
    workflow.add_conditional_edges("check_seeds", route_seeds)
    workflow.add_edge("expand_graph", "reasoning")
    workflow.add_conditional_edges("reasoning", route_reasoning)
    workflow.add_edge("fallback", END)
    
    return workflow.compile()

# ==========================================
# 7. 主程序入口
# ==========================================
if __name__ == "__main__":
    # --- 路径配置 (请根据实际情况修改) ---
    DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\hotpot\ds100"
    ADAPT_ROOT = r"D:\Code\jupyter\knowledge_graph\adapt"
    
    QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
    GRAPH_FILE = os.path.join(DATA_ROOT, "graph.csv")
    CHUNK_EMB_FILE = os.path.join(DATA_ROOT, "chunks_with_embeddings.parquet")
    CHUNK_RAW_FILE = os.path.join(DATA_ROOT, "chunk.csv")
    PARQUET_FILE = os.path.join(DATA_ROOT, "concepts_merged_with_vectors.parquet")
    PROX_FILE = os.path.join(DATA_ROOT, "contextual_proximity.csv")
    MODEL_FILE = os.path.join(ADAPT_ROOT, "intent_classifier_struct.pth")

    # --- 1. 数据加载 ---
    try:
        print("📂 Loading data...")
        df_qa = pd.read_csv(QA_FILE, sep="|")
        df_graph = pd.read_csv(GRAPH_FILE, sep="|")
        
        if os.path.exists(CHUNK_EMB_FILE):
            df_chunk = pd.read_parquet(CHUNK_EMB_FILE)
        else:
            df_chunk = pd.read_csv(CHUNK_RAW_FILE, sep="|")
            
        df_prox = pd.read_csv(PROX_FILE, sep="|") if os.path.exists(PROX_FILE) else None
    except Exception as e:
        print(f"❌ Load failed: {e}")
        sys.exit(1)

    # --- 2. 初始化资源与图 ---
    print("🚀 Initializing Resources & Graph...")
    resources = ID_SGTR_Resources(MODEL_FILE, PARQUET_FILE, df_graph, df_chunk, df_prox)
    app = build_agent_graph()

    # --- 3. 并发处理 Wrapper ---
    def process_query_wrapper(i: int, row: pd.Series) -> Tuple[int, Any]:
        q = row['question']
        ctx = row['context_id']
        
        # ===> 这里控制 VERBOSE <===
        # True: 打印详细日志 (单线程调试建议开启)
        # False: 静默 (多线程跑批建议关闭)
        VERBOSE_MODE = True  
        
        initial_state = {
            "query": q, "context_id": ctx, "max_hops": 3,
            "visited_nodes": [], "history_facts": [], "accumulated_chunks": [],
            "relevant_entities": [], "entity_memory": {}
        }
        
        # 将 verbose 注入 config
        config = {"configurable": {"resources": resources, "verbose": VERBOSE_MODE}}
        
        try:
            # LangGraph 执行
            final_state = app.invoke(initial_state, config=config)
            
            ans = final_state.get("final_answer", "No Answer")
            strat = final_state.get("intent_strategy", "Default")
            if final_state.get("is_solved"):
                strat += f" -> Solved (Hop {final_state.get('current_hop')})"
            else:
                strat += " -> Fallback"
                
        except Exception as e:
            print(f"Error on query {i}: {e}")
            ans, strat = "Error", "Error"

        return i, {
            "question": q, "gold_answer": row['answer'],
            "pred_answer": ans, "strategy": strat, "context_id": ctx
        }

    # --- 4. 执行 ---
    target_data = df_qa.sample(5) # 测试5条
    print(f"\n📝 Processing {len(target_data)} queries...")
    
    results = parallel_llm_processor(
        dataframe=target_data,
        processing_func=process_query_wrapper,
        start_message="Running Agent...",
        max_workers=1, # 调试建议设为 1，跑批可设为 4-8
        max_retries=5,
        initial_delay=1
    )
    
    # --- 5. 保存 ---
    results.sort(key=lambda x: x[0])
    output_file = os.path.join(current_dir, "test_langgraph_full_result.csv")
    pd.DataFrame([r[1] for r in results]).to_csv(output_file, index=False, sep="|")
    print(f"✅ Done. Results saved to {output_file}")