# import torch
# import numpy as np
# import pandas as pd
# import ast
# import re
# import os
# import sys
# import threading
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ==========================================
# # 0. 辅助类：控制台颜色输出
# # ==========================================
# class Colors:
#     HEADER = '\033[93m'
#     BLUE = '\033[94m'
#     CYAN = '\033[96m'
#     GREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'

# # ==========================================
# # 1. 环境与路径配置
# # ==========================================
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# # 引入你的模型加载工具 (请确保路径和你的项目一致)
# from utils import get_embeddings_model, get_chat_model
# from helper import parallel_llm_processor

# # ==========================================
# # 2. 核心引擎: Naive Vector RAG (Dense-Only)
# # ==========================================
# class Naive_Vector_RAG_Engine:
#     def __init__(self, chunk_df, device=None):
#         """
#         初始化纯向量检索引擎
#         """
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"🔧 初始化 Naive RAG 引擎 (Device: {self.device})...")

#         self.gpu_lock = threading.Lock()

#         # -------------------------------------------------------
#         # Module 1: 数据预处理 (仅保留 Chunk 向量逻辑)
#         # -------------------------------------------------------
#         print("📦 [1/3] 数据预处理与 Context 索引构建...")
#         self.chunk_df = chunk_df.copy()
#         if 'chunk_id' in self.chunk_df.columns:
#             self.chunk_df.set_index('chunk_id', inplace=True)

#         def parse_vec_safe(x):
#             if isinstance(x, np.ndarray): return x.astype(np.float32)
#             if isinstance(x, list): return np.array(x, dtype=np.float32)
#             if isinstance(x, str):
#                 try:
#                     if x.strip().startswith('['):
#                         return np.array(ast.literal_eval(x), dtype=np.float32)
#                 except:
#                     return None
#             return None

#         # 确保 embedding 列是 numpy 数组
#         if 'embedding_np' not in self.chunk_df.columns:
#             tqdm.pandas(desc="Parsing Vectors")
#             self.chunk_df['embedding_np'] = self.chunk_df['embedding'].progress_apply(parse_vec_safe)

#         # 建立 context_id 到 chunk_id 的索引，加速查找
#         self.chunk_dict_by_ctx = {}
#         if 'context_id' in self.chunk_df.columns:
#             self.chunk_dict_by_ctx = self.chunk_df.groupby('context_id')['text'].apply(lambda x: x.index.tolist()).to_dict()

#         # -------------------------------------------------------
#         # Module 2: 加载向量模型 (Embedding)
#         # -------------------------------------------------------
#         print("📥 [2/3] 加载 Embedding 模型 (BGE-M3)...")
#         self.embed_model = get_embeddings_model(dimensions=1024)

#         # -------------------------------------------------------
#         # Module 3: 加载推理大模型 (LLM)
#         # -------------------------------------------------------
#         print("🤖 [3/3] 初始化推理大模型 (Qwen3-8B)...")
#         self.llm = get_chat_model(task_type="kg_query")

#     def _get_top_chunks_dense_only(self, candidate_cids, query_vec, top_k=3):
#         """
#         纯稠密向量 (Dense Vector) 相似度计算
#         """
#         valid_cids = [c for c in candidate_cids if c in self.chunk_df.index]
#         if not valid_cids or query_vec is None: 
#             return valid_cids[:top_k]

#         try:
#             # 1. 提取内容向量矩阵
#             content_matrix = np.stack(self.chunk_df.loc[valid_cids, 'embedding_np'].values)
            
#             # 2. 计算余弦相似度 (纯 Content，移除了原有的 Title 加权逻辑，保证纯粹性)
#             q_norm = np.linalg.norm(query_vec)
#             c_norms = np.linalg.norm(content_matrix, axis=1)
#             cosine_sims = (content_matrix @ query_vec) / (c_norms * q_norm + 1e-9)
            
#             # 3. 排序并取 Top-K
#             sorted_indices = np.argsort(cosine_sims)[::-1]
#             return [valid_cids[i] for i in sorted_indices[:top_k]]

#         except Exception as e:
#             print(f"⚠️ Vector retrieval error: {e}")
#             return valid_cids[:top_k]

#     def _get_chunk_text(self, chunk_id):
#         try:
#             return str(self.chunk_df.loc[chunk_id, 'text']).replace('\n', ' ')
#         except: 
#             return ""

#     def solve(self, query, context_id, verbose=False, top_k=3):
#         """
#         端到端检索与生成
#         """
#         if verbose:
#             print(f"{Colors.HEADER}\n=================================================={Colors.ENDC}")
#             print(f"{Colors.BOLD}🔍 [Naive RAG] Query: {query}{Colors.ENDC}")

#         # 1. 向量化 Query (加锁防 CUDA 崩溃)
#         query_vec = None
#         with self.gpu_lock:
#             try:
#                 query_vec = self.embed_model.embed_query(query)
#                 query_vec = np.array(query_vec, dtype=np.float32)
#             except Exception as e: 
#                 print(f"⚠️ Query Embedding Failed: {e}")
#                 return "Error: Embedding failed", "Naive-Vector"

#         # 2. 获取候选 Chunk IDs
#         candidate_cids = []
#         if context_id is not None and context_id in self.chunk_dict_by_ctx:
#             candidate_cids = self.chunk_dict_by_ctx[context_id]

#         # 3. 纯向量检索 Top-K (默认 3)
#         top_cids = self._get_top_chunks_dense_only(candidate_cids, query_vec, top_k=top_k)
        
#         # 4. 组装 Prompt 文本
#         top_chunks_text = []
#         for cid in top_cids:
#             txt = self._get_chunk_text(cid)
#             clean_txt = txt[:1000].replace('\n', ' ')  # 限制单块长度
#             top_chunks_text.append(f"[Ref {cid}] {clean_txt}")
#             if verbose:
#                 print(f"{Colors.CYAN}  -> Selected Chunk [{cid}]{Colors.ENDC}")

#         context_str = "\n".join(top_chunks_text) if top_chunks_text else "No specific context found."

#         # 3. 构建 Prompt (【核心修改】：彻底删除了 seeds 背景知识)
#         prompt = f"""
# You are a high-precision QA system answering a complex question. You must answer based DIRECTLY on the provided Reference Text.

# ### User Query
# "{query}"

# ### Reference Context (Primary Evidence)
# {context_str}

# ### Task
# Answer the query using ONLY the information above. Read very carefully, watching out for distractor entities with similar names.

# ### Strict Rules
# 1. **Format**:
#    - If the answer is explicitly in the text, extract the exact entity/value.
#    - If it's a Yes/No question, answer "Yes" or "No".
# 2. **Output**:  
#    - Keep output minimal. 
#    - `Final Answer: [Clean Entity Name / Yes / No / data / etc.]`

# ### ⛔ OUTPUT RESTRICTIONS
# - **NO** sentences or paragraphs.
# - **NO** explanations.
# """
        
#         # 6. 调用 LLM
#         resp = self.llm.invoke(prompt).content.strip()
#         if "Final Answer:" in resp:
#             resp = resp.split("Final Answer:")[-1].strip()
        
#         if verbose:
#             print(f"{Colors.GREEN}💡 [LLM Answer]: {resp}{Colors.ENDC}")

#         return resp, "Naive-Vector-Top3"


# # ==========================================
# # 3. 主程序入口 (多线程并行处理)
# # ==========================================
# if __name__ == "__main__":
#     # 配置你的数据路径 (按照你原有的结构)
#     DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\2wiki\ds1000"
    
#     QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
#     CHUNK_EMB_FILE = os.path.join(DATA_ROOT, "chunks_with_embeddings.parquet")
#     CHUNK_RAW_FILE = os.path.join(DATA_ROOT, "chunk.csv")

#     # 1. 数据加载
#     try:
#         df_qa = pd.read_csv(QA_FILE, sep="|")
        
#         if os.path.exists(CHUNK_EMB_FILE):
#             print(f"📦 加载带向量的 Chunk 数据: {CHUNK_EMB_FILE}")
#             df_chunk = pd.read_parquet(CHUNK_EMB_FILE)
#         else:
#             print(f"⚠️ 未找到 Embedding Parquet，加载原始 CSV: {CHUNK_RAW_FILE}")
#             df_chunk = pd.read_csv(CHUNK_RAW_FILE, sep="|")
#     except Exception as e:
#         print(f"❌ 数据加载失败: {e}")
#         sys.exit(1)

#     # 2. 初始化纯向量引擎
#     engine = Naive_Vector_RAG_Engine(chunk_df=df_chunk)

#     # 3. 选择测试数据范围 (跑全量 1000 条)
#     target_data = df_qa.head(1000)
#     print(f"\n📝 开始并发处理 {len(target_data)} 条查询 (纯向量模式)...")
    
#     def process_query_wrapper(i: int, row: pd.Series):
#         q = row['question']
#         ctx = row['context_id']
#         gold = row['answer']
        
#         # 调用纯向量检索 (默认 Top-3)
#         pred_answer, strategy = engine.solve(q, ctx, verbose=False, top_k=3)
                
#         return i, {
#             "question": q,
#             "gold_answer": gold,
#             "pred_answer": pred_answer,
#             "strategy": strategy,
#             "context_id": ctx
#         }
        
#     # 4. 启动多线程
#     processed_results = parallel_llm_processor(
#         dataframe=target_data,
#         processing_func=process_query_wrapper,
#         start_message="🚀 启动多线程纯向量推理...",
#         max_workers=5,
#         max_retries=6,
#         initial_delay=2
#     )

#     # 3. 整理并保存结果
#     processed_results.sort(key=lambda x: x[0])
#     final_data = [item[1] for item in processed_results]
    
#     output_path = os.path.join(current_dir, "query_results_naive_rag_dense_only_Qwen3-8B_top3.csv")
#     pd.DataFrame(final_data).to_csv(output_path, index=False, sep="|")
#     print(f"\n✅ 处理完成！Naive RAG 结果已保存至: {output_path}")




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
# 0. 辅助类：控制台颜色输出
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
# 1. 环境与路径配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 引入你的模型加载工具
from utils import get_embeddings_model, get_chat_model
from helper import parallel_llm_processor

# ==========================================
# 2. 核心引擎: Naive Vector RAG (Dense-Only, Global Mode)
# ==========================================
class Naive_Vector_RAG_Engine:
    def __init__(self, chunk_df, device=None):
        """
        初始化纯向量检索引擎
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 初始化 Naive RAG 引擎 (Device: {self.device})...")

        self.gpu_lock = threading.Lock()

        # -------------------------------------------------------
        # Module 1: 数据预处理 (仅保留 Chunk 向量逻辑)
        # -------------------------------------------------------
        print("📦 [1/3] 数据预处理与 Context 索引构建...")
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

        # 确保 embedding 列是 numpy 数组
        if 'embedding_np' not in self.chunk_df.columns:
            tqdm.pandas(desc="Parsing Vectors")
            self.chunk_df['embedding_np'] = self.chunk_df['embedding'].progress_apply(parse_vec_safe)

        # 建立 context_id 到 chunk_id 的索引（备用，如果需要退回局部检索）
        self.chunk_dict_by_ctx = {}
        if 'context_id' in self.chunk_df.columns:
            self.chunk_df['context_id'] = self.chunk_df['context_id'].astype(str)
            self.chunk_dict_by_ctx = self.chunk_df.groupby('context_id')['text'].apply(lambda x: x.index.tolist()).to_dict()

        # -------------------------------------------------------
        # Module 2: 加载向量模型 (Embedding)
        # -------------------------------------------------------
        print("📥 [2/3] 加载 Embedding 模型 (BGE-M3)...")
        self.embed_model = get_embeddings_model(dimensions=1024)

        # -------------------------------------------------------
        # Module 3: 加载推理大模型 (LLM)
        # -------------------------------------------------------
        print("🤖 [3/3] 初始化推理大模型 (Qwen3-8B)...")
        self.llm = get_chat_model(task_type="kg_query")

    def _get_top_chunks_dense_only(self, candidate_cids, query_vec, top_k=3):
        """
        纯稠密向量 (Dense Vector) 全库相似度计算
        """
        valid_cids = [c for c in candidate_cids if c in self.chunk_df.index]
        if not valid_cids or query_vec is None: 
            return valid_cids[:top_k]

        try:
            # 1. 提取内容向量矩阵 (全库检索时这里会提取所有 Chunk 的矩阵)
            content_matrix = np.stack(self.chunk_df.loc[valid_cids, 'embedding_np'].values)
            
            # 2. 计算余弦相似度
            q_norm = np.linalg.norm(query_vec)
            c_norms = np.linalg.norm(content_matrix, axis=1)
            cosine_sims = (content_matrix @ query_vec) / (c_norms * q_norm + 1e-9)
            
            # 3. 排序并取 Top-K
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
        端到端检索与生成 (无缝支持 Global 或 Local)
        """
        if verbose:
            print(f"{Colors.HEADER}\n=================================================={Colors.ENDC}")
            print(f"{Colors.BOLD}🔍 [Naive RAG] Query: {query}{Colors.ENDC}")

        # 1. 向量化 Query (加锁防 CUDA 崩溃)
        query_vec = None
        with self.gpu_lock:
            try:
                query_vec = self.embed_model.embed_query(query)
                query_vec = np.array(query_vec, dtype=np.float32)
            except Exception as e: 
                print(f"⚠️ Query Embedding Failed: {e}")
                return "Error: Embedding failed", "Naive-Vector"

        # 2. ✅ 【核心修改】：获取候选 Chunk IDs，彻底放开隔离！
        candidate_cids = []
        if context_id is None or str(context_id).lower() == 'global':
            # 🌍 全局模式：把知识库里所有的 Chunk ID 全拿出来大海捞针
            candidate_cids = self.chunk_df.index.tolist()
        elif str(context_id) in self.chunk_dict_by_ctx:
            # 🏠 局部模式（仅作备用兼容）
            candidate_cids = self.chunk_dict_by_ctx[str(context_id)]
        else:
            # 兜底全库
            candidate_cids = self.chunk_df.index.tolist()

        # 3. 纯向量检索 Top-K 全库大比拼 (默认 3)
        top_cids = self._get_top_chunks_dense_only(candidate_cids, query_vec, top_k=top_k)
        
        # 4. 组装 Prompt 文本
        top_chunks_text = []
        for cid in top_cids:
            txt = self._get_chunk_text(cid)
            clean_txt = txt[:1000].replace('\n', ' ')  # 限制单块长度
            top_chunks_text.append(f"[Ref {cid}] {clean_txt}")
            if verbose:
                print(f"{Colors.CYAN}  -> Selected Chunk [{cid}]{Colors.ENDC}")

        context_str = "\n".join(top_chunks_text) if top_chunks_text else "No specific context found."

        # 5. 构建 Prompt (严格限制基于参考资料作答)
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
        
        # 6. 调用 LLM
        resp = self.llm.invoke(prompt).content.strip()
        if "Final Answer:" in resp:
            resp = resp.split("Final Answer:")[-1].strip()
        
        if verbose:
            print(f"{Colors.GREEN}💡 [LLM Answer]: {resp}{Colors.ENDC}")

        return resp, "Naive-Vector-Global-Top3"


# ==========================================
# 3. 主程序入口 (多线程并行处理)
# ==========================================
if __name__ == "__main__":
    # 配置你的数据路径
    DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\2wiki\ds1000"
    
    QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
    CHUNK_EMB_FILE = os.path.join(DATA_ROOT, "chunks_with_embeddings.parquet")
    CHUNK_RAW_FILE = os.path.join(DATA_ROOT, "chunk.csv")

    # 1. 数据加载
    try:
        df_qa = pd.read_csv(QA_FILE, sep="|")
        
        if os.path.exists(CHUNK_EMB_FILE):
            print(f"📦 加载带向量的 Chunk 数据: {CHUNK_EMB_FILE}")
            df_chunk = pd.read_parquet(CHUNK_EMB_FILE)
        else:
            print(f"⚠️ 未找到 Embedding Parquet，加载原始 CSV: {CHUNK_RAW_FILE}")
            df_chunk = pd.read_csv(CHUNK_RAW_FILE, sep="|")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        sys.exit(1)

    # 2. 初始化纯向量引擎
    engine = Naive_Vector_RAG_Engine(chunk_df=df_chunk)

    # 3. 选择测试数据范围
    target_data = df_qa.head(1000)
    print(f"\n📝 开始并发处理 {len(target_data)} 条查询 (纯向量全局检索模式)...")
    
    def process_query_wrapper(i: int, row: pd.Series):
        q = row['question']
        ctx = row['context_id'] # 留着记在日志里比对用
        gold = row['answer']
        
        # ✅ 【核心修改】：强制传入 'global'，彻底抛弃原有 Context 隔离！
        pred_answer, strategy = engine.solve(q, context_id="global", verbose=False, top_k=3)
                
        return i, {
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "strategy": strategy,
            "context_id": ctx # 保留原有的 context_id 以便评估脚本计算分数
        }
        
    # 4. 启动多线程
    processed_results = parallel_llm_processor(
        dataframe=target_data,
        processing_func=process_query_wrapper,
        start_message="🚀 启动多线程纯向量推理...",
        max_workers=5,
        max_retries=6,
        initial_delay=2
    )

    # 5. 整理并保存结果
    processed_results.sort(key=lambda x: x[0])
    final_data = [item[1] for item in processed_results]
    
    output_path = os.path.join(current_dir, "query_results_naive_rag_dense_only_GLOBAL_Qwen3-8B_top3.csv")
    pd.DataFrame(final_data).to_csv(output_path, index=False, sep="|")
    print(f"\n✅ 处理完成！全局 Naive RAG 结果已保存至: {output_path}")