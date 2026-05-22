# D:\Code\jupyter\knowledge_graph\SQuAD2.0\kg_query_engine.py
import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import random
# 从 langchain 导入必要的模块
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage

# 添加父目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具函数
from utils import get_embeddings_model, get_llm_model
from zhipu import extractConcepts,classify_query_type
from prompt import KG_QA_GENERATION_SYS_PROMPT
from helper import call_llm_with_retry,get_unique_chunks

class GassQueryEngine:
    """
    一个基于图谱增强的语义搜索引擎。
    它加载由知识图谱提取流程生成的四个核心CSV文件，
    并提供一个接口来查询这些数据。
    """
    def __init__(self, data_directory: str):
        print("Initializing GASS Query Engine...")
        self.data_path = Path(data_directory)
        self.chunks_df = None
        self.graph_df = None
        self.contextual_df = None
        self.semantic_df = None
        
        # 初始化模型
        self.llm_model = get_llm_model(model='glm-4-flash-250414') # 用于 query 扩展
        self.llm_model_answer = get_llm_model(model='GLM-Z1-Flash',api_key=os.getenv('gerenate_answer')) # 用于最终答案生成
        self.embeddings_model = get_embeddings_model()
        
        self._load_data()
        self._prepare_data()
        # --- 新增: 动态权重配置 ---
        self.WEIGHT_MAP = {
            "Factual": {"Freq": 0.20, "Semantic": 0.65, "Edge": 0.15},
            "Conceptual": {"Freq": 0.10, "Semantic": 0.80, "Edge": 0.10},
            "Relational": {"Freq": 0.10, "Semantic": 0.30, "Edge": 0.60},
            "Default": {"Freq": 0.20, "Semantic": 0.70, "Edge": 0.10},

        }
        print("动态权重映射表已加载。")
        print("Engine Ready.")
        
    def _load_data(self):
        """加载所有必需的CSV文件到Pandas DataFrame中。"""
        try:
            print(f"Loading data from: {self.data_path}")
            self.chunks_df = pd.read_csv(self.data_path / "chunk.csv", sep="|")
            # 确保 chunks_df 以 chunk_id 为索引
            if 'chunk_id' in self.chunks_df.columns:
                 self.chunks_df.set_index('chunk_id', inplace=True)
            self.graph_df = pd.read_csv(self.data_path / "graph.csv", sep="|")
            self.contextual_df = pd.read_csv(self.data_path / "contextual_proximity.csv", sep="|")
            self.semantic_df = pd.read_csv(self.data_path / "semantic_enhancements.csv", sep=",")
            print("所有数据文件加载成功。")
        except FileNotFoundError as e:
            print(f"错误: {e}. 请确保所有必需的CSV文件都在指定的目录中。")
            raise

    def _prepare_data(self):
        """对加载的数据进行预处理和优化，包括嵌入向量的缓存和计算。"""
        # --- 优化：预处理核心图谱 (graph_df) ---
        print("Preprocessing core graph edges for fast lookup...")
        if self.graph_df is not None and not self.graph_df.empty:
            # 规范化 (小写) 节点
            graph_edges = self.graph_df.copy()
            graph_edges['node_1_lower'] = graph_edges['node_1'].astype(str).str.lower()
            graph_edges['node_2_lower'] = graph_edges['node_2'].astype(str).str.lower()
            
            # 创建规范化的关系键 (frozenset 确保 (A, B) 和 (B, A) 视为同一关系)
            graph_edges['relationship_key'] = graph_edges.apply(
                lambda row: frozenset({row['node_1_lower'], row['node_2_lower']}), axis=1
            )
            
            # 提取唯一的规范化键
            self.core_graph_keys: Set[frozenset] = set(graph_edges['relationship_key'].unique())
            print(f"核心图谱中预处理了 {len(self.core_graph_keys)} 个唯一的规范化关系键。")
        else:
            self.core_graph_keys = set()
            print("核心图谱为空，跳过边查找优化。")
        EMBEDDING_CACHE_FILE = self.data_path / "semantic_df_with_embeddings.parquet"
        
        # 基础数据清理和索引设置
        if 'node' in self.semantic_df.columns:
            self.semantic_df['node'] = self.semantic_df['node'].fillna('').astype(str)
            self.semantic_df.set_index('node', inplace=True)
            self.semantic_df.index = self.semantic_df.index.str.lower()
            
        # 将字符串形式的列表转换为真实的Python列表
        for col in ['hyper_concepts', 'synonyms', 'commonsense_relations']:
            if col in self.semantic_df.columns:
                self.semantic_df[col] = self.semantic_df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else []
                )

        # 缓存检查和加载逻辑
        if EMBEDDING_CACHE_FILE.exists():
            print(f"检测到缓存文件: {EMBEDDING_CACHE_FILE}，正在加载嵌入向量...")
            try:
                cached_df = pd.read_parquet(EMBEDDING_CACHE_FILE)
                cached_df.index = cached_df.index.astype(str).str.lower()
                
                self.semantic_df = cached_df
                
                for col in ['hyper_concepts', 'synonyms', 'commonsense_relations']:
                    if col in self.semantic_df.columns:
                        self.semantic_df[col] = self.semantic_df[col].apply(
                            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else x
                        )

                print("嵌入向量加载成功，跳过计算。")
                return
            except Exception as e:
                print(f"警告: 加载缓存文件失败 ({e})，将重新计算嵌入向量。")
        
        # 嵌入向量计算逻辑
        print("未找到缓存或加载失败，开始批量计算并缓存 semantic_df 中所有节点的嵌入向量...")
        
        node_names = self.semantic_df.index.tolist()
        all_descriptions = [self._get_description_for_node(node) for node in node_names]
        
        BATCH_SIZE = 64 
        all_embeddings = []
        
        for i in range(0, len(all_descriptions), BATCH_SIZE):
            batch_descriptions = all_descriptions[i:i + BATCH_SIZE]
            print(f"  > 正在处理批次: {i} 到 {i + len(batch_descriptions) - 1}")
            
            clean_batch = [str(desc) for desc in batch_descriptions]
            final_batch_for_api = [desc if desc else " " for desc in clean_batch]

            try:
                batch_embeddings = self.embeddings_model.embed_documents(final_batch_for_api)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"批次 {i} 嵌入失败: {e}")
                raise

        # 将嵌入向量存储到 semantic_df 中
        if len(all_embeddings) != len(node_names):
            raise ValueError(f"错误: 计算出的嵌入向量数量 ({len(all_embeddings)}) 与节点数量 ({len(node_names)}) 不匹配。")

        self.semantic_df['embedding'] = pd.Series([np.array(e) for e in all_embeddings], index=node_names)
        
        # 保存包含嵌入向量的 DataFrame 到 Parquet 文件
        print(f"嵌入向量计算完成，正在保存到缓存文件: {EMBEDDING_CACHE_FILE}")
        try:
            df_to_save = self.semantic_df.copy()
            for col in ['context_id','hyper_concepts', 'synonyms', 'commonsense_relations']:
                if col in df_to_save.columns:
                    df_to_save[col] = df_to_save[col].astype(str)
                    
            df_to_save.to_parquet(EMBEDDING_CACHE_FILE, index=True)
            print("缓存保存成功。")
        except Exception as e:
            print(f"警告: 嵌入缓存保存失败 ({e})，下次启动仍需重新计算。")
            
    def _get_description_for_node(self, node: str) -> str:
        """
        从 semantic_df 中汇集一个节点的描述文本，进行严格截断，
        并确保返回的文本是标准字符串类型和安全编码。
        """
        MAX_DESCRIPTION_LENGTH = 2000 
        
        try:
            if node not in self.semantic_df.index:
                return ""
            
            row = self.semantic_df.loc[node]
            description_parts = []
            
            if 'hyper_concepts' in self.semantic_df.columns and isinstance(row['hyper_concepts'], list):
                for hc in row['hyper_concepts']:
                    if isinstance(hc, dict) and 'description' in hc:
                        description_parts.append(hc['description'])
            
            if 'commonsense_relations' in self.semantic_df.columns and isinstance(row['commonsense_relations'], list):
                description_parts.extend(row['commonsense_relations'])
            
            full_description = " ".join(
                str(part).strip() for part in description_parts if str(part).strip()
            )
            
            full_description = full_description.encode('utf-8', 'ignore').decode('utf-8')
            
            if len(full_description) > MAX_DESCRIPTION_LENGTH:
                return full_description[:MAX_DESCRIPTION_LENGTH]
            return full_description
            
        except Exception as e:
            print(f"Error processing node {node} in _get_description_for_node: {e}")
            return ""

    def _expand_query(self, query: str, top_k: int = 5, verbose: bool = False) -> Set[str]:
        """扩展查询词，返回一组小写、标准化的实体词汇集。"""
        if not query:
            return set()
            
        concepts = extractConcepts(query, model=self.llm_model) 
        if verbose:
            print("扩展前的查询词", concepts)
        # expanded_terms = {term.strip().lower() for term in re.split(r'\s+', query) if term.strip()}
            
        if not concepts:
            # 兼容处理：如果 LLM 未返回概念，则回退到简单的空格分词
            expanded_terms = {term.strip().lower() for term in re.split(r'\s+', query) if term.strip()}
            return expanded_terms

        sorted_concepts = sorted(
            concepts, 
            key=lambda x: x.get('importance', 0), 
            reverse=True
        )
        selected_entities = sorted_concepts[:top_k]
        entity_names = [entity_dict['entity'].lower() for entity_dict in selected_entities]
        expanded_terms = set(entity_names)

        for node, row in self.semantic_df.iterrows():
            if node.lower() in expanded_terms:
                if 'synonyms' in row and isinstance(row['synonyms'], list):
                    expanded_terms.update({s.lower() for s in row['synonyms']})
            else:
                if 'synonyms' in row and isinstance(row['synonyms'], list):
                    node_synonyms = {s.lower() for s in row['synonyms']}
                    if any(core_entity in node_synonyms for core_entity in entity_names):
                        expanded_terms.add(node.lower())
        
        return expanded_terms
        
    def _summarize_evidence(self, text: str, MAX_CHARACTERS: int = 1500) -> str:
        """总结证据文本，确保不超过最大长度并保持可读性。"""
        text = str(text).strip()
        if not text:
            return ""
        if len(text) <= MAX_CHARACTERS:
            return text
            
        truncated_text = text[:MAX_CHARACTERS]
        last_space_index = truncated_text.rfind(' ')
        last_period_index = truncated_text.rfind('.')
        last_other_punctuation = max(
            truncated_text.rfind('!'), 
            truncated_text.rfind('?'),
            truncated_text.rfind(','),
            truncated_text.rfind(';')
        )
        
        split_point = max(last_space_index, last_period_index, last_other_punctuation)
        if split_point > (MAX_CHARACTERS * 0.9):
            return truncated_text[:split_point].strip() + "..."
        else:
            return truncated_text + "..."

    def _generate_answer(self, query: str, relationships: List[Dict]) -> str:
        """生成最终答案"""
        if not relationships:
            return "对不起，我无法根据现有信息找到相关答案。"

        context = ""
        for rel in relationships:
            context += f"关系: {rel['query_node']} ↔️ {rel['related_node']}\n"
            if rel['description']:
                # 确保 description 在加入 context 前是单行文本
                description = rel['description'].replace('\n', ' ').replace('\r', '') 
                context += f"描述: {description}\n"
            context += f"证据: {rel['evidence']}\n\n"
            
        system_message = SystemMessage(content=KG_QA_GENERATION_SYS_PROMPT)
        user_message = HumanMessage(
            content=f"上下文信息:\n{context}\n\n用户问题: {query}"
        )
            
        messages = [system_message, user_message]
            
        # 调用新的封装函数，使用 answer 专用的 LLM
        final_answer = call_llm_with_retry(
            llm_model=self.llm_model_answer, 
            messages=messages, 
            query=query, 
            max_retries=5, 
            initial_delay=1.0
        )
            
        return final_answer
        
    def _retrieve_from_documents(self, query_string: str, expanded_terms: set, doucment_top_k: int, verbose: bool, context_id: int) -> list:
        """
        纯文本语义回退搜索：根据查询字符串，从指定 context_id 的 Chunk (文档ID) 中检索最相关的文本。
        返回格式化后的 LLM 上下文列表。
        """
        all_relationships_for_llm = []
        
        try:
            chunks_embed_path = self.data_path / "chunks_with_embeddings.parquet"
            if not chunks_embed_path.exists():
                print("错误: 找不到 chunks_with_embeddings.parquet，无法执行回退搜索。")
                return []

            chunks_with_embeddings_df = pd.read_parquet(chunks_embed_path) 
            
            # --- 关键修改 1: 限制到指定的 context_id (文档ID) ---
            if context_id != -1:
                # 根据 'context_id' 列（即文档ID）进行过滤，只保留属于指定文档的 Chunk
                chunks_to_search_df = chunks_with_embeddings_df[
                    chunks_with_embeddings_df['context_id'] == context_id
                ].copy()
                
                if chunks_to_search_df.empty:
                    print(f"警告: 在回退搜索中，找不到指定的文档 ID ({context_id}) 对应的任何 Chunk。")
                    return []
            else:
                # 如果 context_id 为 -1，则在整个 DataFrame 上搜索
                chunks_to_search_df = chunks_with_embeddings_df
                
            # --- 执行语义搜索 ---
            query_embedding = self.embeddings_model.embed_query(query_string)
            query_vec = np.array(query_embedding)
            
            chunk_embeddings_matrix = np.stack(chunks_to_search_df['embedding'].to_numpy())
            
            # 归一化并计算余弦相似度
            norm_query = np.linalg.norm(query_vec)
            chunk_norms = np.linalg.norm(chunk_embeddings_matrix, axis=1)
            
            normalized_chunk_vectors = np.zeros_like(chunk_embeddings_matrix)
            valid_chunks_mask = chunk_norms > 1e-6 # 避免除以零
            normalized_chunk_vectors[valid_chunks_mask] = chunk_embeddings_matrix[valid_chunks_mask] / chunk_norms[valid_chunks_mask, np.newaxis]
            
            normalized_query_vec = query_vec / norm_query if norm_query > 1e-6 else np.zeros_like(query_vec)
            
            chunk_scores_array = normalized_chunk_vectors @ normalized_query_vec
            
            chunks_to_search_df['semantic_score'] = chunk_scores_array
            actual_top_k = min(doucment_top_k, len(chunks_to_search_df))
            
            # 从搜索结果中取出 top_k
            top_chunks_df = chunks_to_search_df.sort_values(by='semantic_score', ascending=False).head(actual_top_k)

            # --- 统一结果处理 ---
            for chunk_id, row in top_chunks_df.iterrows(): # chunk_id 是 DataFrame 的索引
                evidence_text = row['text']
                # 假设 self._summarize_evidence 方法存在
                summary = self._summarize_evidence(evidence_text) 
                chunk_id_str = str(chunk_id) 
                
                score = row['semantic_score'] 
                
                llm_entry = {
                    "query_node": f"Text Evidence Block {chunk_id_str}",
                    "related_node": f"Score: {score:.4f}",
                    "description": f"Highly relevant context snippet derived from chunk {chunk_id_str}. (Doc ID: {row['context_id']})",
                    "evidence": summary
                }
                all_relationships_for_llm.append(llm_entry)
                
                if verbose:
                    print(f"📚 证据块 ID: {chunk_id_str} (Score: {score:.4f}, 文档ID: {row['context_id']})")
                    print(f"  * 证据: \"...{summary}...\"")
                    print("-" * 70)
                    
            return all_relationships_for_llm

        except Exception as e:
            print(f"回退到纯文本搜索时发生严重错误: {e}")
            return []
            
    def _retrieve_from_graph(self, expanded_terms: set, verbose: bool, context_id: int) -> pd.DataFrame:
        """
        仅执行基于扩展词的图谱遍历和关系去重。
        新增：如果 context_id != -1，则额外过滤，只保留包含该 context_id (文档ID) 的关系。
        返回一个已经去重但尚未进行语义评分的 DataFrame。
        """
        # 1. 基于 expanded_terms 进行初步过滤
        # 假设 self.contextual_df 存在
        mask = self.contextual_df['node_1'].str.lower().isin(expanded_terms) | \
               self.contextual_df['node_2'].str.lower().isin(expanded_terms)
        mask &= (self.contextual_df['node_1'].str.lower() != self.contextual_df['node_2'].str.lower())
        
        results_df = self.contextual_df[mask].copy()

        if results_df.empty:
            return results_df
        
        # --- 关键修改 2: 限制到指定的 context_id (文档ID) ---
        if context_id != -1:
            if verbose:
                print(f"🚨 应用 Context ID (文档ID) 过滤: 只保留与 context_id {context_id} 相关的关系。")
            
            # HACK: 临时加载映射表 (Chunk ID -> Context ID) 以便过滤
            chunks_embed_path = self.data_path / "chunks_with_embeddings.parquet"
            if chunks_embed_path.exists():
                chunks_map_df = pd.read_parquet(chunks_embed_path)[['context_id']].copy()
                chunks_map_df.index.name = 'chunk_id' 
                
                # 过滤出所有属于目标 context_id (文档ID) 的 Chunk 索引
                target_chunk_indices = set(
                    chunks_map_df[chunks_map_df['context_id'] == context_id].index.tolist()
                )

                if not target_chunk_indices:
                    if verbose:
                        print(f"警告: 找不到与文档 ID {context_id} 相关的任何 Chunk 索引。")
                    return pd.DataFrame() # 空结果
                
                # 定义检查函数：判断关系的 chunk_id 字符串中是否包含任何目标索引
                def check_chunk_relevance(chunk_id_str, target_indices):
                    # 将逗号分隔的字符串转换为整数集合
                    # 确保处理字符串中的非数字/空格
                    related_chunks = {int(i.strip()) for i in re.split(r',\s*', str(chunk_id_str)) if i.strip().isdigit()}
                    # 检查是否有交集
                    return bool(related_chunks.intersection(target_indices))

                results_df = results_df[
                    results_df['chunk_id'].astype(str).apply(
                        lambda x: check_chunk_relevance(x, target_chunk_indices)
                    )
                ].copy()
                
                if results_df.empty and verbose:
                    print(f"警告: 应用 Context ID {context_id} 过滤后，图谱检索结果为空。")
            else:
                if verbose:
                    print("警告: 无法加载 chunks_with_embeddings.parquet，跳过 Context ID 过滤。")
                 
        if results_df.empty:
            return results_df
        
        # 2. 关系去重
        # relationship_key: frozenset - 用于去重的关系键
        results_df['relationship_key'] = results_df.apply(
            lambda row: frozenset({row['node_1'].lower(), row['node_2'].lower()}), axis=1
        )
        results_df.drop_duplicates(subset='relationship_key', keep='first', inplace=True)
        
        return results_df
        
    def _calculate_graph_scores(self, query_string: str, results_df: pd.DataFrame, expanded_terms: set, weights: Dict[str, float]) -> pd.DataFrame:
        """
        计算图谱关系的分数（语义、频率、边存在性）并计算最终组合分数。
        """
        import numpy as np
        import pandas as pd
        from typing import Dict
        
        unique_nodes = set(results_df['node_1'].str.lower().unique()) | set(results_df['node_2'].str.lower().unique())
        # 假设 self.embeddings_model 和 self.semantic_df 存在
        query_embedding = self.embeddings_model.embed_query(query_string)
        query_vec = np.array(query_embedding)
        
        node_semantic_scores = {}
        available_nodes = [node for node in unique_nodes if node in self.semantic_df.index and 'embedding' in self.semantic_df.columns]
        
        if available_nodes:
            scoring_df = self.semantic_df.loc[available_nodes].copy()
            scoring_df['embedding'] = scoring_df['embedding'].apply(np.array) # 确保 embedding 是 np.array
            node_vectors_matrix = np.stack(scoring_df['embedding'].to_numpy())
            
            norm_query = np.linalg.norm(query_vec)
            node_norms = np.linalg.norm(node_vectors_matrix, axis=1)
            
            normalized_node_vectors = np.zeros_like(node_vectors_matrix)
            valid_nodes_mask = node_norms > 1e-6
            
            normalized_node_vectors[valid_nodes_mask] = node_vectors_matrix[valid_nodes_mask] / node_norms[valid_nodes_mask, np.newaxis]
            normalized_query_vec = query_vec / norm_query if norm_query > 1e-6 else np.zeros_like(query_vec)
            
            scores_array = normalized_node_vectors @ normalized_query_vec
            
            for idx, node in enumerate(scoring_df.index):
                node_semantic_scores[node] = scores_array[idx]
            
        for node in unique_nodes:
            # 确保所有节点都有分数，如果不存在则为 0.0
            node_semantic_scores[node] = node_semantic_scores.get(node, 0.0)
            
        results_df['score_node_1'] = results_df['node_1'].str.lower().map(node_semantic_scores)
        results_df['score_node_2'] = results_df['node_2'].str.lower().map(node_semantic_scores)
        
        # 两个节点分数的和作为关系语义分数
        results_df['semantic_score'] = results_df['score_node_1'].fillna(0) + results_df['score_node_2'].fillna(0)
        
        # --- 权重和分数归一化 ---
        max_count = max(results_df['count'].max(), 1e-6) 
        # 使用 log1p 进行频率归一化，以避免极端值的影响
        results_df['normalized_count'] = np.log1p(results_df['count']) / np.log1p(max_count)

        max_semantic_score = results_df['semantic_score'].max()
        if max_semantic_score > 0:
            results_df['normalized_semantic_score'] = results_df['semantic_score'] / max_semantic_score
        else:
            results_df['normalized_semantic_score'] = 0

        # --- 优化：高效标记边存在性 (has_edge) ---
        if hasattr(self, 'core_graph_keys') and self.core_graph_keys:
            # 1. 为 results_df 中的每一行计算规范化的关系键 (与 _retrieve_from_graph 中一致)
            results_df['relationship_key'] = results_df.apply(
                lambda row: frozenset({row['node_1'].lower(), row['node_2'].lower()}), axis=1
            )
            # 2. 使用集合的 issubset/intersection 或 .map() 检查存在性 (Set lookup is O(1) average)
            # 如果 relationship_key 存在于 self.core_graph_keys 集合中，则 has_edge=1
            results_df['has_edge'] = results_df['relationship_key'].apply(
                lambda key: 1 if key in self.core_graph_keys else 0
            )
        else:
            # 如果没有预处理核心图谱，回退到 0
            results_df['has_edge'] = 0
            
        # ***核心修改：使用动态权重计算最终分数***
        results_df['combined_score'] = (
            weights.get('Freq', 0.0) * results_df['normalized_count'] + 
            weights.get('Semantic', 0.0) * results_df['normalized_semantic_score'] + 
            weights.get('Edge', 0.0) * results_df['has_edge'] 
        )
        return results_df
    def _format_graph_results_for_llm(self, query_string: str, scored_df: pd.DataFrame, expanded_terms: set, verbose: bool, active_weights, graph_top_k: int) -> list:
        """
        将已评分的图谱结果聚合到 Chunk 级别，并格式化为 LLM 上下文列表。
        """
        # 选取更多以确保 Chunk 覆盖
        scored_df = scored_df.sort_values(by='combined_score', ascending=False).head(graph_top_k) 
        
        evidence_map: Dict[str, Dict[str, Any]] = {} 

        for _, row in scored_df.iterrows():
            node_1, node_2 = str(row['node_1']), str(row['node_2']) 
            query_node = node_1 if node_1.lower() in expanded_terms else node_2
            related_node = node_2 if node_1.lower() in expanded_terms else node_1
            
            # --- 使用 top scoring chunk ID ---
            chunk_ids_str = str(row['chunk_id']).split(',')
            first_chunk_id = chunk_ids_str[0].strip()

            description = ""
            graph_rel = self.graph_df[
                ((self.graph_df['node_1'] == node_1) & (self.graph_df['node_2'] == node_2)) |
                ((self.graph_df['node_1'] == node_2) & (self.graph_df['node_2'] == node_1))
            ]
            if not graph_rel.empty:
                description = graph_rel.iloc[0]['edge']
                
            # 完整分数显示 
            relationship_line = (
                f"{active_weights['Freq']}*{row['normalized_count']:.2f} + {active_weights['Semantic']}*{row['normalized_semantic_score']:.2f} + {active_weights['Edge']}*{row['has_edge']} = "
                f"{row['combined_score']:.2f} ({query_node} ↔️ {related_node})"
            )
            
            if description:
                relationship_line += f" [{description}]"

            if first_chunk_id not in evidence_map:
                try:
                    # 确保 chunks_df 以 chunk_id 为索引
                    evidence_text = self.chunks_df.loc[int(first_chunk_id), 'text'] 
                    summary = self._summarize_evidence(evidence_text)
                except KeyError:
                    summary = f"找不到块ID '{first_chunk_id}' 的源证据。"
                    
                evidence_map[first_chunk_id] = {
                    "evidence": summary,
                    "relationships": [relationship_line]
                }
            else:
                evidence_map[first_chunk_id]["relationships"].append(relationship_line)
                
        all_relationships_for_llm = []
        for chunk_id, data in evidence_map.items():
            relationship_block = "\n".join(data["relationships"])
            
            llm_entry = {
                "query_node": f"Evidence Chunk {chunk_id}",
                "related_node": f"{len(data['relationships'])} aggregated relations",
                "description": relationship_block,
                "evidence": data["evidence"]
            }
            all_relationships_for_llm.append(llm_entry)
            
            if verbose:
                print(f"📦 证据块 ID: {chunk_id} ({len(data['relationships'])} 个关系)")
                print(f"  * 关系/分数: \n{relationship_block}")
                print(f"  * 证据: \"...{data['evidence']}...\"")
                print("-" * 70)
                
        # 最终按块关系数量排序，确保最重要的块排在前面
        all_relationships_for_llm.sort(key=lambda x: int(x['related_node'].split()[0]), reverse=True)
        
        return all_relationships_for_llm
        
    def query(self, query_string: str, context_id: int = -1, graph_top_k: int = 10,document_top_k:int = 5, verbose: bool = False) -> str:
        """
        执行完整的查询流程：优先图谱检索，图谱失败则回退到文档语义检索。
        
        :param context_id: 如果指定 (> -1)，将搜索限制在这个 context_id 对应的 Chunk。
        """
        if verbose and context_id != -1:
            print(f"==================================================")
            print(f"  🛑 Query is restricted to Context ID: {context_id}")
            print(f"==================================================")

        # --- NEW: 查询类型分类与权重确定 ---
        query_type = classify_query_type(query_string, self.llm_model)
        active_weights = self.WEIGHT_MAP.get(query_type, self.WEIGHT_MAP["Default"])
        if verbose:
            print(f"-> 分类结果: {query_type}. 激活权重: Freq={active_weights['Freq']}, Semantic={active_weights['Semantic']}, Edge={active_weights['Edge']}")
        # -----------------------------------
        expanded_terms = self._expand_query(query_string, verbose=verbose) 
        if verbose:
            print(f"🔍 扩展后的查询词: {expanded_terms}")

        all_relationships_for_llm = []
        
        # 1. 图谱检索 (Graph Retrieval) - 传入 context_id
        graph_results_df = self._retrieve_from_graph(expanded_terms, verbose, context_id)

        if not graph_results_df.empty:
            if verbose:
                print(f"\n✅ 图谱检索成功，找到 {len(graph_results_df)} 条去重关系。")
            
            # 2. 分数计算 (Scoring)
            scored_df = self._calculate_graph_scores(query_string, graph_results_df, expanded_terms, active_weights)
            
            # 3. 格式化 (Formatting)
            all_relationships_for_llm = self._format_graph_results_for_llm(
                query_string, scored_df, expanded_terms, verbose, active_weights, graph_top_k
            )
        else:
            if verbose:
                print(f"\n--- 图谱检索为空，触发文档语义回退 ---")
            
            # 2. 回退到纯文档检索 (Fallback to Document Retrieval) - 传入 context_id
            all_relationships_for_llm = self._retrieve_from_documents(
                query_string=query_string,
                expanded_terms=expanded_terms,
                verbose=verbose,
                doucment_top_k=document_top_k, # 回退时也使用 top_k 限制
                context_id=context_id # 传递 context_id 限制搜索范围
            )
            
        if not all_relationships_for_llm:
            if verbose:
                print("\n⚠️ 未能生成任何上下文，直接调用 LLM 失败模式。")
            return self._generate_answer(query_string, [])
            
        # 4. 最终答案生成 (Answer Generation)
        final_answer = self._generate_answer(query_string, all_relationships_for_llm)
        if verbose:
            print("\n" + "="*80)
            print("最终答案 (由LLM生成):",final_answer)
            print("="*80)
        
        return final_answer

def run_single_query_task(engine: GassQueryEngine, question: str, answer: str, context_id: int) -> Dict[str, str]:
    """执行单个查询任务，新增 context_id 参数"""
    print(f"-> 正在处理查询: {question} (Context ID: {context_id})")
    try:
        import time
        # 减小线程间竞争和过快请求的可能性
        time.sleep(random.uniform(0.5, 1.5))  
        # 保持 top_k 为 20，以确保有足够的多样性上下文给 LLM
        # *** 关键修改 3: 传递 context_id ***
        my_answer = engine.query(question, context_id=context_id, graph_top_k=20,document_top_k=5, verbose=True) 
    except Exception as e:
        print(f"处理查询 '{question}' 时发生错误: {e}")
        my_answer = f"Error during query processing: {e}"
        
    print(f"<- 完成查询: {question}")
    
    return {
        "question": question,
        "context_id": context_id, # 新增 context_id 到结果
        "answer": answer,
        "my_answer": my_answer
    }

if __name__ == '__main__':
    BASE_DIR = Path(r"D:\Code\jupyter\knowledge_graph\data_output\dataset\NewsQA\test")
    DATA_DIRECTORY = BASE_DIR
    QA_FILE = BASE_DIR / "qa.csv"
    OUTPUT_FILE = BASE_DIR / "llm_answer_context_restricted.csv" # 修改输出文件名以区分
    
    # 建议在测试时，设置 MAX_WORKERS=1 保持顺序执行，便于调试 verbose 输出
    # 设置为 >= 1 即可
    MAX_WORKERS = 1 

    if not DATA_DIRECTORY.exists():
        print(f"Error: Knowledge Graph Data Directory '{DATA_DIRECTORY}' not found.")
    elif not QA_FILE.exists():
        print(f"Error: QA Input File '{QA_FILE}' not found.")
    else:
        try:
            engine = GassQueryEngine(data_directory=str(DATA_DIRECTORY))
            
            qa_df = pd.read_csv(QA_FILE, sep="|")
            # 确保 context_id 是整数，默认为 -1 (表示不限制)
            if 'context_id' not in qa_df.columns:
                qa_df['context_id'] = -1
            qa_df['context_id'] = qa_df['context_id'].fillna(-1).astype(int)
            
            print(f"\n成功加载 {len(qa_df)} 个查询任务。")

            all_results = []
            
            print(f"--- Starting Multi-threaded Query Execution with {MAX_WORKERS} workers ---")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_query = {
                    executor.submit(
                        run_single_query_task, 
                        engine, 
                        row['question'], 
                        row['answer'], 
                        row['context_id'] # *** 关键修改 4: 传递 context_id ***
                    ): row['question'] 
                    for index, row in qa_df.iterrows()
                }
                
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as exc:
                        print(f"查询 '{query}' 产生了一个异常: {exc}")

            print("\n--- All Queries Completed ---")
            
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(OUTPUT_FILE, index=False, sep="|", encoding='utf-8')
            print(f"所有 {len(results_df)} 条结果已保存到: {OUTPUT_FILE}")

        except Exception as e:
            print(f"致命错误: {e}")