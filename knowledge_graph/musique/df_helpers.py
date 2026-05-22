import uuid
import ast
import pandas as pd
import numpy as np
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any, Dict, List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from helper import parallel_llm_processor
from zhipu import extractConcepts, graphPrompt, resolve_coreferences
from utils import get_llm_model,get_chat_model


# ======================================================================
# 0. 通用辅助函数 (新增)
# ======================================================================

def parse_list(x):
    """解析字符串列表，如果失败返回空列表"""
    if isinstance(x, list): return x
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            return val if isinstance(val, list) else []
        except:
            return []
    return []

def build_enriched_name_text(row):
    """
    构建格式: "Standard_Entity, synonyms: Synonym A, Synonym B"
    用于增强向量检索的准确性
    """
    entity_name = str(row['Standard_Entity']).strip()
    synonyms_list = row['synonyms'] if isinstance(row['synonyms'], list) else []
    
    # 过滤掉与标准名完全相同的同义词，避免冗余
    valid_syns = [s for s in synonyms_list if s and str(s).strip() != entity_name]
    
    if valid_syns:
        # 截断过长的同义词列表 (前10个)
        syn_str = ", ".join(valid_syns[:10]) 
        return f"{entity_name}, synonyms: {syn_str}"
    
    return entity_name

# ======================================================================
# 1. 基础数据处理
# ======================================================================

def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        clean_text = chunk.page_content.replace('\n', ' ').replace('\r', ' ')
        row = {
            "text": clean_text,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ======================================================================
# 2. 实体聚合逻辑 (新增 - 来自 merge.py)
# ======================================================================

def merge_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """
    聚合相同 (context_id, Standard_Entity) 的实体，合并描述和同义词。
    """
    print("🔄 [Helper] 正在聚合实体数据...")
    df = df.copy()
    
    # 1. 解析列表列
    for col in ['synonyms']:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)
        else:
            df[col] = [[] for _ in range(len(df))] # 填充空列表
    
    if 'description' not in df.columns:
        df['description'] = ""
    df['description'] = df['description'].fillna("")
    
    # 2. 将 'Entity' (原始提及) 加入到同义词候选中
    df['synonyms'] = df.apply(lambda row: row['synonyms'] + [str(row['Entity'])], axis=1)

    # 3. 定义聚合函数
    def merge_descriptions(series):
        unique_descs = set()
        for d in series:
            d_str = str(d).strip()
            if d_str and d_str != '-1':
                unique_descs.add(d_str)
        return ". ".join(sorted(list(unique_descs)))

    def merge_lists(series):
        merged = set()
        for lst in series:
            merged.update(lst)
        return list(merged)

    # 4. 执行 GroupBy
    # 确保 Standard_Entity 不为空
    df = df[df['Standard_Entity'].notna() & (df['Standard_Entity'] != "")]
    
    df_grouped = df.groupby(['context_id', 'Standard_Entity', 'category'], as_index=False).agg({
        'chunk_id': 'first', # 取第一个出现的 chunk_id 作为代表
        'cluster_id': 'first',
        'category': 'first',
        'description': merge_descriptions,
        'synonyms': merge_lists,
    })
    # 【修改点】确保聚合后的数据依然按照 chunk_id 排序
    df_grouped = df_grouped.sort_values(by=['chunk_id']).reset_index(drop=True)
    
    print(f"✅ [Helper] 聚合完成。原始行数: {len(df)} -> 聚合后行数: {len(df_grouped)}")
    return df_grouped

# ======================================================================
# 3. LLM 任务封装
# ======================================================================
def ExtractConcepts(dataframe: pd.DataFrame, model=None, max_workers: int = 100) -> pd.DataFrame:
    """
    抽取概念节点。
    """
    def concept_extractor(i, row):
        text_input = row.get('resolved_text', row['text'])
        # 此处若发生 429 或 JSON 错误，会由 parallel_llm_processor 捕获并重试
        current_model = get_chat_model(task_type="extraction")
        # current_model = get_llm_model(model='glm-4-flash-250414')
        concepts_list = extractConcepts(text_input, model=current_model)
        return (row['chunk_id'], concepts_list)

    start_msg = f"--- 步骤: 开始并行抽取 {len(dataframe)} 个 Chunk 的概念节点 ---"
    successful_results = parallel_llm_processor(
        dataframe=dataframe,
        processing_func=concept_extractor,
        start_message=start_msg,
        max_workers=max_workers
    )

    final_records = []
    id_map = dataframe.set_index('chunk_id')['context_id'].to_dict()

    for chunk_id, concepts_list in successful_results:
        context_id = id_map.get(chunk_id)
        if not concepts_list: continue
        for concept in concepts_list:
            record = {
                'context_id': context_id,
                'chunk_id': chunk_id,
                'Entity': concept.get('entity'),
                'category': concept.get('category'),
                'description': concept.get('description'),
                'synonyms': concept.get('synonyms'),
            }
            final_records.append(record)

    df_concepts = pd.DataFrame(final_records)
    if not df_concepts.empty:
        df_concepts = df_concepts.sort_values(by=['chunk_id']).reset_index(drop=True)
        
    print(f"✅ 概念抽取完成，共提取 {len(df_concepts)} 个实体。")
    return df_concepts


def ResolveCoreferences(dataframe: pd.DataFrame, model=None, max_workers=15) -> pd.DataFrame:
    """
    指代消解。
    """
    def coref_resolver(i, row):
        # 此处 resolve_coreferences 已不再自行吞掉 429 错误
        current_model = get_chat_model(task_type="reasoning")
        # current_model = get_llm_model(model='glm-4-flash-250414')
        resolved_text = resolve_coreferences(row.text, model=current_model) 
        return (row['chunk_id'], resolved_text)

    start_msg = f"--- 步骤: 开始并行指代消解 {len(dataframe)} 个 Chunk ---"
    successful_results = parallel_llm_processor(
        dataframe=dataframe,
        processing_func=coref_resolver,
        start_message=start_msg,
        max_workers=max_workers
    )

    results_dict = {res[0]: res[1] for res in successful_results}
    dataframe['resolved_text'] = dataframe['chunk_id'].map(results_dict)
    # 最终如果彻底失败（重试3次后），回退到原始文本
    dataframe['resolved_text'] = dataframe['resolved_text'].fillna(dataframe['text'])
    print(f"✅ 指代消解完成。")
    return dataframe

# ======================================================================
# 4. 图谱构建与共现
# ======================================================================

def df2Graph(dataframe: pd.DataFrame, entity_map: dict, model=None, max_workers=100) -> list:
    """
    并行关系抽取。
    """
    def graph_extractor(i, row):
        current_map = entity_map.get((row['context_id'], row['chunk_id']), {})
        if not current_map: return (row['chunk_id'], [])
        metadata = {"context_id": row['context_id'], "chunk_id": row['chunk_id']}
        current_model = get_chat_model(task_type="extraction")
        # current_model = get_llm_model(model='glm-4-flash-250414')
        # graphPrompt 抛出的异常会被重试机制捕获
        result = graphPrompt(row.resolved_text, current_map, metadata, current_model)
        return (row['chunk_id'], result)

    start_msg = f"--- 步骤: 开始并行抽取图谱关系 ---"
    successful_results = parallel_llm_processor(
        dataframe=dataframe,
        processing_func=graph_extractor,
        start_message=start_msg,
        max_workers=max_workers
    )

    concept_list = []
    for _, relations_list in successful_results:
        if relations_list: concept_list.extend(relations_list)
    print(f"✅ 关系抽取完成，共提取 {len(concept_list)} 条边。")
    return concept_list

def graph2Df(nodes_list) -> pd.DataFrame:
    if not nodes_list:
        return pd.DataFrame(columns=["context_id", "node_1", "node_2", "edge", "chunk_id"])
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    required_cols = ["context_id", "node_1", "node_2", "edge", "chunk_id"]
    for col in required_cols:
        if col not in graph_dataframe.columns: graph_dataframe[col] = np.nan
    graph_dataframe = graph_dataframe.filter(items=required_cols, axis=1)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    # 【修改点】确保关系表按 chunk_id 排序
    if 'chunk_id' in graph_dataframe.columns:
        graph_dataframe['chunk_id'] = pd.to_numeric(graph_dataframe['chunk_id'], errors='coerce')
        graph_dataframe = graph_dataframe.sort_values(by=['chunk_id']).reset_index(drop=True)
    return graph_dataframe


def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["context_id", "node_1", "node_2", "chunk_id", "count", "edge"])

    def unique_chunks_str(x):
        return sorted(list(set(str(i) for i in x)))

    dfg_long = pd.melt(df, id_vars=["chunk_id", "context_id"], value_vars=["node_1", "node_2"], value_name="node")
    dfg_long.drop(columns=["variable"], inplace=True)
    dfg_long["chunk_id"] = dfg_long["chunk_id"].astype(str)

    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    dfg_wide = dfg_wide[dfg_wide["node_1"] != dfg_wide["node_2"]]

    dfg2 = (
        # ✅ 把 context_id_1 加入主键，切断跨文章的融合
        dfg_wide.groupby(["context_id_1", "node_1", "node_2"])
        .agg({
            "chunk_id": [lambda x: ",".join(unique_chunks_str(x)), "count"]
        })
        .reset_index()
    )
    
    # ✅ 修复点：严格按照 reset_index() 生成的列顺序进行重命名
    dfg2.columns = ["context_id", "node_1", "node_2", "chunk_id", "count"]
    
    dfg2["context_id"] = dfg2["context_id"].astype(str)
    
    # 现在 "count" 列确实是数字了，比较不会再报错
    dfg2 = dfg2[dfg2["count"] > 2]
    dfg2["edge"] = "contextual_proximity"
    
    return dfg2[["context_id", "node_1", "node_2", "chunk_id", "count", "edge"]]