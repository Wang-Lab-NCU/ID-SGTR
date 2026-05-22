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
# 导入辅助函数
from helper import apply_genealogical_penalty,post_process_person_entities
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
            print(f"📥 下载 Tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(local_path)
            return tokenizer

    # =================================================================
    # 1. 核心 Embedding 逻辑 (私有方法)
    # =================================================================
    def _batch_embed(self, texts: list, batch_size=64, desc="Embedding"):
        """内部批量嵌入方法"""
        model = get_embeddings_model(dimensions=self.embedding_dim)
        # 清理空文本
        texts = [str(t).strip() if str(t).strip() else " " for t in texts]
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[i : i + batch_size]
            try:
                batch_emb = model.embed_documents(batch)
                all_embeddings.extend(batch_emb)
            except Exception as e:
                print(f"❌ Batch Embedding Failed: {e}")
                # 填充 None 或 零向量，这里选择填充 None 并在后续处理
                all_embeddings.extend([None] * len(batch))
        return [np.array(e) if e else np.zeros(self.embedding_dim) for e in all_embeddings]

    def compute_embeddings(self, df: pd.DataFrame, text_col: str, id_col: str, file_name: str):
        """通用嵌入计算并保存单个文件"""
        cache_file = self.output_dir / file_name
        emb_col_name = 'entity_embedding' if 'entity' in file_name.lower() else 'embedding'

        if cache_file.exists():
            print(f"🔍 [Pipeline] 加载嵌入缓存: {cache_file}")
            try:
                df_cached = pd.read_parquet(cache_file)
                if emb_col_name in df_cached.columns:
                     df_cached[emb_col_name] = df_cached[emb_col_name].apply(lambda x: np.array(x) if isinstance(x, list) else x)
                return df_cached
            except Exception:
                pass

        print(f"⏳ [Pipeline] 开始计算 {len(df)} 条数据的嵌入向量...")
        embeddings = self._batch_embed(df[text_col].tolist(), desc="Computing Embeddings")
        df[emb_col_name] = pd.Series(embeddings, index=df.index)
        
        # 保存前排序
        if 'chunk_id' in df.columns:
            df = df.sort_values(by=['chunk_id'])
            
        # Save
        df_to_save = df.copy()
        df_to_save[emb_col_name] = df_to_save[emb_col_name].apply(lambda x: x.tolist())
        df_to_save.to_parquet(cache_file, index=False)
        print(f"✅ [Pipeline] 保存至 {cache_file}")
        return df

    # =================================================================
    # 2. 实体聚合与双重嵌入 (Merge Logic)
    # =================================================================
    def merge_and_embed_concepts(self, df_concepts: pd.DataFrame = None):
        """
        聚合实体 -> 生成 merge_entity.csv
        计算向量 -> 生成 concepts_merged_with_vectors.parquet (包含 vec_entity 和 vec_desc)
        """
        cache_file = self.output_dir / "concepts_merged_with_vectors.parquet"
        merge_csv_file = self.output_dir / "merge_entity.csv"
        
        if cache_file.exists():
            print(f"🔍 [Pipeline] 加载聚合实体缓存: {cache_file}")
            df_merged = pd.read_parquet(cache_file)
            # 恢复向量格式
            for col in ['vec_entity', 'vec_desc']:
                if col in df_merged.columns:
                    df_merged[col] = df_merged[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)
            return df_merged

        if df_concepts is None:
             # 尝试从之前步骤的文件加载
             prev_file = self.output_dir / "dp_extracted_concepts.csv"
             if not prev_file.exists():
                 raise FileNotFoundError("Missing input for merge step.")
             df_concepts = pd.read_csv(prev_file, sep="|")

        # 1. 聚合
        print("🚀 [Pipeline] 开始实体聚合...")
        df_merged = merge_concepts(df_concepts)
        # 保存 merge_entity.csv 前排序
        df_merged = df_merged.sort_values(by=['chunk_id']).reset_index(drop=True)
        df_merged.to_csv(merge_csv_file, sep="|", index=False)
        
        # 2. 文本增强
        print("🛠️ [Pipeline] 构建增强实体文本...")
        enriched_texts = df_merged.apply(build_enriched_name_text, axis=1).tolist()
        
        # 3. 计算向量 (双重: Name+Synonyms 和 Description)
        print("🚀 [Pipeline] 计算 Enriched Entity Vectors...")
        df_merged['vec_entity'] = self._batch_embed(enriched_texts, desc="Vec: Entity")
        
        print("🚀 [Pipeline] 计算 Description Vectors...")
        desc_texts = df_merged['description'].fillna("").astype(str).tolist()
        df_merged['vec_desc'] = self._batch_embed(desc_texts, desc="Vec: Desc")
        
        # 4. 保存
        print(f"💾 [Pipeline] 保存聚合向量表到: {cache_file}")
        df_to_save = df_merged.copy()
        # 再次确保最终输出有序
        df_to_save = df_to_save.sort_values(by=['chunk_id']).reset_index(drop=True)
        df_to_save['vec_entity'] = df_to_save['vec_entity'].apply(lambda x: x.tolist())
        df_to_save['vec_desc'] = df_to_save['vec_desc'].apply(lambda x: x.tolist())
        df_to_save.to_parquet(cache_file, index=False)
        
        return df_merged

    # =================================================================
    # 3. 数据加载与分块 (纯净极速版 - 专供 MuSiQue)
    # =================================================================
    def load_and_split_data(self, json_path, max_contexts=None, chunk_size=300, chunk_overlap=50):
        cache_file = self.output_dir / "chunk.csv"
        print(f"🚀 [Pipeline] 开始加载 MuSiQue 数据: {json_path}")
        
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        global_chunk_id = 0
        limit = max_contexts if max_contexts else len(data)

        # 遍历每一个问答对
        for i, item in enumerate(data[:limit]):
            for para in item.get("paragraphs", []):
                title = para.get("title", "")
                paragraph_text = para.get("paragraph_text", "")
                
                # 构造 Chunk：标题 + 正文
                chunks.append({
                    "context_id": i,
                    "chunk_id": global_chunk_id,
                    "title": title,
                    "text": f"{title} information: {paragraph_text}"
                })
                global_chunk_id += 1
                    
        df_chunks = pd.DataFrame(chunks)
        df_chunks.to_csv(cache_file, sep="|", index=False)
        print(f"✅ Loaded {len(df_chunks)} chunks from MuSiQue.")
        return df_chunks

    def standardize_entities(self, df_concepts_flat: pd.DataFrame):
        """
        核心函数：
        1. 解析并聚合实体同义词 (Entity + Synonyms) -> Rich Input Text
        2. 计算实体嵌入
        3. 执行 DBSCAN 聚类
        4. 生成标准名称映射
        """
        print("🚀 [Pipeline] 开始实体标准化流程...")
        
        # --- 3.1 数据预处理与解析 ---
        df_concepts_flat = df_concepts_flat.copy()
        df_concepts_flat['Entity'] = df_concepts_flat['Entity'].astype(str).str.strip()
        df_concepts_flat['category'] = df_concepts_flat['category'].astype(str).str.strip()

        # 【新增】辅助解析函数：处理 CSV 读取后的字符串列表 "['a', 'b']"
        def parse_synonyms_col(x):
            if isinstance(x, list): return x
            if isinstance(x, str):
                try: 
                    val = ast.literal_eval(x)
                    return val if isinstance(val, list) else []
                except: 
                    return []
            return []

        # 【新增】解析 synonyms 列
        df_concepts_flat['synonyms'] = df_concepts_flat['synonyms'].apply(parse_synonyms_col)
        
        # 【关键修改】将 'Entity' (原始提及) 加入到 synonyms 列表中
        # 这一步对应您提到的 merge 逻辑：确保 Rich_Input_Text 包含实体本身
        # 新代码：相加后立刻 set() 去重，再转回 list 并排序
        df_concepts_flat['synonyms'] = df_concepts_flat.apply(
            lambda row: sorted(list(set(row['synonyms'] + [str(row['Entity'])]))), axis=1
        )
        print("前5条展开：",df_concepts_flat.head(5))
        # --- 3.2 聚合生成 Rich Input Text ---
        # 【修改】groupby 聚合逻辑：处理列表的扁平化和去重 (List of Lists -> Set -> String)
        grouped = df_concepts_flat.groupby(['context_id', 'Entity', 'category'])['synonyms'].apply(
            lambda x: ' | '.join(sorted(list(set([item for sublist in x for item in sublist if item]))))
        ).reset_index()
        
        grouped.rename(columns={'synonyms': 'Aggregated_Synonyms'}, inplace=True)
        
        # 构建 Rich Input Text
        grouped['Rich_Input_Text'] = grouped.apply(
            lambda row: f"Entity: {row['Entity']}. synonyms: {row['Aggregated_Synonyms']}", axis=1
        )
        
        print("分组后前5条：",grouped.head(5))
        # --- 3.3 计算实体嵌入 (复用通用函数) ---
        df_embedded = self.compute_embeddings(
            df=grouped, 
            text_col='Rich_Input_Text', 
            id_col='Entity', 
            file_name="entity_embeddings.parquet"
        )

        # --- 3.4 DBSCAN 聚类 ---
        df_map = self._perform_clustering(df_embedded)
        
        # --- 3.5 应用映射 ---
        return self._apply_mapping(df_concepts_flat, df_map)

    def _perform_clustering(self, df_linked: pd.DataFrame):
        EPS_CONFIG = {
            # --- 原有类别 ---
            'Person': 0.05,       # 名字变体多 (e.g., J. Biden vs Joe Biden)
            'Organization': 0.08, # 全称与简称 (e.g., NASA vs National Aeronautics...)
            'Location': 0.05,     # 地名相对固定，但可能有 "City of X" vs "X"
            'Structure': 0.05,    # 建筑名称相对固定
            'Natural': 0.08,      # 物种/化学品可能有别名
            'Work': 0.05,         # 书名/电影名通常较为特定
            'Event': 0.05,        # 事件名称
            'Time': 0.01,         # 时间应高度标准化，严格匹配
            'Quantity': 0.01,     # 数值应严格匹配
            'Product': 0.05,      # 产品/交通工具 (e.g., Boeing 747 vs 747)
            'Award': 0.05,        # 奖项 (e.g., Oscar vs Academy Award)
            'Role': 0.08,         # 职位/角色 (e.g., CEO vs Chief Executive Officer) - 语义相近
            'Concept': 0.08,      # 概念/学科 (e.g., AI vs Artificial Intelligence) - 语义跨度较大
            'Group': 0.08         # 群体/国籍 (e.g., American vs Americans)
        }
        
        DEFAULT_EPS = 0.10
        final_entity_map = []
        context_cluster_max_id = {}
        
        print(f"⏳ [Pipeline] 执行智能分组聚类 (HAC + Genealogy Penalty)...")        # 按 Context 和 Category 分组，互不干扰
        for (context_id, category), group in tqdm(df_linked.groupby(['context_id', 'category']), desc="Clustering"):
            current_max = context_cluster_max_id.get(context_id, -1)
            eps = EPS_CONFIG.get(category, DEFAULT_EPS)
            
            # 1. 提取唯一实体 (去重)
            unique_group = group[['Entity', 'entity_embedding']].drop_duplicates(subset=['Entity']).reset_index(drop=True)
            entities = unique_group['Entity'].tolist()
            
            # 情况 A: 只有一个实体，无需聚类，自成一派
            if len(unique_group) < 2:
                unique_group['cluster_id'] = current_max + 1
                current_max += 1 
            
            # 情况 B: 多个实体，执行高级聚类
            else:
                # 堆叠向量
                matrix = np.vstack(unique_group['entity_embedding'].values)
                
                # --- 步骤 1: 计算基础余弦距离矩阵 ---
                # cosine_distances = 1 - cosine_similarity (范围 0~2, 越小越近)
                dist_matrix = cosine_distances(matrix)
                
                # --- 步骤 2: (仅人物) 应用谱系惩罚 ---
                if category == 'Person':
                    dist_matrix = apply_genealogical_penalty(entities, dist_matrix)

                # --- 步骤 3: 层次聚类 (HAC) ---
                # metric='precomputed': 告诉算法我们传入的是距离矩阵，而不是原始向量
                # linkage='single': 单链策略，允许 A->B->C 链式聚合 (解决长短名问题的关键)
                # distance_threshold=eps: 距离小于此值则合并
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    metric='precomputed', 
                    linkage='single', 
                    distance_threshold=eps
                )
                
                # 拟合
                clusters = clustering.fit_predict(dist_matrix)
                
                # 分配 Cluster ID (加上当前 context 的偏移量，防止冲突)
                unique_group['cluster_id'] = clusters + (current_max + 1)
                
                if len(clusters) > 0:
                    current_max = unique_group['cluster_id'].max()

            # --- 标准化名称生成 ---
            # 策略：选择簇内最长的名字作为 Standard Entity (通常全名包含信息最多)
            def get_standard(sub_df):
                valid = sub_df[sub_df['Entity'].str.len() > 0]
                if valid.empty: return sub_df['Entity'].iloc[0] if not sub_df.empty else ""
                return valid.loc[valid['Entity'].str.len().idxmax(), 'Entity']
            
            cluster_standards = unique_group.groupby('cluster_id').apply(get_standard, include_groups=False).to_dict()
            unique_group['Standard_Entity'] = unique_group['cluster_id'].map(cluster_standards)
            
            # 更新 Context 状态
            context_cluster_max_id[context_id] = current_max
            unique_group['context_id'] = context_id
            unique_group['category'] = category
            final_entity_map.append(unique_group)
            
        # 合并所有分组结果
        df_map = pd.concat(final_entity_map, ignore_index=True)
        df_map.rename(columns={'Entity': 'Original_Entity'}, inplace=True)
        
        # 调用已有的后处理函数 (如有)
        df_map = post_process_person_entities(df_map)
        
        return df_map

    def _apply_mapping(self, df_concepts, df_map):
        # 【关键修改】Set Index 时必须包含 'category'，防止同名不同类的实体冲突
        # df_map 中的列名应该是 'Original_Entity', 对应 df_concepts 中的 'Entity'
        
        # 1. 构建包含 category 的复合键字典
        std_lookup = df_map.set_index(['context_id', 'Original_Entity', 'category'])['Standard_Entity'].to_dict()
        cls_lookup = df_map.set_index(['context_id', 'Original_Entity', 'category'])['cluster_id'].to_dict()
        
        def lookup(row, lookup_dict, default_col=None):
            # 【关键修改】查找时也必须带上 category
            key = (row['context_id'], row['Entity'], row['category'])
            val = lookup_dict.get(key)
            if val is not None: return val
            
            # 备用方案：如果完全匹配失败（极少见），尝试回退到仅匹配 Entity（可选，视情况而定）
            # 但为了严谨，这里建议直接返回默认值
            return row[default_col] if default_col else -1

        print("🚀 [Pipeline] 应用实体映射 (Key: Context + Entity + Category)...")
        df_concepts['Standard_Entity'] = df_concepts.apply(lambda r: lookup(r, std_lookup, 'Entity'), axis=1)
        df_concepts['cluster_id'] = df_concepts.apply(lambda r: lookup(r, cls_lookup), axis=1)
        
        # 确保输出有序
        df_concepts = df_concepts.sort_values(by=['chunk_id']).reset_index(drop=True)
        
        save_path = self.output_dir / "dp_extracted_concepts.csv"
        df_concepts.to_csv(save_path, sep="|", index=False)
        print(f"✅ [Pipeline] 标准化完成。")
        return df_concepts
        
    def generate_entity_map_for_graph(self, df_concepts):
        valid = df_concepts[df_concepts['Standard_Entity'].notna() & (df_concepts['Standard_Entity'] != "")]
        return valid.groupby(['context_id', 'chunk_id']).apply(
            lambda x: dict(zip(x['Entity'], x['Standard_Entity']))
        ).to_dict()
    
    def extract_qa_pairs(self, json_path, max_contexts=None):
        print(f"🚀 [Pipeline] 提取 MuSiQue QA 对...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        limit = max_contexts if max_contexts else len(data)
        relations = []
        
        for i, item in enumerate(data[:limit]):
            question_text = item.get("question", "").strip()
            
            # 获取 MuSiQue 的终极答案：优先从分解步骤的最后一步拿
            if "question_decomposition" in item and len(item["question_decomposition"]) > 0:
                answer_text = str(item["question_decomposition"][-1].get("answer", ""))
            else:
                answer_text = str(item.get("answer", "")) # 兜底

            relations.append({
                "question": question_text,
                "answer": answer_text,
                "context_id": i
            })
            
        df_qa = pd.DataFrame(relations)
        df_qa.to_csv(self.output_dir / "qa.csv", sep="|", index=False)
        print(f"✅ 成功提取 {len(df_qa)} 对 MuSiQue QA 数据。")
        return df_qa