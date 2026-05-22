import pandas as pd
import numpy as np
import os
import sys
import ast
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi  # [新增] 引入 BM25

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import get_embeddings_model

class SemanticMatcher:
    def __init__(self, parquet_path):
        """
        初始化匹配器：加载 Parquet 并构建内存矩阵
        """
        self.parquet_path = Path(parquet_path)
        print(f"🔄 正在加载锚点数据库: {self.parquet_path.name} ...")
        
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"找不到文件: {self.parquet_path}")
            
        # 1. 读取数据
        self.df = pd.read_parquet(self.parquet_path)
        
        # 2. 预处理：解析 synonyms 列
        print("⚙️ 正在解析同义词...")
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

        # 3. 预处理：构建 Numpy 矩阵
        print("⚙️ 正在构建向量矩阵...")
        
        self.matrix_entity = np.stack(self.df['vec_entity'].values)
        
        # 防御性处理空描述向量
        if self.df['vec_desc'].isnull().any():
             print("⚠️ 警告: 发现空的描述向量，将用零向量填充。")
             dim = self.matrix_entity.shape[1]
             self.df['vec_desc'] = self.df['vec_desc'].apply(
                 lambda x: x if x is not None else np.zeros(dim)
             )
        self.matrix_desc = np.stack(self.df['vec_desc'].values)
        
        # 4. 初始化 Embedding 模型
        self.embed_model = get_embeddings_model(dimensions=1024)
        print(f"✅ 初始化完成。内存中包含 {len(self.df)} 个聚合实体。")

    def get_query_vector(self, query):
        """调用 API 获取 Query 向量"""
        vec = self.embed_model.embed_query(query)
        return np.array(vec).reshape(1, -1)

    def _extract_keywords_for_vector(self, query):
        """
        [关键优化] 针对描述向量检索的 Query 净化
        """
        # 1. 优先提取引号内容 (强意图)
        quotes = re.findall(r'"([^"]*)"', query)
        if quotes: return " ".join(quotes)

        # 2. 简单的词性过滤 (仅保留名词和大写词)
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
        [重命名] 计算绝对精确的字面包含分数 (Exact Substring Match)
        依然保留，因为对于简单的实体（比如年份或短名字），精确匹配的置信度最高。
        """
        query_lower = query.lower()
        scores = []
        
        for idx, row in candidate_df.iterrows():
            match_score = 0.0
            
            # 检查标准名
            std_name = str(row['Standard_Entity']).lower().strip()
            if len(std_name) >= 2 and std_name in query_lower:
                match_score = 1.0
            else:
                # 检查同义词
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
        [新增核心] 动态构建当前上下文的 BM25 索引，计算稀疏相似度
        将实体的 名字 + 同义词 + 描述 组合成“文档”，利用 TF-IDF 逻辑锁定低频专有名词。
        """
        tokenized_corpus = []
        
        for idx, row in candidate_df.iterrows():
            # 拼装实体所有可能出现的名字和描述
            text_parts = [str(row.get('Standard_Entity', ''))]
            text_parts.extend([str(s) for s in row.get('synonyms_list', [])])
            text_parts.append(str(row.get('description', '')))
            
            full_text = " ".join(text_parts).lower()
            # 简单分词 (保留字母和数字)
            tokens = re.findall(r'\w+', full_text)
            tokenized_corpus.append(tokens)
            
        if not tokenized_corpus:
            return np.zeros(len(candidate_df))
            
        # 实例化 BM25
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = re.findall(r'\w+', query.lower())
        
        # 计算得分
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Min-Max 归一化到 0~1 之间
        max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        return np.array([s / max_score for s in bm25_scores])

    def link(self, query, context_id, top_k=10, lambda_weights=(0.25, 0.40, 0.15, 0.20)):
        """
        核心链接函数 (BM25 + Dense 终极形态)
        :param lambda_weights: (w_name, w_desc, w_exact, w_bm25) 四路召回权重
        """
        w_name, w_desc, w_exact, w_bm25 = lambda_weights
        
        # === 修改点：智能识别全局/局部模式 ===
        if context_id is None or str(context_id).lower() == 'global':
            # 🌍 全局模式：所有人放行，mask 全为 True
            mask = np.ones(len(self.df), dtype=bool)
        else:
            # 🏠 局部模式：严格检查 context_id
            mask = self.df['context_id'].astype(str) == str(context_id)
            
        if not mask.any(): 
            return pd.DataFrame()
        # ==================================)

        # --- 1. 双路 Query 向量化 (Dense) ---
        v_q_name = self.get_query_vector(query)
        clean_query = self._extract_keywords_for_vector(query)
        v_q_desc = self.get_query_vector(clean_query)

        # --- 2. 矩阵切片 ---
        sub_matrix_entity = self.matrix_entity[mask]
        sub_matrix_desc = self.matrix_desc[mask]
        candidate_rows = self.df[mask].copy()

        # --- 3. 向量计算 (Dense Scores) ---
        s_name = cosine_similarity(v_q_name, sub_matrix_entity)[0]
        s_desc = cosine_similarity(v_q_desc, sub_matrix_desc)[0]
        
        # --- 4. 字面匹配与 BM25 (Sparse/Lexical Scores) ---
        s_exact = self._calculate_exact_match_score(candidate_rows, query)
        s_bm25 = self._calculate_bm25_score(candidate_rows, query)  # [新增]

        # --- 5. 综合打分 (4D Fusion) ---
        final_scores = (w_name * s_name) + (w_desc * s_desc) + (w_exact * s_exact) + (w_bm25 * s_bm25)
        
        candidate_rows['Score'] = final_scores
        candidate_rows['S_name'] = s_name
        candidate_rows['S_desc'] = s_desc
        candidate_rows['S_exact'] = s_exact
        candidate_rows['S_bm25'] = s_bm25  # 保存用于 debug
        
        # 结果返回
        cols = ['Standard_Entity', 'context_id', 'category', 'Score', 'S_name', 'S_desc', 'S_exact', 'S_bm25', 'synonyms']
        return candidate_rows.sort_values(by='Score', ascending=False).head(top_k)[cols]
            
# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # --- 配置路径 ---
    # 请根据你的实际路径修改这里
    BASE_DIR = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\hotpot\ds1000_2"
    PARQUET_FILE = os.path.join(BASE_DIR, "concepts_merged_with_vectors.parquet")
    QA_FILE = os.path.join(BASE_DIR, "qa.csv")
    
    # 1. 实例化 (加载一次，常驻内存)
    try:
        linker = SemanticMatcher(PARQUET_FILE)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)
    
    # 2. 读取 QA 文件进行测试
    if os.path.exists(QA_FILE):
        df_qa = pd.read_csv(QA_FILE, sep="|")
        
        print("\n🚀 开始测试实体链接 (Top 5 Questions)...")
        print("="*80)
        
        # 混合权重：(Vector_Name, Vector_Desc, Lexical_Match)
        # 这里的 0.3 Lexical 权重能让精确匹配的实体得分显著增加
        HYBRID_WEIGHTS = (0.25, 0.40, 0.15, 0.20) 

        # 测试前 5 个问题
        for i, row in df_qa.head(10).iterrows():
            q = row['question']
            ctx = row['context_id']
            
            print(f"\n❓ [Q{i+1}] Context: {ctx}")
            print(f"   Query: {q}")
            
            # 调用链接
            df_res = linker.link(q,context_id=None, top_k=15, lambda_weights=HYBRID_WEIGHTS)
            
            if not df_res.empty:
                print(f"✅ 找到 {len(df_res)} 个候选实体 (按 Score 降序):")
                # 打印所有分数细节
                # 🌍 更新打印列名：加入 S_exact 和 S_bm25
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
                print("❌ 未找到匹配实体")
            print("-" * 80)
    else:
        print(f"❌ 找不到 QA 文件: {QA_FILE}")