# D:\Code\jupyter\knowledge_graph\adapt\adapt.py
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
# 重新导入特征提取器
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import extract_structural_features 

# --- 1. 配置常量 ---
API_EMBED_DIM = 1024
STRUCT_FEAT_DIM = 4      # 新增：4维结构特征
INPUT_DIM = API_EMBED_DIM + STRUCT_FEAT_DIM # 总维度 1028

NUM_INTENTS = 3
HIDDEN_DIM = 64
CONFIDENCE_THRESHOLD = 0.60 # 建议设为 0.55 - 0.60

MODEL_WEIGHTS_PATH = 'adapt/intent_classifier_struct.pth' # 改个新名字

# --- 2. 意图类别与权重映射 ---
INTENT_CLASSES: List[str] = ["Retrieval", "Reasoning", "Comparative"]

# 权重矩阵 [w_f, w_s, w_e]
#w_f隐式共现权重:
#w_s 纯语义相似度权重:的 Query向量 与 邻居节点向量
#w_e 显式结构权重
# hotpot & musique
WEIGHT_MATRIX: Dict[int, List[float]] = {
    0: [0.10, 0.40, 0.50], # Retrieval
    1: [0.15, 0.30, 0.55], # Reasoning
    2: [0.10, 0.25, 0.65], # Comparative
    3: [0.15, 0.40, 0.45]  # Default (兜底)
}

# 针对 2Wiki 的专用权重：
# WEIGHT_MATRIX: Dict[int, List[float]] = {
#     0: [0.05, 0.25, 0.70], # Retrieval: 既然是找属性，直接信图谱
#     1: [0.05, 0.35, 0.60], # Reasoning: 推理链条主要靠显式边
#     2: [0.00, 0.30, 0.70], # Comparative: 比较数值完全靠结构
#     3: [0.05, 0.30, 0.65]  # Default
# }

# WEIGHT_MATRIX: Dict[int, List[float]] = {
#     0: [0.10, 0.30, 0.60], # Retrieval: 显式图谱主导，允许小幅度的语义游走防漏抽
#     1: [0.15, 0.35, 0.50], # Reasoning: [混合创新点] 保障 50% 的显式权重，赋予 w_f 和 w_s 足够的力量去修补 2Wiki 的逻辑断链
#     2: [0.05, 0.25, 0.70], # Comparative: 比较题依然需要最严密的结构
#     3: [0.15, 0.30, 0.55]  # Default: 兜底意图，激活隐式图谱（w_f）寻找线索
# }

WEIGHTS_NP = np.array([WEIGHT_MATRIX[i] for i in range(NUM_INTENTS)])
DEFAULT_WEIGHTS_NP = np.array(WEIGHT_MATRIX[3])

# --- 3. 意图分类网络 ---
class IntentClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM):
        super(IntentClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6) 
        self.layer_out = nn.Linear(hidden_dim, NUM_INTENTS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.layer_out(x)
    
    def predict_proba(self, embedding_tensor: torch.Tensor, raw_text_list: List[str]) -> torch.Tensor:
        """
        接收向量 + 原始文本
        """
        self.eval() 
        
        # 1. 提取结构特征
        struct_feats = [extract_structural_features(t) for t in raw_text_list]
        struct_tensor = torch.tensor(struct_feats, dtype=torch.float32).to(embedding_tensor.device)
        
        # 2. 拼接: (Batch, 512) + (Batch, 4) -> (Batch, 516)
        combined_input = torch.cat((embedding_tensor, struct_tensor), dim=1)
        
        with torch.no_grad():
            logits = self.forward(combined_input)
            return torch.softmax(logits, dim=1)

# --- 4. 权重调制函数 (保持不变) ---
def dynamic_weight_modulation(
    probs: torch.Tensor, 
    raw_text: str, # <--- 新增参数：我们需要看原始文本
    confidence_threshold: float = CONFIDENCE_THRESHOLD
) -> Tuple[np.ndarray, str]:
    
    if probs.is_cuda: probs_np = probs.cpu().numpy()
    else: probs_np = probs.numpy()
    if probs_np.ndim > 1: probs_np = probs_np[0]

    # --- 🛡️ 规则修正 (Rule-Based Correction) ---
    # 获取结构特征: [comp_score, clause_score, of_count, length]
    feats = extract_structural_features(raw_text)
    comp_score = feats[0]   # or/than 的数量
    clause_score = feats[1] # that/which 的数量
    
    # 规则 1: 绝杀 Comparative
    # 如果完全没有比较词，Comparative 概率直接除以 10 (惩罚)
    if comp_score == 0:
        probs_np[2] *= 0.1 
    
    # 规则 2: 扶持 Retrieval
    # 如果句子短且没有复杂从句，Retrieval 概率乘以 1.2 (奖励)
    if clause_score == 0 and len(raw_text.split()) < 10:
        probs_np[0] *= 1.2

    # 重新归一化 (让概率和为 1)
    probs_np = probs_np / probs_np.sum()
    # -------------------------------------------

    max_prob = np.max(probs_np)
    
    # 低置信度回退
    if max_prob < confidence_threshold:
        return DEFAULT_WEIGHTS_NP, f"(Default, P={max_prob:.2f})"

    # 概率加权混合
    w_final = np.dot(probs_np, WEIGHTS_NP)
    predicted_index = np.argmax(probs_np)
    predicted_type = INTENT_CLASSES[predicted_index]
    
    return w_final, f"({predicted_type}, P={max_prob:.2f})"