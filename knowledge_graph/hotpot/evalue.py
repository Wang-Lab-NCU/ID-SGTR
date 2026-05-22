# import pandas as pd
# import re
# import string
# import os
# import sys
# from collections import Counter

# # ==========================================
# # 配置
# # ==========================================
# DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\hotpot\result\q3"
# INPUT_FILE = "query_results_agent_100glm46_1_27.csv"
# FULL_PATH = os.path.join(DATA_ROOT, INPUT_FILE)

# # ==========================================
# # 1. 官方标准化函数 (不动它)
# # ==========================================
# def normalize_answer(s):
#     """
#     HotpotQA 官方评估逻辑
#     """
#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))

# # ==========================================
# # 2. [修正] 更稳健的答案提取
# # ==========================================
# def extract_final_answer(pred_text):
#     if not isinstance(pred_text, str): 
#         return ""
    
#     # 1. 预处理：只移除 Markdown 加粗，别的标点都不动（交给官方函数）
#     # 防止把 "Schindler's List" 变成 "Schindlers List" 导致和官方处理结果不一致
#     text = pred_text.replace('**', '').replace('__', '')
    
#     # 2. 核心提取逻辑：基于字符串分割，而非复杂的正则
#     # 查找所有可能的“答案标签”
#     markers = ["Final Answer:", "Final Answer", "Answer:", "Conclusion:", "Answer"]
    
#     target_part = text # 默认就是全文（如果找不到标签）
    
#     for marker in markers:
#         # 查找最后一个标签的位置（防止 CoT 中间提到 Final Answer）
#         # 使用 case-insensitive 的查找
#         lower_text = text.lower()
#         lower_marker = marker.lower()
        
#         if lower_marker in lower_text:
#             # rsplit 从右边切分，取最后一部分
#             # 注意：这里要用原文本切分，不能用 lower_text
#             # 我们找到 index，然后切分原文本
#             last_index = lower_text.rfind(lower_marker)
#             target_part = text[last_index + len(marker):]
#             break # 找到最高优先级的标签就停止
    
#     # 3. 清洗提取后的部分
#     # 去除首尾空白
#     target_part = target_part.strip()
    
#     # [关键修复] 处理 "Final Answer:\nAnswerString" 这种情况
#     # 如果开头是换行符，去掉
#     lines = [line.strip() for line in target_part.split('\n') if line.strip()]
    
#     if not lines:
#         return ""
        
#     # 通常答案就在第一行非空文本里
#     # 但有时候是 "The answer is X."，我们需要去除 "The answer is" 这种废话吗？
#     # 官方 normalize_answer 会去除 "the"，所以不用太担心
#     final_candidate = lines[0]
    
#     # 去除末尾的句号（大模型喜欢加句号）
#     if final_candidate.endswith('.'):
#         final_candidate = final_candidate[:-1]
        
#     # 去除可能的反引号 (`)
#     final_candidate = final_candidate.replace('`', '')
    
#     return final_candidate

# # ==========================================
# # 3. 指标计算（新增 Contain-Match 函数）
# # ==========================================
# def contain_match_score(prediction, ground_truth):
#     """Contain-Match 准确率（子串匹配）"""
#     norm_pred = normalize_answer(prediction)
#     norm_gold = normalize_answer(ground_truth)
    
#     # 对于 yes/no/noanswer，需要完全匹配
#     if norm_gold in ['yes', 'no', 'noanswer']:
#         return norm_pred == norm_gold
    
#     # 核心逻辑：检查标准化后的黄金答案是否在标准化后的预测答案中
#     return norm_gold in norm_pred

# def f1_score(prediction, ground_truth):
#     normalized_prediction = normalize_answer(prediction)
#     normalized_ground_truth = normalize_answer(ground_truth)

#     ZERO_METRIC = (0, 0, 0)

#     if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
#         return ZERO_METRIC
#     if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
#         return ZERO_METRIC

#     prediction_tokens = normalized_prediction.split()
#     ground_truth_tokens = normalized_ground_truth.split()
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
    
#     if num_same == 0:
#         return ZERO_METRIC
    
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1, precision, recall

# def exact_match_score(prediction, ground_truth):
#     return (normalize_answer(prediction) == normalize_answer(ground_truth))

# # ==========================================
# # 4. 主程序
# # ==========================================
# if __name__ == "__main__":
#     try:
#         df = pd.read_csv(FULL_PATH, sep="|", on_bad_lines='skip')
#         df.columns = [c.strip() for c in df.columns]
#     except Exception as e:
#         print(f"读取失败: {e}")
#         sys.exit(1)

#     metrics = {'em': 0, 'contain_acc': 0, 'f1': 0, 'prec': 0, 'recall': 0}  # 添加 contain_acc

#     print(f"🏆 正在评测 {len(df)} 条数据 (稳健修复版)...")
    
#     for idx, row in df.iterrows():
#         gold = str(row.get('gold_answer', row.get('answer', '')))
#         raw_pred = str(row.get('pred_answer', row.get('prediction', '')))
        
#         pred = extract_final_answer(raw_pred)
        
#         em = exact_match_score(pred, gold)
#         contain = contain_match_score(pred, gold)  # 新增
#         f1, prec, recall = f1_score(pred, gold)
        
#         metrics['em'] += float(em)
#         metrics['contain_acc'] += float(contain)  # 新增
#         metrics['f1'] += f1
#         metrics['prec'] += prec
#         metrics['recall'] += recall

#     count = len(df)
    
#     print("\n" + "="*50)
#     print("🎓 评分结果")
#     print("="*50)
#     print(f"Exact Match (EM):         {metrics['em'] / count:.2%}")
#     print(f"Contain-Match (Contain-Acc): {metrics['contain_acc'] / count:.2%}")  # 新增
#     print(f"F1 Score:                 {metrics['f1'] / count:.2%}")
#     print(f"Precision:                {metrics['prec'] / count:.2%}")
#     print(f"Recall:                   {metrics['recall'] / count:.2%}")
#     print("-" * 50)
import pandas as pd
import re
import string
import os
import sys
from collections import Counter, defaultdict

# ==========================================
# 配置
# ==========================================
DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\hotpot\result\resolve"
INPUT_FILE = "query_results_agent_1000Qwen3-8B_4_15_global.csv"
FULL_PATH = os.path.join(DATA_ROOT, INPUT_FILE)

# ==========================================
# 1. 官方标准化函数
# ==========================================
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def improved_normalize(text):
        text = text.lower()
        # 1. 日期/范围标准化
        text = text.replace(" until ", "-").replace(" to ", "-")
        
        # 2. 列表连接词标准化
        text = text.replace(" and ", ", ")
        
        # 3. 移除多余的冠词和标点 (保留原有逻辑)
        # ... (保留你代码中的 remove_articles, remove_punc) ...
    
        return text.strip()
    
    return white_space_fix(remove_articles(remove_punc(improved_normalize(lower(s)))))
# ==========================================
# 2. 答案提取逻辑
# ==========================================
def extract_final_answer(pred_text):
    if not isinstance(pred_text, str): 
        return ""
    
    text = pred_text.replace('**', '').replace('__', '')
    markers = ["Final Answer:", "Final Answer", "Answer:", "Conclusion:", "Answer"]
    target_part = text 
    
    for marker in markers:
        lower_text = text.lower()
        lower_marker = marker.lower()
        if lower_marker in lower_text:
            last_index = lower_text.rfind(lower_marker)
            target_part = text[last_index + len(marker):]
            break 
    
    target_part = target_part.strip()
    lines = [line.strip() for line in target_part.split('\n') if line.strip()]
    
    if not lines:
        return ""
        
    final_candidate = lines[0]
    if final_candidate.endswith('.'):
        final_candidate = final_candidate[:-1]
    final_candidate = final_candidate.replace('`', '')
    
    return final_candidate

# ==========================================
# 3. 核心指标计算
# ==========================================
def contain_match_score(prediction, ground_truth):
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)
    if norm_gold in ['yes', 'no', 'noanswer']:
        return norm_pred == norm_gold
    return norm_gold in norm_pred

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    try:
        df = pd.read_csv(FULL_PATH, sep="|", on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"读取失败: {e}")
        sys.exit(1)

    # 全局指标
    metrics = {'em': 0, 'contain_acc': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    
    # 统计容器
    intent_stats = defaultdict(lambda: {'count': 0, 'em': 0, 'contain': 0, 'f1': 0})
    strategy_stats = defaultdict(lambda: {'count': 0, 'em': 0, 'contain': 0, 'f1': 0})
    
    # 映射表
    INTENT_MAP = {
        "Relational": "Reasoning", "Retrieval": "Retrieval", 
        "Comparative": "Comparative", "Default": "Default","Reasoning": "Reasoning"
    }

    print(f"🏆 正在评测 {len(df)} 条数据 (含意图与跳数策略双维度分析)...")
    
    for idx, row in df.iterrows():
        gold = str(row.get('gold_answer', row.get('answer', '')))
        raw_pred = str(row.get('pred_answer', row.get('prediction', '')))
        strategy_col = str(row.get('strategy', ''))
        
        # --- 1. 意图解析 (Intent) ---
        match = re.search(r'\(([^,]+),', strategy_col)
        if match:
            raw_intent = match.group(1).strip()
        else:
            raw_intent = "Default" if ("Default" in strategy_col or "Fallback" in strategy_col) else "Default"
        
        intent_type = INTENT_MAP.get(raw_intent, "Default")
        
        # --- 2. 策略解析 (Hop Strategy) ---
        # 格式示例: "(Comparative, ...) -> Agent-Zero-Shot"
        # 或者仅仅是 "Fallback"
        if "->" in strategy_col:
            # 取 "->" 后面的部分并去除首尾空格
            strat_name = strategy_col.split("->")[1].strip()
        else:
            # 如果没有 ->, 检查是否直接是 Fallback
            if "Fallback" in strategy_col:
                strat_name = "Fallback"
            else:
                strat_name = "Unknown"
        
        # 归一化策略名称 (防止原有数据里有细微差别)
        valid_strategies = ["Agent-Zero-Shot", "Agent-Hop-1", "Agent-Hop-2", "Agent-Hop-3", "Fallback"]
        # 如果解析出的名称不在标准列表里，保留原名以便发现问题，或者归类为 Unknown
        # 这里直接用 strat_name 即可，因为你的数据看起来很规范
        
        # --- 3. 评测计算 ---
        pred = extract_final_answer(raw_pred)
        
        em = exact_match_score(pred, gold)
        contain = contain_match_score(pred, gold)
        f1, prec, recall = f1_score(pred, gold)
        
        # 更新全局
        metrics['em'] += float(em)
        metrics['contain_acc'] += float(contain)
        metrics['f1'] += f1
        metrics['prec'] += prec
        metrics['recall'] += recall

        # 更新意图统计
        intent_stats[intent_type]['count'] += 1
        intent_stats[intent_type]['em'] += float(em)
        intent_stats[intent_type]['contain'] += float(contain)
        intent_stats[intent_type]['f1'] += f1
        
        # 更新策略统计
        strategy_stats[strat_name]['count'] += 1
        strategy_stats[strat_name]['em'] += float(em)
        strategy_stats[strat_name]['contain'] += float(contain)
        strategy_stats[strat_name]['f1'] += f1

    count = len(df)
    
    # --------------------------------------------------------
    # 输出 1: 全局总体评分
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("📊 总体评分结果 (Overall Evaluation)")
    print("="*60)
    print(f"Exact Match (EM):          {metrics['em'] / count:.2%}")
    print(f"Contain-Match (Acc):       {metrics['contain_acc'] / count:.2%}")
    print(f"F1 Score:                  {metrics['f1'] / count:.2%}")
    print(f"Precision:                 {metrics['prec'] / count:.2%}")
    print(f"Recall:                    {metrics['recall'] / count:.2%}")
    print("="*60)

    # --------------------------------------------------------
    # 输出 2: 按意图分类 (Intent)
    # --------------------------------------------------------
    print("\n" + "="*75)
    print("📑 意图分类准确率 (Per-Intent Performance)")
    print("="*75)
    print(f"{'Intent Type':<15} | {'Count':<6} | {'EM':<8} | {'Contain-Acc':<12} | {'F1 Score':<8}")
    print("-" * 75)
    
    intent_order = ['Retrieval', 'Reasoning', 'Comparative', 'Default']
    for intent in intent_order:
        data = intent_stats[intent]
        n = data['count']
        if n > 0:
            print(f"{intent:<15} | {n:<6} | {data['em']/n:<8.2%} | {data['contain']/n:<12.2%} | {data['f1']/n:<8.2%}")
        else:
            print(f"{intent:<15} | {0:<6} | {'N/A':<8} | {'N/A':<12} | {'N/A':<8}")
    print("="*75)

    # --------------------------------------------------------
    # 输出 3: 按跳数策略分类 (Hop Strategy)
    # --------------------------------------------------------
    print("\n" + "="*75)
    print("🐇 跳数策略准确率 (Agent Hop Strategy Performance)")
    print("="*75)
    print(f"{'Strategy':<18} | {'Count':<6} | {'EM':<8} | {'Contain-Acc':<12} | {'F1 Score':<8}")
    print("-" * 75)
    
    # 定义你要的显示顺序
    strat_order = ['Agent-Zero-Shot', 'Agent-Hop-1', 'Agent-Hop-2', 'Agent-Hop-3', 'Fallback']
    
    # 为了防止有不在列表里的策略（比如 Unknown），把它们也加在后面
    all_keys = list(strategy_stats.keys())
    remaining_keys = [k for k in all_keys if k not in strat_order]
    final_order = strat_order + remaining_keys
    
    for strat in final_order:
        data = strategy_stats[strat]
        n = data['count']
        if n > 0:
            print(f"{strat:<18} | {n:<6} | {data['em']/n:<8.2%} | {data['contain']/n:<12.2%} | {data['f1']/n:<8.2%}")
        else:
            # 如果样本里没有 Agent-Hop-3，这里会显示 N/A，这是正常的
            print(f"{strat:<18} | {0:<6} | {'N/A':<8} | {'N/A':<12} | {'N/A':<8}")
            
    print("="*75 + "\n")