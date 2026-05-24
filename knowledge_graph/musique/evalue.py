import pandas as pd
import re
import string
import os
import sys
from collections import Counter, defaultdict

# ==========================================
# Configuration
# ==========================================
DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\musique\result\resolve"
INPUT_FILE = "query_results_agent_1000Qwen3-8B_4_15_global.csv"
FULL_PATH = os.path.join(DATA_ROOT, INPUT_FILE)


# ==========================================
# 1. Official normalization function
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
        # 1. Date/range normalization
        text = text.replace(" until ", "-").replace(" to ", "-")
        
        # 2. List conjunction normalization
        text = text.replace(" and ", ", ")
        
        # 3. Remove extra articles and punctuation (keep original logic)
        # ... (keep remove_articles, remove_punc from your code) ...
    
        return text.strip()
    
    return white_space_fix(remove_articles(remove_punc(improved_normalize(lower(s)))))

# ==========================================
# 2. Answer extraction logic
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
# 3. Core metric calculation
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
# 4. Main program
# ==========================================
if __name__ == "__main__":
    try:
        df = pd.read_csv(FULL_PATH, sep="|", on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"Failed to read file: {e}")
        sys.exit(1)

    # Global metrics
    metrics = {'em': 0, 'contain_acc': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    
    # Statistics containers
    intent_stats = defaultdict(lambda: {'count': 0, 'em': 0, 'contain': 0, 'f1': 0})
    strategy_stats = defaultdict(lambda: {'count': 0, 'em': 0, 'contain': 0, 'f1': 0})
    
    # Mapping table
    INTENT_MAP = {
        "Relational": "Reasoning", "Retrieval": "Retrieval", 
        "Comparative": "Comparative", "Default": "Default","Reasoning": "Reasoning"
    }

    print(f"🏆 Evaluating {len(df)} entries (with dual-dimension analysis: intent and hop strategy)...")
    
    for idx, row in df.iterrows():
        gold = str(row.get('gold_answer', row.get('answer', '')))
        raw_pred = str(row.get('pred_answer', row.get('prediction', '')))
        strategy_col = str(row.get('strategy', ''))
        
        # --- 1. Intent parsing ---
        match = re.search(r'\(([^,]+),', strategy_col)
        if match:
            raw_intent = match.group(1).strip()
        else:
            raw_intent = "Default" if ("Default" in strategy_col or "Fallback" in strategy_col) else "Default"
        
        intent_type = INTENT_MAP.get(raw_intent, "Default")
        
        # --- 2. Strategy parsing (Hop Strategy) ---
        # Format example: "(Comparative, ...) -> Agent-Zero-Shot"
        # Or simply "Fallback"
        if "->" in strategy_col:
            # Take the part after "->" and strip whitespace
            strat_name = strategy_col.split("->")[1].strip()
        else:
            # If no "->", check if directly "Fallback"
            if "Fallback" in strategy_col:
                strat_name = "Fallback"
            else:
                strat_name = "Unknown"
        
        # Normalize strategy name (in case there are subtle differences in the original data)
        valid_strategies = ["Agent-Zero-Shot", "Agent-Hop-1", "Agent-Hop-2", "Agent-Hop-3", "Fallback"]
        # If the parsed name is not in the standard list, keep it as is to identify issues, or classify as Unknown
        # Just use strat_name directly because your data looks clean
        
        # --- 3. Evaluation calculation ---
        pred = extract_final_answer(raw_pred)
        
        em = exact_match_score(pred, gold)
        contain = contain_match_score(pred, gold)
        f1, prec, recall = f1_score(pred, gold)
        
        # Update global metrics
        metrics['em'] += float(em)
        metrics['contain_acc'] += float(contain)
        metrics['f1'] += f1
        metrics['prec'] += prec
        metrics['recall'] += recall

        # Update intent statistics
        intent_stats[intent_type]['count'] += 1
        intent_stats[intent_type]['em'] += float(em)
        intent_stats[intent_type]['contain'] += float(contain)
        intent_stats[intent_type]['f1'] += f1
        
        # Update strategy statistics
        strategy_stats[strat_name]['count'] += 1
        strategy_stats[strat_name]['em'] += float(em)
        strategy_stats[strat_name]['contain'] += float(contain)
        strategy_stats[strat_name]['f1'] += f1

    count = len(df)
    
    # --------------------------------------------------------
    # Output 1: Overall global scores
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("📊 Overall Evaluation Results")
    print("="*60)
    print(f"Exact Match (EM):          {metrics['em'] / count:.2%}")
    print(f"Contain-Match (Acc):       {metrics['contain_acc'] / count:.2%}")
    print(f"F1 Score:                  {metrics['f1'] / count:.2%}")
    print(f"Precision:                 {metrics['prec'] / count:.2%}")
    print(f"Recall:                    {metrics['recall'] / count:.2%}")
    print("="*60)

    # --------------------------------------------------------
    # Output 2: By intent type
    # --------------------------------------------------------
    print("\n" + "="*75)
    print("📑 Per-Intent Performance")
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
    # Output 3: By hop strategy
    # --------------------------------------------------------
    print("\n" + "="*75)
    print("🐇 Agent Hop Strategy Performance")
    print("="*75)
    print(f"{'Strategy':<18} | {'Count':<6} | {'EM':<8} | {'Contain-Acc':<12} | {'F1 Score':<8}")
    print("-" * 75)
    
    # Define the order you want to display
    strat_order = ['Agent-Zero-Shot', 'Agent-Hop-1', 'Agent-Hop-2', 'Agent-Hop-3', 'Fallback']
    
    # In case there are strategies not in the list (e.g., Unknown), append them afterwards
    all_keys = list(strategy_stats.keys())
    remaining_keys = [k for k in all_keys if k not in strat_order]
    final_order = strat_order + remaining_keys
    
    for strat in final_order:
        data = strategy_stats[strat]
        n = data['count']
        if n > 0:
            print(f"{strat:<18} | {n:<6} | {data['em']/n:<8.2%} | {data['contain']/n:<12.2%} | {data['f1']/n:<8.2%}")
        else:
            # If there are no samples for Agent-Hop-3, this will show N/A, which is normal
            print(f"{strat:<18} | {0:<6} | {'N/A':<8} | {'N/A':<12} | {'N/A':<8}")
            
    print("="*75 + "\n")
