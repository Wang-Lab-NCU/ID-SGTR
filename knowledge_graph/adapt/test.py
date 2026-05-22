
import torch
import numpy as np
from adapt import IntentClassifier, dynamic_weight_modulation, INPUT_DIM, HIDDEN_DIM
from utils import get_embeddings_model

# 1. 设置设备与模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntentClassifier(INPUT_DIM, HIDDEN_DIM).to(device)

# 修改为新训练的模型路径
MODEL_PATH = 'adapt/intent_classifier_struct.pth' 

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ 纯语义模型加载成功: {MODEL_PATH}")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    print("注意：因为输入维度从 515 变为了 512，旧的权重文件无法加载，请先运行 train_adapt.py 重新训练。")
    exit()

model.eval()

# 2. 初始化 API (512维)
print("正在初始化 Embedding API...")
embed_model = get_embeddings_model(dimensions=1024) 

test_queries = [
    # --- Retrieval (简单查找/定义/事实) ---
    "What is the atomic number of Gold?", 
    "Define the concept of 'Recursion' in computer science.",
    "Where is the headquarters of Microsoft located?",
    "When was the first iPhone released?",
    "Python的发明者是谁？",
    "What is the capital city of Canada?", 
    
    # --- Comparative (比较/二选一/排序) ---
    "Which planet is larger, Mars or Earth?",
    "Who was born first, Albert Einstein or Isaac Newton?",
    "Did 'Avengers: Endgame' earn more box office revenue than 'Avatar'?",
    "Are distinct prime factors of 12 and 18 the same?", # 陷阱：same 关键词
    "百度和谷歌哪个成立得更早？",
    "Is Python slower than C++?",
    "Which creates more energy, nuclear fission or fusion?",

    # --- Relational (多跳推理/嵌套关系/属性的属性) ---
    # 关键特征：contains "of the", "that", "who", "which"
    "Who is the director of the movie that starred Heath Ledger as the Joker?",
    "The author of 'Harry Potter' was born in which city?",
    "What is the currency of the country where Tokyo is located?",
    "Describe the wife of the 44th President of the United States.",
    "周杰伦的妻子的出生地在哪里？", # 实体->关系->实体->关系
    "The company that acquired DeepMind is headquartered in which country?",
    "Who is the CEO of the parent company of Instagram?"
]


# --- 打印表头 ---
header = f"{'Query':<50} | {'Strategy':<25} | {'Weights [wf, ws, we]':<30}"
print("-" * 110)
print(header)
print("-" * 110)

for query in test_queries:
    # A. 生成向量
    try:
        emb_list = embed_model.embed_documents([query])
        emb_tensor = torch.tensor(np.array(emb_list), dtype=torch.float32).to(device)
    except Exception as e:
        print(f"Embedding Error: {e}")
        continue
    
    # B. 推理 
    probs = model.predict_proba(emb_tensor, [query])    
    # C. 获取策略和权重
    weights, strategy_str = dynamic_weight_modulation(probs, query) # 传入 query    
    # D. 格式化权重显示
    w_str = f"[{weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f}]"
    
    # E. 打印
    q_display = (query[:47] + '..') if len(query) > 47 else query
    print(f"{q_display:<50} | {strategy_str:<25} | {w_str}")

print("-" * 110)