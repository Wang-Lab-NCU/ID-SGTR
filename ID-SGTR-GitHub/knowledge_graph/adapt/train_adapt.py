# D:\Code\jupyter\knowledge_graph\adapt\train_adapt.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# 导入工具
# ⚠️ 请确保 utils.py 中已定义 extract_structural_features
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
from utils import get_embeddings_model, extract_structural_features

# --- 1. 配置常量 ---
CSV_PATH = r"D:\Code\jupyter\knowledge_graph\adapt\train2.csv"
# 保存为新模型名，以免覆盖旧的
MODEL_SAVE_PATH = 'adapt/intent_classifier_struct.pth' 

# 维度配置
API_EMBED_DIM = 1024        # 语义向量维度
STRUCT_FEAT_DIM = 4        # 新增：结构特征维度 (or/than, that/which, of, length)
INPUT_DIM = API_EMBED_DIM + STRUCT_FEAT_DIM # 总维度 = 1028

NUM_INTENTS = 3 
HIDDEN_DIM = 64            
DROPOUT_RATE = 0.6
BATCH_SIZE = 64 
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.02
NUM_EPOCHS = 60 
PATIENCE = 10

LABEL_MAP = {"Retrieval": 0, "Reasoning": 1, "Comparative": 2}

# --- 2. 模型定义 (Structure Fusion Version) ---
class IntentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(IntentClassifier, self).__init__()
        # 输入层接受 1028 维
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE) 
        self.layer_out = nn.Linear(hidden_dim, NUM_INTENTS)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.bn1(x) 
        x = self.relu(x)
        x = self.dropout(x)
        return self.layer_out(x)

# --- 3. 增强版 Dataset ---
class FusionDataset(Dataset):
    def __init__(self, embeddings, raw_texts, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # 计算结构特征 (关键步骤)
        # 这将调用 utils.py 中的函数提取 [comp_score, clause_score, of_count, length]
        feats = [extract_structural_features(t) for t in raw_texts]
        self.struct_feats = torch.tensor(feats, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 拼接语义向量(1024) 和 结构特征(4) -> (1028)
        combined_input = torch.cat((self.embeddings[idx], self.struct_feats[idx]), dim=0)
        return combined_input, self.labels[idx]

# --- 4. 辅助函数 ---
def load_data_and_process():
    print(f"读取数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, sep='|', encoding='utf-8', on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype(str).str.strip()
    else:
        raise ValueError("CSV中找不到 'Type' 列")
        
    df = df[df['Type'].isin(LABEL_MAP.keys())]
    
    y = np.array([LABEL_MAP[t] for t in df['Type']])
    questions = df['question'].tolist()
    
    # 生成 Embedding
    print("正在生成 Embedding (这可能需要几分钟)...")
    embed_model = get_embeddings_model(dimensions=API_EMBED_DIM,)
    
    embeddings = []
    embed_batch_size = 64
    
    for i in tqdm(range(0, len(questions), embed_batch_size), desc="Embedding"):
        batch = questions[i:i+embed_batch_size]
        batch = [str(t).strip() if str(t).strip() else " " for t in batch]
        try:
            embs = embed_model.embed_documents(batch)
            embeddings.extend(embs)
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            raise e
            
    embeddings = np.array(embeddings)
    if embeddings.shape[1] != API_EMBED_DIM:
        raise ValueError(f"API返回维度 {embeddings.shape[1]} 不等于配置的 {API_EMBED_DIM}")
        
    return embeddings, questions, y

# --- 5. 主程序 ---
if __name__ == '__main__':
    # 固定种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # A. 数据处理
    try:
        X_emb, X_text, y = load_data_and_process()
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        exit()

    # B. 划分数据集 (保留索引以获取对应的原始文本)
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    # C. 构建 Dataset
    # ⚠️ 注意：这里必须传入 X_text (原始文本列表)，以便 Dataset 内部提取结构特征
    train_dataset = FusionDataset(X_emb[train_idx], [X_text[i] for i in train_idx], y[train_idx])
    val_dataset = FusionDataset(X_emb[val_idx], [X_text[i] for i in val_idx], y[val_idx])

    # D. 类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"类别权重: {weights_tensor.cpu().numpy()}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # E. 初始化模型
    model = IntentClassifier(INPUT_DIM, HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    # F. 训练循环
    best_acc = 0.0
    patience_counter = 0
    
    print("\n--- 开始训练 (Structure Fusion Mode) ---")
    for epoch in range(NUM_EPOCHS):
        # 1. Train
        model.train()
        train_loss = 0
        train_correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            
        # 2. Val
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        # 3. Metrics
        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
        
        # 4. Save & Early Stop
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n✋ 早停触发！最佳验证集准确率: {best_acc:.3f}")
                break
                
    # --- G. 最终详细评估 ---
    print("\n=== 加载最佳模型进行评估 ===")
    
    # 1. 加载权重
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 2. 遍历验证集收集结果
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    # 3. 打印报告
    target_names = ["Retrieval", "Relational", "Comparative"]
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=3))
    print(f"模型文件已保存至: {MODEL_SAVE_PATH}")