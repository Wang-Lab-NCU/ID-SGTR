from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
import os
import time
import threading
import requests # 确保顶部导入了 requests

from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 基于时间窗口 (10分钟) 的 API Key 安全轮转池
# ==========================================
_siliconflow_keys = [
    k.strip() for k in os.getenv('SILICONFLOW_API_KEYS-2', '').split(',') if k.strip()
]

_current_key_index = 0
_last_switch_time = time.time()  # 记录上一次切换的时间戳
_key_lock = threading.Lock()
_SWITCH_INTERVAL = 600  # 切换间隔，单位为秒 (600秒 = 10分钟)

def get_current_siliconflow_key():
    """
    获取当前生效的 API Key（线程安全）。
    每 10 分钟自动切换到下一个 Key，防止高频并发触发 403 封控。
    """
    global _current_key_index, _last_switch_time
    
    if not _siliconflow_keys:
        raise ValueError("❌ 未在环境变量中找到 SILICONFLOW_API_KEYS，请检查 .env 文件。")
    
        # 如果只有一个 Key，直接返回，无需轮转
    if len(_siliconflow_keys) == 1:
        return _siliconflow_keys[0]
    
    with _key_lock:
        current_time = time.time()
        # 如果当前时间减去上次切换时间大于等于 10 分钟 (600秒)，则执行切换
        if current_time - _last_switch_time >= _SWITCH_INTERVAL:
            _current_key_index = (_current_key_index + 1) % len(_siliconflow_keys)
            _last_switch_time = current_time
            # print(f"\n🔄 [API Key 调度] 已满 10 分钟，自动切换至新的 API Key (池中索引: {_current_key_index})\n")
            
        return _siliconflow_keys[_current_key_index]


# ==========================================
# 2. 基础模型获取函数 (自定义原生 SiliconFlow Embeddings)
# ==========================================

class SiliconFlowEmbeddings:
    """
    自定义的 SiliconFlow Embedding 包装器。
    彻底绕过 LangChain 自带的 tiktoken 下载检测，解决 Azure 连接超时问题。
    """
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.model_name = model_name
        self.api_base = os.getenv('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')
        self.url = f"{self.api_base}/embeddings"

    def embed_documents(self, texts):
        # 每次调用动态获取当前生效的轮转 Key
        current_api_key = get_current_siliconflow_key()
        
        headers = {
            "Authorization": f"Bearer {current_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=120)
            response.raise_for_status() # 如果是 4xx 或 5xx 会直接抛出异常
            data = response.json()
            # 提取向量并按照传入的 texts 顺序返回
            return [item['embedding'] for item in data['data']]
        except Exception as e:
            print(f"❌ SiliconFlow Embedding 请求失败: {e}")
            if 'response' in locals() and response is not None:
                print(f"📝 错误详情: {response.text}")
            raise e

    def embed_query(self, text):
        # 查询通常只是一条文本
        return self.embed_documents([text])[0]


def get_embeddings_model(dimensions=os.getenv('dimensions')):
    """
    获取 Embedding 模型实例
    """
    model_type = os.getenv('EMBEDDINGS_MODEL', 'siliconflow')
    
    if model_type == 'siliconflow':
        # 使用我们自己写的极简无干涉 Wrapper
        return SiliconFlowEmbeddings(
            model_name=os.getenv('SILICONFLOW_EMBEDDINGS_MODEL', 'BAAI/bge-m3')
        )
        
    elif model_type == 'zhipuai':
        # 保留智谱作为备用兜底
        return ZhipuAIEmbeddings(
            api_key=os.getenv('ZHIPUAI_API_KEY'),
            model=os.getenv('ZHIPUAI_EMBEDDINGS_MODEL'),
            dimensions=int(dimensions) if dimensions else 512
        )
    else:
        raise ValueError(f"未知的 Embedding 模型类型: {model_type}")

def get_llm_model(api_key=os.getenv('ZHIPUAI_API_KEY'), model=os.getenv('ZHIPUAI_LLM_MODEL')):
    model_map = {
        'zhipuai': ChatZhipuAI(
            api_key=api_key,
            temperature=os.getenv('ZHIPUAI_TEMPERATURE'),
            model=model,
            max_tokens=os.getenv('MAX_TOKENS'),
            top_p=(os.getenv('ZHIPUAI_TOP_P')),
            Verbose=os.getenv('VERBOSE'),
        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))

# ==========================================
# 3. 动态调度 Chat 模型
# ==========================================

def get_chat_model(task_type="extraction"):
    """
    根据任务类型返回不同参数配置的模型
    支持 SiliconFlow，并自动进行基于时间窗口的 API Key 轮转
    """
    # 获取当前时间窗口内处于激活状态的 Key
    current_api_key = get_current_siliconflow_key()

    base_kwargs = {
        "model": os.getenv('SILICONFLOW_MODEL'),
        "openai_api_key": current_api_key,
        "openai_api_base": os.getenv('SILICONFLOW_BASE_URL'),
        "max_tokens": 4096,
        "timeout": 120,
    }

    config_map = {
        "extraction": {
            "temperature": 0.1,
            "extra_body": {
                "top_p": 0.6,
                "top_k": 50,
                "frequency_penalty": 0.0,
                "enable_thinking": False 
            }
        },
        "reasoning": {
            "temperature": 0.5,
            "extra_body": {
                "top_p": 0.9,
                "top_k": 50,
                "enable_thinking": False,
                "thinking_budget": 1024
            }
        },
        "kg_query": {
            "temperature": 0.5, 
            "extra_body": {
                "top_p": 0.8,
                "top_k": 50,
                "frequency_penalty": 0.0, 
                "presence_penalty": 0.0,
                "enable_thinking": True,
                "thinking_budget": 2048
            }
        }
    }

    task_config = config_map.get(task_type, config_map["extraction"])
    final_kwargs = {**base_kwargs, **task_config}

    # 返回兼容 OpenAI 格式的 SiliconFlow 模型实例
    return ChatOpenAI(**final_kwargs)



def extract_structural_features(text):
    """
    提取 4 维结构特征 (V2.0 增强版)
    """
    if not isinstance(text, str):
        text = str(text)
    
    text_lower = text.lower()
    
    # --- 1. Comparative Keywords (比较类) ---
    # 英文: or, than, vs, rank, compare, difference between
    # 中文: 比 (A比B), 哪个 (哪个更..), 区别, 差异
    # 注意: 为了避免 'difference engine' 误判，我们检查 'difference between'
    
    comp_indicators = [
        ' or ', ' than ', ' vs ', ' vs. ', ' rank ', ' compare ', 
        ' better ', ' worse ', ' more ', ' less ',
        ' difference between ', # 关键修复：只匹配 "difference between"
        ' 比 ', ' 哪个 ', ' 区别 ', ' 差异 ' # 中文支持
    ]
    
    comp_score = 0
    for k in comp_indicators:
        if k in text_lower:
            comp_score += 1
            
    # 特殊补丁：如果出现了 "compare" 但没有 "to/with/between"，可能也是命令式比较
    if text_lower.startswith("compare ") or text_lower.startswith("rank "):
        comp_score += 1

    # --- 2. Relational / Clause Keywords (关系类) ---
    # 英文: that, which, who, whose, where (在从句中)
    # 中文: 的 (多层嵌套时), 所在, 位于
    rel_indicators = [
        ' that ', ' which ', ' who ', ' whose ', 
        ' where ', # where is ... located (Retrieval) vs country where ... (Relational) 很难分，暂时保留
        ' 所在 ', ' 位于 '
    ]
    
    clause_score = 0
    for k in rel_indicators:
        if k in text_lower:
            clause_score += 1
            
    # --- 3. 'of' depth (层级深度) ---
    # 英文 'of'，中文 '的'
    # 注意：中文 '的' 出现频率极高，只作为辅助特征
    of_count = text_lower.count(' of ') + text_lower.count('的') * 0.5 
    
    # --- 4. Length (简单归一化) ---
    length = len(text.split()) / 20.0
    
    return [float(comp_score), float(clause_score), float(of_count), length]