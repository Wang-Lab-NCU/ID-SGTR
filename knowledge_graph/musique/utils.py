from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
import os
import time
import threading
import requests  # Ensure requests is imported at the top

from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. API key secure rotation pool based on time window (10 minutes)
# ==========================================
# Read all keys from environment variable, split by comma, automatically strip spaces and remove empty values
_siliconflow_keys = [
    k.strip() for k in os.getenv('SILICONFLOW_API_KEYS-2', '').split(',') if k.strip()
]

_current_key_index = 0
_last_switch_time = time.time()  # Timestamp of last switch
_key_lock = threading.Lock()
_SWITCH_INTERVAL = 600  # Switch interval in seconds (600 seconds = 10 minutes)

def get_current_siliconflow_key():
    """
    Get the currently active API key (thread-safe).
    Automatically switches to the next key every 10 minutes to prevent 403 blocking due to high-frequency concurrency.
    """
    global _current_key_index, _last_switch_time
    
    if not _siliconflow_keys:
        raise ValueError("❌ SILICONFLOW_API_KEYS not found in environment variables. Please check the .env file.")
    
    # If there is only one key, return it directly without rotation
    if len(_siliconflow_keys) == 1:
        return _siliconflow_keys[0]
    
    with _key_lock:
        current_time = time.time()
        # If the time since last switch is >= 10 minutes (600 seconds), perform the switch
        if current_time - _last_switch_time >= _SWITCH_INTERVAL:
            _current_key_index = (_current_key_index + 1) % len(_siliconflow_keys)
            _last_switch_time = current_time
            # print(f"\n🔄 [API Key Scheduler] 10 minutes elapsed, automatically switching to new API key (index in pool: {_current_key_index})\n")
            
        return _siliconflow_keys[_current_key_index]


# ==========================================
# 2. Basic model retrieval functions (custom native SiliconFlow Embeddings)
# ==========================================

class SiliconFlowEmbeddings:
    """
    Custom SiliconFlow Embedding wrapper.
    Completely bypasses the tiktoken download detection that comes with LangChain, solving Azure connection timeout issues.
    """
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.model_name = model_name
        self.api_base = os.getenv('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')
        self.url = f"{self.api_base}/embeddings"

    def embed_documents(self, texts):
        # Dynamically get the currently active rotating key for each call
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
            response.raise_for_status()  # Raises exception for 4xx or 5xx status codes
            data = response.json()
            # Extract embeddings and return them in the order of the input texts
            return [item['embedding'] for item in data['data']]
        except Exception as e:
            print(f"❌ SiliconFlow Embedding request failed: {e}")
            if 'response' in locals() and response is not None:
                print(f"📝 Error details: {response.text}")
            raise e

    def embed_query(self, text):
        # A query is usually just a single text
        return self.embed_documents([text])[0]


def get_embeddings_model(dimensions=os.getenv('dimensions')):
    """
    Get the embedding model instance.
    """
    model_type = os.getenv('EMBEDDINGS_MODEL', 'siliconflow')
    
    if model_type == 'siliconflow':
        # Use our minimal non-interfering wrapper
        return SiliconFlowEmbeddings(
            model_name=os.getenv('SILICONFLOW_EMBEDDINGS_MODEL', 'BAAI/bge-large-zh-v1.5')
        )
        
    elif model_type == 'zhipuai':
        # Keep Zhipu as a fallback
        return ZhipuAIEmbeddings(
            api_key=os.getenv('ZHIPUAI_API_KEY'),
            model=os.getenv('ZHIPUAI_EMBEDDINGS_MODEL'),
            dimensions=int(dimensions) if dimensions else 1024
        )
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")

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
# 3. Dynamically dispatched Chat model
# ==========================================

def get_chat_model(task_type="extraction"):
    """
    Return a model with different parameter configurations based on the task type.
    Supports SiliconFlow and automatically performs time-window-based API key rotation.
    """
    # Get the active key within the current time window
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

    # Return a SiliconFlow model instance compatible with OpenAI format
    return ChatOpenAI(**final_kwargs)


def extract_structural_features(text):
    """
    Extract 4-dimensional structural features (V2.0 enhanced version).
    """
    if not isinstance(text, str):
        text = str(text)
    
    text_lower = text.lower()
    
    # --- 1. Comparative Keywords ---
    # English: or, than, vs, rank, compare, difference between
    # Chinese: 比 (A比B), 哪个 (which one is more...), 区别, 差异
    # Note: To avoid misclassifying 'difference engine', we check for 'difference between'
    
    comp_indicators = [
        ' or ', ' than ', ' vs ', ' vs. ', ' rank ', ' compare ', 
        ' better ', ' worse ', ' more ', ' less ',
        ' difference between ', # Key fix: only match "difference between"
        ' 比 ', ' 哪个 ', ' 区别 ', ' 差异 ' # Chinese support
    ]
    
    comp_score = 0
    for k in comp_indicators:
        if k in text_lower:
            comp_score += 1
            
    # Special patch: if "compare" appears without "to/with/between", it might also be an imperative comparison
    if text_lower.startswith("compare ") or text_lower.startswith("rank "):
        comp_score += 1

    # --- 2. Relational / Clause Keywords ---
    # English: that, which, who, whose, where (in clauses)
    # Chinese: 的 (in multi-level nesting), 所在 (located), 位于 (located)
    rel_indicators = [
        ' that ', ' which ', ' who ', ' whose ', 
        ' where ', # where is ... located (Retrieval) vs country where ... (Relational) - hard to distinguish, keep for now
        ' 所在 ', ' 位于 '
    ]
    
    clause_score = 0
    for k in rel_indicators:
        if k in text_lower:
            clause_score += 1
            
    # --- 3. 'of' depth (hierarchy depth) ---
    # English 'of', Chinese '的'
    # Note: Chinese '的' appears very frequently, only as an auxiliary feature
    of_count = text_lower.count(' of ') + text_lower.count('的') * 0.5 
    
    # --- 4. Length (simple normalization) ---
    length = len(text.split()) / 20.0
    
    return [float(comp_score), float(clause_score), float(of_count), length]
