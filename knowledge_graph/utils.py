# from langchain_community.embeddings import ZhipuAIEmbeddings
# from langchain_community.chat_models import ChatZhipuAI
# from py2neo import Graph
# from langchain_openai import ChatOpenAI

# import os
# from dotenv import load_dotenv
# load_dotenv()

# def get_embeddings_model(dimensions = os.getenv('dimensions')):
#     model_map = {
#         'zhipuai': ZhipuAIEmbeddings(
#             api_key=os.getenv('ZHIPUAI_API_KEY'),
#             model=os.getenv('ZHIPUAI_EMBEDDINGS_MODEL'),
#             dimensions = dimensions
#         )
#     }
#     return model_map.get(os.getenv('EMBEDDINGS_MODEL'))

# def get_llm_model(api_key=os.getenv('ZHIPUAI_API_KEY'),model=os.getenv('ZHIPUAI_LLM_MODEL')):
#     model_map = {
#         'zhipuai': ChatZhipuAI(
#             api_key=api_key,
#             temperature=os.getenv('ZHIPUAI_TEMPERATURE'),
#             model=model,
#             max_tokens=os.getenv('MAX_TOKENS'),
#             top_p=(os.getenv('ZHIPUAI_TOP_P')),
#             Verbose = os.getenv('VERBOSE'),
#         )
#     }
#     return model_map.get(os.getenv('LLM_MODEL'))
# def get_chat_model(task_type="extraction"):
#     """
#     根据任务类型返回不同参数配置的模型
#     task_type: 
#       - "extraction": 实体/关系抽取 (严谨)
#       - "reasoning":  指代消解 (逻辑)
#       - "kg_query":   图谱问答/ID-SGTR推理 (严格遵循上下文)
#     """
#     provider = os.getenv('CHAT_MODEL_PROVIDER', 'siliconflow')
    
#     base_kwargs = {
#         "model": os.getenv('SILICONFLOW_MODEL'),
#         "openai_api_key": os.getenv('SILICONFLOW_API_KEY'),
#         "openai_api_base": os.getenv('SILICONFLOW_BASE_URL'),
#         "max_tokens": 4096,
#     }

#     config_map = {
#         # 场景1：实体抽取 (保持不变)
#         "extraction": {
#             "temperature": 0.01,
#             "extra_body": {
#                 "top_p": 0.05,
#                 "top_k": 10,
#                 "frequency_penalty": 0.0, 
#             }
#         },
        
#         # 场景2：指代消解 (保持不变)
#         "reasoning": {
#             "temperature": 0.1,
#             "extra_body": {
#                 "top_p": 0.1,
#                 "top_k": 20,
#                 "enable_thinking": False, # 如果模型支持思维链
#                 "thinking_budget": 1024
#             }
#         },

#         # --- [修改点] 场景3：ID-SGTR 知识图谱推理问答 ---
#         "kg_query": {
#             # 极低温度：确保严格基于 Evidence，不产生幻觉
#             "temperature": 0.01, 
#             "extra_body": {
#                 # 极低 Top_P：锁定事实，减少发散
#                 "top_p": 0.05,
#                 # Top_K：限制候选词，防止跑题
#                 "top_k": 20,
#                 # 惩罚为0：允许模型在 Reasoning 步骤中重复引用原文实体名
#                 "frequency_penalty": 0.0, 
#                 # 存在惩罚为0：不要强制模型转换话题，专注当前 Query
#                 "presence_penalty": 0.0,
                
#                 # [可选] 如果使用的是 DeepSeek-R1 等推理模型，建议开启 thinking
#                 # ID-SGTR 框架本身就是一种思维链，模型内置的 thinking 能辅助这一过程
#                 "enable_thinking": False 
#             }
#         }
#     }

#     # 获取配置，默认 fallback 到 extraction
#     task_config = config_map.get(task_type, config_map["extraction"])
#     final_kwargs = {**base_kwargs, **task_config}

#     if provider == 'siliconflow':
#         return ChatOpenAI(**final_kwargs)
    
#     return None


# def extract_structural_features(text):
#     """
#     提取 4 维结构特征 (V2.0 增强版)
#     """
#     if not isinstance(text, str):
#         text = str(text)
    
#     text_lower = text.lower()
    
#     # --- 1. Comparative Keywords (比较类) ---
#     # 英文: or, than, vs, rank, compare, difference between
#     # 中文: 比 (A比B), 哪个 (哪个更..), 区别, 差异
#     # 注意: 为了避免 'difference engine' 误判，我们检查 'difference between'
    
#     comp_indicators = [
#         ' or ', ' than ', ' vs ', ' vs. ', ' rank ', ' compare ', 
#         ' better ', ' worse ', ' more ', ' less ',
#         ' difference between ', # 关键修复：只匹配 "difference between"
#         ' 比 ', ' 哪个 ', ' 区别 ', ' 差异 ' # 中文支持
#     ]
    
#     comp_score = 0
#     for k in comp_indicators:
#         if k in text_lower:
#             comp_score += 1
            
#     # 特殊补丁：如果出现了 "compare" 但没有 "to/with/between"，可能也是命令式比较
#     if text_lower.startswith("compare ") or text_lower.startswith("rank "):
#         comp_score += 1

#     # --- 2. Relational / Clause Keywords (关系类) ---
#     # 英文: that, which, who, whose, where (在从句中)
#     # 中文: 的 (多层嵌套时), 所在, 位于
#     rel_indicators = [
#         ' that ', ' which ', ' who ', ' whose ', 
#         ' where ', # where is ... located (Retrieval) vs country where ... (Relational) 很难分，暂时保留
#         ' 所在 ', ' 位于 '
#     ]
    
#     clause_score = 0
#     for k in rel_indicators:
#         if k in text_lower:
#             clause_score += 1
            
#     # --- 3. 'of' depth (层级深度) ---
#     # 英文 'of'，中文 '的'
#     # 注意：中文 '的' 出现频率极高，只作为辅助特征
#     of_count = text_lower.count(' of ') + text_lower.count('的') * 0.5 
    
#     # --- 4. Length (简单归一化) ---
#     length = len(text.split()) / 20.0
    
#     return [float(comp_score), float(clause_score), float(of_count), length]