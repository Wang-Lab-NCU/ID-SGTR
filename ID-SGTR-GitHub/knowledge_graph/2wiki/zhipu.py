import sys
import os
import json
import re
from typing import Any, List, Dict, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from utils import get_llm_model,get_chat_model
from prompt import * 
from helper import safe_json_parse

# 初始化模型
llm_model = get_llm_model(model='glm-4-flash-250414')
llm_extract = get_chat_model(task_type="extraction")
llm_resolve = get_chat_model(task_type="reasoning")

def extractConcepts(prompt: str, metadata={}, model=None):
    """
    抽取概念节点。去掉内部捕获，允许重试机制介入。
    """
    messages = [
        SystemMessage(content=EXTRACT_KNOWLEDGE_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    active_model = model if model is not None else llm_extract
    response = active_model.invoke(messages)
    response_content = response.content if hasattr(response, 'content') else str(response)
    
    # 解析 JSON
    concepts_list = safe_json_parse(response_content, default=None)
    
    # 如果解析结果为 None，说明 LLM 输出格式不对，手动抛出异常以触发重试
    if concepts_list is None:
        raise ValueError(f"LLM 返回了无效的 JSON 格式: {response_content[:100]}...")
        
    return concepts_list

def resolve_coreferences(text: str, model=None) -> str:
    """
    执行指代消解。不再捕获异常，由 parallel_llm_processor 处理。
    """
    messages = [
        SystemMessage(content=COREFERENCE_RESOLUTION_SYS_PROMPT),
        HumanMessage(content=f"请对以下文本进行指代消解:\n\n{text}")
    ]
    
    active_model = model if model is not None else llm_resolve
    response = active_model.invoke(messages)
    return response.content.strip()

def graphPrompt(input_text: str, entity_map: dict, metadata={}, model=None):
    """
    图谱关系抽取。确保失败时抛出异常。
    """
    map_str = "\n".join([f'- "{raw}": Target Node "{std}"' for raw, std in entity_map.items()])
    USER_PROMPT = (
        f"Context:\n```\n{input_text}\n```\n\n"
        f"Entity Mapping (Text Mention -> Target Standard Node):\n"
        f"{map_str}\n\n"
        f"Task: Identify relations between the entities strictly based on the mapping above.\n"
        f"Output:"
    )
    
    messages = [
        SystemMessage(content=GUIDED_GRAPH_RELATIONS_SYS_PROMPT), 
        HumanMessage(content=USER_PROMPT)
    ]
    
    active_model = model if model is not None else llm_extract
    response = active_model.invoke(messages)
    response_content = response.content if hasattr(response, 'content') else str(response)

    result = safe_json_parse(response_content, default=None)
    if result is None:
        raise ValueError(f"关系抽取 JSON 解析失败: {response_content[:100]}...")
    
    if not isinstance(result, list):
        result = [result]
        
    # 合并 metadata
    result = [dict(item, **metadata) for item in result]
    return result

def classify_query_type(query_string: str, model=None) -> str:
    """
    意图分类。
    """
    system_prompt = (
        "You are an expert classifier for Question-Answering systems. "
        "Your MUST respond with ONLY one: 'Factual', 'Conceptual', or 'Relational'."
    )
    user_prompt = f"Now, classify the following query:\nQuery: \"{query_string}\"\nCategory: "
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    active_model = model if model is not None else llm_extract
    response = active_model.invoke(messages)
    response_text = response.content.strip()
    
    valid_labels = {'Factual', 'Conceptual', 'Relational'}
    if response_text not in valid_labels:
        # 即使模型能跑通，如果标签不对，也视为失败以便重试
        raise ValueError(f"LLM 分类标签无效: {response_text}")
        
    return response_text