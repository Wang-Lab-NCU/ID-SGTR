import sys
import os
import json
import re
from typing import Any, List, Dict, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from utils import get_llm_model, get_chat_model
from prompt import * 
from helper import safe_json_parse

# Initialize models
llm_model = get_llm_model(model='glm-4-flash-250414')
llm_extract = get_chat_model(task_type="extraction")
llm_resolve = get_chat_model(task_type="reasoning")

def extractConcepts(prompt: str, metadata={}, model=None):
    """
    Extract concept nodes. Remove internal catch and allow retry mechanism to intervene.
    """
    messages = [
        SystemMessage(content=EXTRACT_KNOWLEDGE_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    active_model = model if model is not None else llm_extract
    response = active_model.invoke(messages)
    response_content = response.content if hasattr(response, 'content') else str(response)
    
    # Parse JSON
    concepts_list = safe_json_parse(response_content, default=None)
    
    # If parsing returns None, it means the LLM output format is invalid; manually raise an exception to trigger retry
    if concepts_list is None:
        raise ValueError(f"LLM returned invalid JSON format: {response_content[:100]}...")
        
    return concepts_list

def resolve_coreferences(text: str, model=None) -> str:
    """
    Perform coreference resolution. No longer catch exceptions; handled by parallel_llm_processor.
    """
    messages = [
        SystemMessage(content=COREFERENCE_RESOLUTION_SYS_PROMPT),
        HumanMessage(content=f"Perform coreference resolution on the following text:\n\n{text}")
    ]
    
    active_model = model if model is not None else llm_resolve
    response = active_model.invoke(messages)
    return response.content.strip()

def graphPrompt(input_text: str, entity_map: dict, metadata={}, model=None):
    """
    Graph relation extraction. Ensure exceptions are raised on failure.
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
        raise ValueError(f"Relation extraction JSON parsing failed: {response_content[:100]}...")
    
    if not isinstance(result, list):
        result = [result]
        
    # Merge metadata
    result = [dict(item, **metadata) for item in result]
    return result

def classify_query_type(query_string: str, model=None) -> str:
    """
    Intent classification.
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
        # Even if the model runs, if the label is incorrect, treat as failure to allow retry
        raise ValueError(f"LLM returned invalid classification label: {response_text}")
        
    return response_text
