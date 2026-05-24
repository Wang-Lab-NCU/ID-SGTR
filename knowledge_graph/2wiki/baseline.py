import os
import sys
import pandas as pd
from typing import Tuple, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_chat_model
from helper import parallel_llm_processor


if __name__ == "__main__":
    DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\2wiki\ds1000"    
    QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
    
    # 1. Initialize the model
    print("🚀 The LLM model is currently in the initialization process.")
    llm = get_chat_model(task_type="kg_query") 
    if llm is None:
        print("❌ Model initialization failed. Please check the configuration of environment variables.")
        sys.exit(1)

    # 2. Load dataset
    if not os.path.exists(QA_FILE):
        print(f"❌ File not found: {QA_FILE}")
        sys.exit(1)
        
    print(f"📂 Loading dataset: {QA_FILE}")
    target_data = pd.read_csv(QA_FILE, sep='|')
    print(f"\n📝 Starting concurrent processing of {len(target_data)} queries...")

    # 3. Define the wrapper for single-query processing
    def process_query_wrapper(i: int, row: pd.Series) -> Tuple[int, Any]:
        """
        The core function for handling single-line QA data, adapted for multi-threaded executor.        """
        q = row['question']
        gold = row['answer']
        ctx = row['context_id']
        
        # Establish the baseline prompt (Since it is a baseline, here we do not rely on external search but directly ask)
        prompt = f"""
        Please answer the following hotpot dataset question as briefly and accurately as possible.
        ### User Query
        "{q}" 
        ### Output Format
        Scenario A. If Answerable(Answer Found):
        `Final Answer: [Clean Entity Name / Yes / No / data / etc.]`(Precise and Concise)
        
        ### ✅ POSITIVE INSTRUCTIONS
        - **ALWAYS** keep output minimal - just the required lines with no explanations
        ### ⛔ OUTPUT RESTRICTIONS
        - **NO** sentences or paragraphs
        - **NO** explanations or reasoning
        """       
        try:
            # Call the LLM, compatible with different return formats
            response = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm(prompt)
            pred_answer = response.content if hasattr(response, 'content') else str(response)
            pred_answer = pred_answer.strip()
            strategy = "baseline_direct_query"
            
        except Exception as e:
            # If the operation fails after 6 retries, record the error.
            print(f"\n[Error] Index {i} processing failed: {str(e)}")
            pred_answer = f"ERROR: {str(e)}"
            strategy = "error"
            
        # Return the row index i and the result dictionary
        return i, {
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "strategy": strategy,
            "context_id": ctx
        }

    # 4. Start multi-threaded concurrent execution
    # 注意：请确保 parallel_llm_processor 会根据返回的 `i` 自动将结果排序并返回列表
    processed_results = parallel_llm_processor(
        dataframe=target_data,
        processing_func=process_query_wrapper,
        start_message="Starting multi-threaded inference...",
        max_workers=10,        # Suggested concurrent number: 3-5 for free tier, up to 10-20 for paid tier
        max_retries=6,        # Maximum retry attempts when API Rate Limit is encountered
        initial_delay=2       # Initial delay
    )
    
    # 5. Print preview and save (optional)
    print("\n✅ Concurrent processing completed!")
    # 如果需要将结果转换为 DataFrame 并保存，可以取消下方代码注释：If you need to convert the result into a DataFrame and save it, you can remove the comments below:
    processed_results.sort(key=lambda x: x[0])
    final_data = [item[1] for item in processed_results]
    output_path = os.path.join(current_dir, "baseline_results.csv")
    pd.DataFrame(final_data).to_csv(output_path, index=False, sep="|")
    print(f"💾 Results saved to: {output_path}")