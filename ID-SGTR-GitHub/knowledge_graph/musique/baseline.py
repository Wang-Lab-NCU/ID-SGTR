import os
import sys
import pandas as pd
from typing import Tuple, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Assume parallel_llm_processor is also defined in utils or other custom modules
from utils import get_chat_model
from helper import parallel_llm_processor


if __name__ == "__main__":
    # Path configuration
    DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\musique\ds1000"    
    QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
    
    # 1. Initialize the model
    print("🚀 Initializing LLM model...")
    llm = get_chat_model(task_type="kg_query") 
    if llm is None:
        print("❌ Model initialization failed. Please check your environment variable configuration.")
        sys.exit(1)

    # 2. Load dataset
    # Note: qa.csv is separated by '|', so we must specify sep='|'
    if not os.path.exists(QA_FILE):
        print(f"❌ File not found: {QA_FILE}")
        sys.exit(1)
        
    print(f"📂 Reading dataset: {QA_FILE}")
    target_data = pd.read_csv(QA_FILE, sep='|')
    print(f"\n📝 Starting concurrent processing of {len(target_data)} queries...")

    # 3. Define the wrapper for single-item processing logic
    def process_query_wrapper(i: int, row: pd.Series) -> Tuple[int, Any]:
        """
        Core function to process a single QA row, adapted for the multi-threaded executor.
        """
        q = row['question']
        gold = row['answer']
        ctx = row['context_id']
        
        # Build the baseline prompt (since this is baseline, no external retrieval, ask directly)
        prompt = f"""
        Please answer the following musique dataset question as briefly and accurately as possible.
        ### User Query
        "{q}" 
        ### Output Format
        Scenario A. If Answerable:
        `Final Answer: [Clean Entity Name / Yes / No / data / etc.]`(Precise and Concise)
        
        ### ✅ POSITIVE INSTRUCTIONS
        - **ALWAYS** keep output minimal - just the required lines with no explanations
        ### ⛔ OUTPUT RESTRICTIONS
        - **NO** sentences or paragraphs
        - **NO** explanations or reasoning
        """       
        # Invoke LLM, compatible with different return formats
        response = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm(prompt)
        pred_answer = response.content if hasattr(response, 'content') else str(response)
        pred_answer = pred_answer.strip()
        strategy = "baseline_direct_query"
            
        # Return row index i and result dictionary, so that the order can be restored later
        return i, {
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "strategy": strategy,
            "context_id": ctx
        }

    # 4. Start multi-threaded concurrent execution
    # Note: ensure that parallel_llm_processor automatically sorts the results by returned `i` and returns a list
    processed_results = parallel_llm_processor(
        dataframe=target_data,
        processing_func=process_query_wrapper,
        start_message="Starting multi-threaded inference...",
        max_workers=10,        # Suggested concurrency: for free tier of Gemini/SiliconFlow, set to 3-5; for paid tier, can increase to 10-20
        max_retries=6,        # Maximum retries when encountering API rate limits
        initial_delay=2       # Initial delay
    )
    
    # 5. Print preview and save (optional)
    print("\n✅ Concurrent processing completed!")
    # If you want to convert the results to a DataFrame and save, uncomment the following lines:
    processed_results.sort(key=lambda x: x[0])
    final_data = [item[1] for item in processed_results]
    output_path = os.path.join(current_dir, "baseline_results.csv")
    pd.DataFrame(final_data).to_csv(output_path, index=False, sep="|")
    print(f"💾 Results saved to: {output_path}")