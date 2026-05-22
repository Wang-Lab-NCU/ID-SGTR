import os
import sys
import pandas as pd
from typing import Tuple, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 假设 parallel_llm_processor 也定义在 utils 或其他自定义模块中
from utils import get_chat_model
from helper import parallel_llm_processor


if __name__ == "__main__":
    # 路径配置
    DATA_ROOT = r"D:\Code\jupyter\knowledge_graph\data_output\dataset\musique\ds1000"    
    QA_FILE = os.path.join(DATA_ROOT, "qa.csv")
    
    # 1. 初始化模型
    print("🚀 正在初始化 LLM 模型...")
    llm = get_chat_model(task_type="kg_query") 
    if llm is None:
        print("❌ 模型初始化失败，请检查环境变量配置。")
        sys.exit(1)

    # 2. 加载数据集
    # 注意：qa.csv 是用竖线 '|' 分隔的，所以必须指定 sep='|'
    if not os.path.exists(QA_FILE):
        print(f"❌ 找不到文件: {QA_FILE}")
        sys.exit(1)
        
    print(f"📂 正在读取数据集: {QA_FILE}")
    target_data = pd.read_csv(QA_FILE, sep='|')
    print(f"\n📝 开始并发处理 {len(target_data)} 条查询...")

    # 3. 定义单条处理逻辑的 Wrapper
    def process_query_wrapper(i: int, row: pd.Series) -> Tuple[int, Any]:
        """
        处理单行 QA 数据的核心函数，适配多线程执行器。
        """
        q = row['question']
        gold = row['answer']
        ctx = row['context_id']
        
        # 构建基线 Prompt（由于是 baseline，这里不借助外部检索，直接询问）
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
        # 调用 LLM，兼容不同的返回格式
        response = llm.invoke(prompt) if hasattr(llm, 'invoke') else llm(prompt)
        pred_answer = response.content if hasattr(response, 'content') else str(response)
        pred_answer = pred_answer.strip()
        strategy = "baseline_direct_query"
            
            
        # 返回行索引 i 和结果字典，方便外部还原顺序
        return i, {
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "strategy": strategy,
            "context_id": ctx
        }

    # 4. 启动多线程并发执行
    # 注意：请确保 parallel_llm_processor 会根据返回的 `i` 自动将结果排序并返回列表
    processed_results = parallel_llm_processor(
        dataframe=target_data,
        processing_func=process_query_wrapper,
        start_message="启动多线程推理...",
        max_workers=10,        # 建议并发数：Gemini/SiliconFlow 免费级可设为 3-5，付费级可拉高到 10-20
        max_retries=6,        # 发生 API Rate Limit 时的最大重试次数
        initial_delay=2       # 初始延迟
    )
    
    # 5. 打印预览并保存（可选）
    print("\n✅ 并发处理完成！")
    # 如果需要将结果转换为 DataFrame 并保存，可以取消下方代码注释：
    processed_results.sort(key=lambda x: x[0])
    final_data = [item[1] for item in processed_results]
    output_path = os.path.join(current_dir, "baseline_results.csv")
    pd.DataFrame(final_data).to_csv(output_path, index=False, sep="|")
    print(f"💾 结果已保存至: {output_path}")