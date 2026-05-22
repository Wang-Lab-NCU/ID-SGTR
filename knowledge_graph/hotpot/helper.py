import time
import random
import re
from tqdm import tqdm
import ast
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from difflib import SequenceMatcher
from typing import Callable, Any, Dict, List, Tuple
# 确保 LangChain 相关的类型也从正确的位置导入，此处假设它们与 kg_query_engine.py 共享导入
from langchain_core.messages import BaseMessage 


# ==========================================
# 并行执行器
# ==========================================
def parallel_llm_processor(
    dataframe: pd.DataFrame,
    processing_func: Callable[[int, pd.Series], Tuple[Any, Any]],
    start_message: str,
    max_workers: int = 15,
    max_retries: int = 6,
    initial_delay: int = 2,
    task_timeout: int = 100 # 单个任务的强制超时时间
) -> List[Tuple[Any, Any]]:
    """
    优化后的并行执行器：
    1. 使用 tqdm 显示进度。
    2. 解决 'with' 语句导致的最后几个线程卡死问题（通过 wait=False）。
    """
    print(start_message)
    results_list: List[Tuple[Any, Any]] = []
    
    # 内部重试逻辑
    def _run_with_retry(i: int, row: pd.Series) -> Tuple[int, Any, Any]:
        identifier = row.get('chunk_id', i)
        for attempt in range(max_retries):
            try:
                # 警告：这里调用的 processing_func 内部最好也有 requests 的 timeout 设置
                current_identifier, result = processing_func(i, row)
                return (i, current_identifier, result)
            except Exception as e:
                error_msg = str(e)
                # 简单的错误分类日志
                if attempt + 1 == max_retries:
                    # 最后一次才打印 ERROR，避免中间的重试刷屏
                    print(f"\n🛑 ID {identifier} 最终失败: {error_msg[:100]}...")
                    return (i, identifier, None)
                
                # 指数退避
                delay = initial_delay * (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(delay)
        return (i, identifier, None)

    # --- 核心修改：手动创建 executor，不使用 with 上下文 ---
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}

    try:
        # 1. 提交所有任务
        print(f"🚀 正在提交 {len(dataframe)} 个任务到线程池...")
        for i, row in dataframe.iterrows():
            future = executor.submit(_run_with_retry, i, row)
            futures[future] = row.get('chunk_id', i)

        # 2. 使用 tqdm 处理进度
        print(f"⏳ 开始并行处理...")
        # total=len(futures) 让进度条知道总数
        with tqdm(total=len(futures), unit="chunk") as pbar:
            for future in as_completed(futures):
                identifier = futures[future]
                try:
                    # 等待结果，设置超时防止主线程被单个任务永久阻塞
                    original_index, current_id, result = future.result(timeout=task_timeout)
                    
                    if result is not None:
                        results_list.append((current_id, result))
                    
                except TimeoutError:
                    print(f"\n⏰ 任务 ID {identifier} 获取结果超时（线程可能卡死），跳过。")
                    # 这里不需要 append，直接跳过即可
                except Exception as e:
                    print(f"\n🔴 任务 ID {identifier} 抛出未知异常: {e}")
                finally:
                    # 无论成功失败，进度条都 +1
                    pbar.update(1)

    finally:
        # --- 核心修复 ---
        # wait=False 表示：不等待那些还在运行（卡死）的线程，直接关闭入口，主程序继续向下执行
        # Python 3.9+ 可以加 cancel_futures=True
        print("\n🧹 正在清理线程池（丢弃卡死的线程）...")
        executor.shutdown(wait=False)
        print(f"✅ 处理结束。成功获取结果: {len(results_list)}/{len(dataframe)}")

    return results_list

def get_unique_chunks(chunk_ids):
    if pd.isna(chunk_ids) or not chunk_ids:
        return set()
    return {c.strip() for c in str(chunk_ids).split(',') if c.strip().isdigit()}


def safe_json_parse(json_string: str, default: List = None) -> Any:
    """
    尝试安全地解析 JSON 字符串，解决因不可见字符、截断、缺失分隔符或单引号导致的解析失败。
    """
    if default is None:
        default = []
    
    if not json_string or not isinstance(json_string, str):
        return default

    # 1. 终极清理：删除 BOM、不可见字符，剥离 Markdown
    cleaned_string = re.sub(r'[\ufeff\u200b\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', json_string).strip()
    
    # 移除 Markdown 代码块 (```json ... ```)
    # 优化正则：非贪婪匹配开头，直到遇到换行或结尾
    cleaned_string = re.sub(r'^```(?:json)?\s*', '', cleaned_string, flags=re.IGNORECASE)
    cleaned_string = re.sub(r'\s*```$', '', cleaned_string)
    cleaned_string = cleaned_string.strip()

    # 2. 边界裁剪：确保字符串以 [ 或 { 开头，以 ] 或 } 结尾
    # 这一步是为了去除 LLM 在 JSON 前后的闲聊
    match = re.search(r'(\[|\{).*(\]|\})', cleaned_string, re.DOTALL)
    if match:
        cleaned_string = match.group(0)
    else:
        # 如果找不到成对的括号，尝试寻找任何可能的起始点（容错）
        start_match = re.search(r'[\{\[]', cleaned_string)
        if start_match:
            cleaned_string = cleaned_string[start_match.start():]
        else:
            return default

    # 3. 尝试标准解析 (JSON)
    try:
        # strict=False 允许字符串中包含控制字符（如换行符）
        result = json.loads(cleaned_string, strict=False)
        if isinstance(result, dict):
            return [result]
        return result
    except json.JSONDecodeError:
        pass # 继续尝试其他方法

    # 4. 尝试 Python 字面量解析 (处理单引号 JSON)
    # 很多 LLM 喜欢输出 Python 字典格式 {'key': 'value'} 而不是 {"key": "value"}
    try:
        result = ast.literal_eval(cleaned_string)
        if isinstance(result, (list, dict)):
            if isinstance(result, dict):
                return [result]
            return result
    except (ValueError, SyntaxError):
        pass # 继续尝试

    # 5. 鲁棒修复：缺失逗号修复
    # 常见错误：objects 之间没有逗号，如 }{ 或 ] [
    try:
        repaired_string = re.sub(r'\}\s*\{', '},{', cleaned_string)
        repaired_string = re.sub(r'\]\s*\[', '],[', repaired_string)
        result = json.loads(repaired_string, strict=False)
        if isinstance(result, dict):
            return [result]
        return result
    except json.JSONDecodeError:
        pass

    # 6. 终极兜底：正则暴力提取对象
    # 如果整体结构坏了（例如缺少闭合的 ]），尝试提取内部完整的 {...}
    print(f"⚠️ 整体解析失败，尝试逐个提取对象...")
    final_records: List[Dict] = []
    
    # 正则解释：非贪婪匹配 { ... }
    # 注意：如果 value 里面包含 }，这个简单正则可能会截断。
    # 对于简单的图谱关系（扁平结构），这个正则通常足够。
    object_pattern = re.compile(r'\{[^{}]+\}', re.DOTALL) 
    # 如果你需要处理嵌套结构（value里有{}），需要更复杂的逻辑，但当前场景不需要。
    
    object_strings = object_pattern.findall(cleaned_string)

    for obj_str in object_strings:
        try:
            # 尝试解析单个对象
            # 同样尝试 json 和 ast 两种方式
            try:
                obj = json.loads(obj_str, strict=False)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(obj_str)
                except:
                    continue # 放弃这个对象
            
            # 💥 关键修改：只要是字典就保留，不再检查 specific key 如 'entity'
            if isinstance(obj, dict):
                final_records.append(obj)
                
        except Exception:
            continue
    
    if final_records:
        print(f"✅ 鲁棒提取成功恢复 {len(final_records)} 条记录。")
        return final_records
    
    # 彻底失败
    print(f"❌ 无法解析 JSON。原始内容片段: {json_string[:100]}...")
    return default
    

def apply_genealogical_penalty(entities: list, dist_matrix: np.ndarray, penalty_value: float = 10.0) -> np.ndarray:
    """
    对距离矩阵应用'谱系冲突惩罚'。
    如果检测到两个实体拥有显式冲突的代际后缀 (如 Sr. vs Jr., II vs III)，
    则强制将其距离设置为 penalty_value (拉大距离，阻止聚类)。
    
    Args:
        entities: 实体名称列表 (与 dist_matrix 的行列对应)
        dist_matrix: 预计算的余弦距离矩阵 (N x N)
        penalty_value: 惩罚值，通常设为 10.0 或更大，确保超过任何 eps
    
    Returns:
        修改后的 dist_matrix (In-place modification)
    """
    
    # 1. 定义后缀映射表 (将不同写法映射到统一的代际 ID)
    # ID 相同表示同一代 (不冲突)，ID 不同表示显式冲突
    SUFFIX_MAP = {
        # 儿子 / 二世
        'jr': 1, 'junior': 1, 'ii': 1, '2nd': 1,
        # 父亲 / 一世 / 老
        'sr': 2, 'senior': 2, 'i': 2, '1st': 2,
        # 三世
        'iii': 3, '3rd': 3,
        # 四世
        'iv': 4, '4th': 4,
        # 五世
        'v': 5, '5th': 5
    }

    # 2. 编译正则：匹配字符串结尾的完整单词，忽略大小写和点号
    # 例如匹配: "Ed Wood Jr", "King George V", "John Smith, 3rd"
    # \b 确保单词边界，\.? 允许有点或没点
    pattern_str = r'\b(' + '|'.join(SUFFIX_MAP.keys()) + r')\.?$'
    regex = re.compile(pattern_str, re.IGNORECASE)

    n = len(entities)
    
    # 预先提取每个实体的代际 ID (如果没有后缀则为 None)
    generations = []
    for name in entities:
        # 清理多余空格，转小写进行匹配
        clean_name = name.strip().lower()
        match = regex.search(clean_name)
        if match:
            # 提取匹配到的后缀 (去掉可能存在的点)
            suffix = match.group(1).replace('.', '')
            gen_id = SUFFIX_MAP.get(suffix)
            generations.append(gen_id)
        else:
            generations.append(None)

    # 3. 遍历矩阵应用惩罚
    # 只有当两个人都拥有后缀，且后缀 ID 不一样时，才视为冲突
    # 注意：一个人有后缀(Jr)，另一个人没后缀(无)，通常**不**视为冲突 (可能是简称)
    for i in range(n):
        gen_i = generations[i]
        if gen_i is None: continue # 如果 i 没有后缀，跳过
        
        for j in range(i + 1, n):
            gen_j = generations[j]
            
            # 只有两人都有后缀，才比较
            if gen_j is not None:
                if gen_i != gen_j:
                    # 💥 发现冲突 (如 Jr vs Sr)，应用惩罚
                    dist_matrix[i, j] = penalty_value
                    dist_matrix[j, i] = penalty_value
                    
                    # (可选) 打印调试信息，方便看谁被强制分开了
                    # print(f"🔨 强制分离: '{entities[i]}' vs '{entities[j]}'")

    return dist_matrix

def clean_entity(entity_name):
    """
    清理实体名称，移除头衔、常见后缀、非字母数字字符和所有空格。
    """
    if pd.isna(entity_name):
        return ""
    entity_name = str(entity_name).strip()
    
    # 移除常见头衔和后缀
    entity_name = re.sub(r'\s+(Jr|Sr|Dr|Prof|King|Sultan|President|Queen|Princess)\b\.?', '', entity_name, flags=re.IGNORECASE).strip()
    
    # 移除所有非字母和空格的字符 (如逗号, 引号等)
    entity_name = re.sub(r'[^a-zA-Z0-9\s]', '', entity_name, flags=re.IGNORECASE) 
    
    # 将多个空格替换为单个空格
    entity_name = re.sub(r'\s+', ' ', entity_name).strip()
    
    # =======================================================
    # 🚨 关键修正：移除所有空格，用于子串匹配
    # =======================================================
    cleaned_no_space = entity_name.replace(' ', '')
    
    return cleaned_no_space.lower() # 统一转小写，确保大小写不敏感
def post_process_person_entities(df_standardization_map):
    """
    对 Person 类别进行后处理，基于字符串包含关系合并全称/简称。
    
    Args:
        df_standardization_map (pd.DataFrame): 包含 Original_Entity, Standard_Entity, context_id 的映射表。
        
    Returns:
        pd.DataFrame: 更新后的映射表。
    """
    print("\n⏳ 开始后处理 'Person' 类型的全称/简称...")
    
    # 仅处理 Person 类型实体
    person_map = df_standardization_map[df_standardization_map['category'] == 'Person'].copy()
    
    # 存储需要更新的映射 (Original_Entity -> New_Standard_Entity)
    updates = {}
    
    # 1. 在当前映射表中添加一个干净的实体名称列
    person_map['Clean_Entity'] = person_map['Original_Entity'].apply(clean_entity)
    
    # 2. 迭代每个 context_id
    for context_id, group in person_map.groupby('context_id'):
        unique_entities = group['Original_Entity'].unique()
        
        # 将当前 context 下的 (Entity, Clean_Entity, Current_Standard) 映射为字典
        entity_info = group.set_index('Original_Entity')[['Clean_Entity', 'Standard_Entity']].to_dict('index')
        
        # 3. 检查所有实体对
        for i in range(len(unique_entities)):
            for j in range(i + 1, len(unique_entities)):
                entity_a = unique_entities[i]
                entity_b = unique_entities[j]

                clean_a = entity_info[entity_a]['Clean_Entity']
                clean_b = entity_info[entity_b]['Clean_Entity']
                
                std_a = entity_info[entity_a]['Standard_Entity']
                std_b = entity_info[entity_b]['Standard_Entity']
                
                # 如果它们已经被 DBSCAN 合并到同一个 Standard_Entity，则跳过
                if std_a == std_b:
                    continue

                # 启发式规则：检查子字符串关系（假设已经过清理）
                is_sub_a = clean_a in clean_b and clean_a != clean_b
                is_sub_b = clean_b in clean_a and clean_b != clean_a

                if is_sub_a or is_sub_b:
                    
                    # 确定更长的实体作为新的 Standard_Entity
                    if len(entity_a) > len(entity_b):
                        new_standard = entity_a
                        shorter_entity = entity_b
                    else:
                        new_standard = entity_b
                        shorter_entity = entity_a
                        
                    # 4. 更新映射
                    # 目标：将所有涉及的实体（包括它们原来的 Standard_Entity）统一到新的 Standard_Entity
                    
                    # 确保新的 Standard Entity 是被选中的那个
                    updates[(context_id, shorter_entity)] = new_standard
                    
                    # 此外，还需要将当前 Std_A 和 Std_B 所代表的整个簇（如果它们是不同的）都指向新的 Standard
                    # 由于我们只在 Original_Entity 级别操作，这里我们只更新 Original_Entity 到 New_Standard
                    updates[(context_id, entity_a)] = new_standard
                    updates[(context_id, entity_b)] = new_standard

    # 5. 应用更新到 person_map
    for (cid, orig_ent), new_std in updates.items():
        # 在 group 级别查找并更新
        # 注意：这里需要精确匹配 context_id 和 Original_Entity
        mask = (person_map['context_id'] == cid) & (person_map['Original_Entity'] == orig_ent)
        person_map.loc[mask, 'Standard_Entity'] = new_std

    # 6. 将更新后的 Person 映射合并回原始映射表
    
    # 从原始映射中移除 Person 类别
    df_without_person = df_standardization_map[df_standardization_map['category'] != 'Person']
    
    # 合并
    df_updated_map = pd.concat([df_without_person, person_map], ignore_index=True)
    print(f"✅ 后处理完成。共更新了 {len(updates)} 个映射关系。")
    return df_updated_map
