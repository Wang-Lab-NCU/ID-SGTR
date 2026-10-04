# ID-SGTR

[English](README.md) | [简体中文](README.zh-CN.md)

ID-SGTR 是面向多跳问答的混合图检索与证据构建系统，结合显式关系边、语境邻近边和原始文本。图路径指导证据选择，答案模型读取原文而非仅仅读取三元组。本仓库包含 HotpotQA、2WikiMultiHopQA、MuSiQue 的代码及固定问题子集；模型权重和已构建图数据单独提供。

## 两种推理设定

- **Retrieval Setting（全域检索）**：BGE-M3 全局召回文本块，BGE-Reranker-v2-M3 重排候选 context，锁定唯一 context，再在其中进行图搜索和证据选择。
- **Reasoning Setting（给定语境）**：基准直接提供局部 context，跳过全局 context 选择；后续重排、图搜索、证据构建与回答流程相同。

两种设定通过不同入口选择：Retrieval 使用 `query_global.py`，Reasoning 使用 `query_local.py`，不是在同一个脚本中修改一个开关。Retrieval 锁定的 context 构成后续搜索边界；Reasoning 则从开始就以基准提供的 context 为边界。

```text
Retrieval：全局文本块 → 候选 context → 重排 → 锁定唯一 context ┐
                                                             ├→ 实体种子 → 有界图路径
Reasoning：基准提供的局部 context ────────────────────────────┘
→ 局部文本块重排 → PCEF v2（B=3）→ 原文证据组装
→ 初始答案证据检查 → 必要时有界多跳搜索 → 终端综合
```

当前均衡配置的路径约束证据预算为 `B=3`。PCEF v2 从相关性 Top-B 出发，保护 Top-1，并在相关性损失上限内搜索邻近证据集合；最多替换 `⌊B/2⌋` 个文本块，即 B=3 时最多 1 个、B=5 时最多 2 个。入选原文按路径信息组织。初始答案通过基于证据的置信检查才提前返回；否则执行有界多跳推理、构造状态感知查询并动态重排，必要时局部补充检索，最后使用 Terminal Recovery v2.1。均衡配置**关闭额外的 Hop Confidence Gate**，但保留原有中间答案返回逻辑。

Hop 阶段的原始问题缓存分数与动态查询分数的配置权重为 `0.35` 和 `0.45`（归一化后为 43.75% 和 56.25%）；独立的 Hop Bridge 权重为 `0`。

种子阶段最多考虑 20 个候选，最终保留最多 5 个；低置信度或通用枢纽实体风险较高时，可调用聊天模型复核。逐跳查询会使用已发现的实体与关系改变重排目标，**不是**每跳都拿不变的原始问题重新给同一批文本打分。初始 `confidence_grounded` 检查要求模型给出 HIGH 置信度，且非空答案能由当前证据窗口中引用或确定性推得的文本支持。它只控制**是否提前退出**，不替代后续检索和答案接受策略。Terminal Recovery v2.1 使用独立的证据预算，在最终综合前按文本内容去除重复段落。

## 环境与数据

建议 Python 3.11，执行 `pip install -r requirements.txt`。推理需要兼容 OpenAI API 的 Qwen3-8B 聊天服务、BGE-M3 向量服务，以及本地或 HTTP 形式的 BGE-Reranker-v2-M3。复制 `.env.example` 为 `.env` 并填写本地地址与密钥；不要提交实际 `.env`。

**本仓库不包含模型服务。** HTTP 配置需要下列接口；服务中的模型名称须与 `.env` 一致：

| 服务 | 示例地址 | 所需接口 |
| --- | --- | --- |
| 聊天模型 | `http://127.0.0.1:8001/v1` | 兼容 OpenAI 的 chat completions；服务名 `qwen3-8b` |
| 嵌入模型 | `http://127.0.0.1:30000/v1` | 兼容 OpenAI 的 embeddings；服务名 `BAAI/bge-m3` |
| 重排模型 | `http://127.0.0.1:30001` | `POST /rerank`，接收 `model`、`query`、`documents` 并返回对应分数 |

正式运行前，可在 PowerShell 检查服务：

```powershell
Invoke-RestMethod http://127.0.0.1:8001/v1/models
Invoke-RestMethod http://127.0.0.1:30000/v1/models
$body = @{ model = 'BAAI/bge-reranker-v2-m3'; query = 'capital of France'; documents = @('Paris is the capital of France.') } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:30001/rerank -Method Post -ContentType 'application/json' -Body $body
```

代码也支持 `ID_SGTR_RERANK_BACKEND=local`，此时 `ID_SGTR_RERANK_MODEL` 应指向本地模型目录。提供的均衡配置使用 HTTP 方式，以避免多个查询 worker 分别加载重排模型。

从 [Google Drive 上的 ID-SGTR 数据](https://drive.google.com/drive/folders/1L9U1EToW3R_VtNAW6VyGM4cSq65kg6gf?usp=drive_link) 下载输入数据和预构建图数据，在仓库根目录解压，保留 `knowledge_graph/data_input/` 与 `knowledge_graph/data_output/` 目录结构。查询入口需要：

| 数据集 | 预构建数据路径 |
| --- | --- |
| HotpotQA | `knowledge_graph/data_output/dataset/hotpot/ds1000_2/` |
| 2WikiMultiHopQA | `knowledge_graph/data_output/dataset/2wiki/ds1000/` |
| MuSiQue | `knowledge_graph/data_output/dataset/musique/ds1000/` |

每个目录需包含 `qa.csv`、`graph.csv`、`chunk.csv`、`contextual_proximity.csv`、`chunks_with_embeddings.parquet` 和 `concepts_merged_with_vectors.parquet`。意图分类器权重位于 `knowledge_graph/adapt/intent_classifier_struct.pth`。数据集和模型仍须遵守各自许可。

| 文件 | 用途 |
| --- | --- |
| `qa.csv` | 查询、答案与 context ID |
| `chunk.csv` | 原始证据文本及 chunk/context ID |
| `graph.csv` | 抽取出的显式关系边 |
| `contextual_proximity.csv` | 语境邻近关系 |
| `chunks_with_embeddings.parquet` | 文本块与标题的稠密向量 |
| `concepts_merged_with_vectors.parquet` | 规范化实体、别名和向量 |

`experiments/subsets/` 下的 `*_1000_gold.csv`、`*_200_gold.csv`、`*_blind800_gold.csv` 用于固定评测 query ID，**不能替代**图与原文数据。子集至少需要 `query_id`、`question`、`answer` 和 `context_id`；提供的 gold 子集还包含证据标注。比较不同方法时，不要混用数据版本或不同 query ID。

## 运行均衡配置

先在 Python 环境中安装依赖。下例使用作者本地的 `andelie` Conda 环境；其他机器可使用任意已安装依赖的环境。在 Windows PowerShell 中进入本仓库目录：

```powershell
conda activate andelie
pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，配置聊天、嵌入和重排器服务。
. .\scripts\pcef_v2_balanced.ps1
$env:ID_SGTR_RERANK_BACKEND = 'http'
$env:ID_SGTR_SAMPLE_SIZE = '3'
$env:ID_SGTR_SUBSET_FILE = 'experiments/subsets/2wiki_200_gold.csv'
$env:ID_SGTR_OUTPUT = 'results/2wiki-retrieval-smoke3.csv'
New-Item -ItemType Directory -Force results | Out-Null
python knowledge_graph/2wiki/query_global.py
```

这 3 条仅是端到端连通性检查，其 EM/F1 不是论文结果。检查通过后，运行固定的 1000 条：

```powershell
$env:ID_SGTR_SAMPLE_SIZE = '1000'
$env:ID_SGTR_SUBSET_FILE = 'experiments/subsets/2wiki_1000_gold.csv'
$env:ID_SGTR_OUTPUT = 'results/2wiki-retrieval-seed42.csv'
python knowledge_graph/2wiki/query_global.py
```

Reasoning Setting 改用 `python knowledge_graph/2wiki/query_local.py`，并改写输出文件名；其他数据集的脚本和子集路径均改用 `hotpot` 或 `musique`。设置 `ID_SGTR_SUBSET_FILE` 才能固定问题集合；不设置时入口读取 `qa.csv` 的前若干行。即使指定子集，`ID_SGTR_SAMPLE_SIZE` 仍会截取前 N 行。`ID_SGTR_OUTPUT` 的父目录必须存在。Linux 用户需在 shell 中导出 `scripts/pcef_v2_balanced.ps1` 列出的变量；该 PowerShell 脚本仅设置参数，不启动模型服务。

均衡配置使用 PCEF v2、保护 Top-1、相关性损失上限 `0.25`、结构权重 `0.35`、初始 `confidence_grounded` 提前退出开启、Hop Confidence Gate 关闭、常规证据预算 `3`、终端预算 `5`、seed `42`、温度 `0`、主推理 thinking 开启、最大输出 `2048` tokens、客户端最多 16 个 worker。Terminal Recovery v2.1 单独关闭 thinking，最多输出 `384` tokens。这些是实验配置，并非所有模块默认值；三种子复现实验需显式修改 seed。

| 参数 | 均衡版数值 | 含义 |
| --- | ---: | --- |
| `ID_SGTR_PCEF_VERSION` / `ID_SGTR_PATH_CONSTRAINED_SELECTION` | `v2` / `true` | 启用 PCEF v2 证据集合搜索 |
| `ID_SGTR_PCEF_PROTECT` | `1` | 保留 Top-B 中相关性最高的文本 |
| `ID_SGTR_PCEF_MAX_RELEVANCE_DROP` / `ID_SGTR_PCEF_STRUCTURE_WEIGHT` | `0.25` / `0.35` | 限制相关性损失、调节结构互补性 |
| `ID_SGTR_EVIDENCE_BUDGET` / `ID_SGTR_TERMINAL_EVIDENCE_BUDGET` | `3` / `5` | 常规证据窗口 / 终端综合窗口 |
| `ID_SGTR_STAGE0_POLICY` / `ID_SGTR_HOP_CONFIDENCE_GATE` | `confidence_grounded` / `false` | 初始提前退出检查；逐跳不施加额外硬门控 |
| `ID_SGTR_HOP_ORIGINAL_WEIGHT` / `ID_SGTR_HOP_DYNAMIC_WEIGHT` / `ID_SGTR_HOP_BRIDGE_WEIGHT` | `0.35` / `0.45` / `0` | 缓存分数、动态查询分数、独立桥接分数权重 |
| `ID_SGTR_SEED_CANDIDATE_LIMIT` / `ID_SGTR_SEED_LIMIT` | `20` / `5` | 候选种子和保留种子上限 |
| `ID_SGTR_MAX_WORKERS` | `16` | 客户端最多并发任务数，不保证 GPU 端始终有 16 个请求 |

进行配对比较时，需固定模型、证据预算、数据版本、query ID 和 gate 协议。把 `ID_SGTR_SEED` 从 42 改成 43/44 得到的是**同一批问题**上的重复运行，不是新增测试样本。

## 评测与测试

```powershell
python -m knowledge_graph.experiments.run_p0 evaluate --results results/2wiki-retrieval-seed42.csv --scored-output results/2wiki-retrieval-seed42-scored.csv --output results/2wiki-retrieval-seed42-metrics.json
python -m knowledge_graph.experiments.run_p0 efficiency --results results/2wiki-retrieval-seed42.csv --output results/2wiki-retrieval-seed42-efficiency.json
python -m pip install pytest
python -m pytest tests -q
```

统一评测输出答案 EM/F1/Precision/Recall，以及 Evidence Recall、Precision 和 Complete Evidence Set Rate。结果 CSV 还记录 query ID、预测答案、证据、模型调用次数、检索轮次、耗时与 token 遥测；效率汇总包含均值及延迟分位数。比较之前要核对 `count` 与空答案数；进程结束不等于预期行数全部进入评测。

同一批 query ID 的两种方法可执行配对比较：

```powershell
python -m knowledge_graph.experiments.run_p0 compare --a results/method-a-scored.csv --b results/method-b-scored.csv --metric f1 --precomputed --samples 10000 --seed 42 --output results/a-vs-b-f1.json
```

如果 ID 不一致或重复，比较命令会报错。Retrieval 与 Reasoning 是不同实验设定，应分别报告。其他子命令见 `python -m knowledge_graph.experiments.run_p0 --help`。

## 常见问题与发布边界

- **缺少 `qa.csv` 或 parquet 文件：** 将对应数据压缩包在仓库根目录重新解压，核对上表的数据集 split 名称。
- **模型不存在或 HTTP 报错：** 检查 `.env` 的端口、模型服务名与实际部署；重排器必须对每条输入文本返回一个带索引的分数。
- **16 workers 但 GPU `Running` 较低：** worker 还会执行检索、重排和图计算，并不代表 GPU 始终有 16 条生成请求。
- **首次运行较慢：** 正式处理问题前需加载实体索引和向量；建议先执行 3 条连通性测试。
- **发布范围：** 本仓库不包含生成数据、模型权重、真实密钥或旧实验结果；基准数据和预训练模型分别遵守其原始许可。

## 代码位置

- `knowledge_graph/{hotpot,2wiki,musique}/query_global.py`：Retrieval 入口。
- `knowledge_graph/{hotpot,2wiki,musique}/query_local.py`：Reasoning 入口。
- `knowledge_graph/experiments/pcef_v2.py`：受约束证据集合选择。
- `knowledge_graph/experiments/query_reranker.py`：Cross-Encoder 与逐跳动态重排。
- `knowledge_graph/experiments/stage0_gate.py`、`terminal_recovery.py`：答案接受与终端综合。
- `scripts/pcef_v2_balanced.ps1`：均衡配置。
- `tests/`：算法及评测检查。

生成数据、结果、日志、缓存、密钥和模型权重均不包含在此代码发布目录中。
