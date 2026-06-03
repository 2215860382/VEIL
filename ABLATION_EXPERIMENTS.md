# VEIL 消融实验——完整实现指南

**基线：** `veil_27b` = **71.56%** (644/900)  
**上界：** `veil_27b_oracle` = 80.71% (431/534)  

---

## 📋 已创建的消融实验脚本

### 共5个消融实验

所有脚本位置：`experiments/veil_27b_*.py`

| 脚本文件 | 消融维度 | 消融类型 | 消融参数 | 消融内容 | 预期影响 |
|---------|--------|--------|--------|--------|---------|
| **veil_27b_singlequery.py** | 维度1 | 搜索策略 | `single_query_iter0=True` | Iter-0不分解，直接用原问题 | -1 ~ -3% |
| **veil_27b_no_rubric_judge.py** | 维度2 | 充分性判断 | `rubric_judgment=False` | Verifier不用rubric，改为纯文本判断 | -2 ~ -5% |
| **veil_27b_strict_dedup.py** | 维度3a | **证据**去重 | `evidence_dedup_threshold=0.90` | 证据去重严格（0.85→0.90），丢弃更多相似证据 | -0.5 ~ -2% |
| **veil_27b_high_query_threshold.py** | 维度3b | **Query**去重 | `query_history_dedup_threshold=0.99` | Query去重条件严格（0.9→0.99），保留相似query | -0.5 ~ +1% |
| **veil_27b_ignore_verifier.py** | 维度4 | 反馈循环 | `ignore_verifier_signal=True` | Planner无视Verifier反馈 | -3 ~ -5% |

---

## 🚀 快速开始

### 1. 单个实验运行

```bash
cd /home2/ycj/Project/VEIL

# 运行No-Rubric消融
PYTHONPATH=. python experiments/veil_27b_no_rubric_judge.py \
    --config configs/videomme_memory_bank.yaml \
    --vlm-api-url http://localhost:8000 \
    --llm-api-url http://localhost:8001 \
    --bge-gpu cuda:3 \
    --workers 1

# 运行Single-Query消融
PYTHONPATH=. python experiments/veil_27b_singlequery.py \
    --config configs/videomme_memory_bank.yaml \
    --vlm-api-url http://localhost:8000 \
    --llm-api-url http://localhost:8001 \
    --bge-gpu cuda:3 \
    --workers 1

# 类似地运行其他消融...
```

### 2. 批量运行（推荐）

```bash
# 自动运行所有P1消融
bash scripts/run_ablations.sh \
    configs/videomme_memory_bank.yaml \
    http://localhost:8000 \
    http://localhost:8001 \
    cuda:3
```

### 3. 比较结果

```bash
python scripts/compare_ablations.py
```

输出示例：
```
====================================================================================================
Pipeline                       Accuracy        vs Baseline              N
====================================================================================================
📌 veil_27b                    71.56%           baseline               900
🔴 veil_27b_no_rubric_judge    69.12%           -2.44%                 900
📍 veil_27b_singlequery        70.30%           -1.26%                 900
📝 veil_27b_verifier_text_only 70.45%           -1.11%                 900
🔗 veil_27b_ignore_verifier    68.95%           -2.61%                 900
```

---

## 📊 实验参数总览

### 参数对应关系

```python
# 维度1：Iter-0初始化策略
single_query_iter0=False         # 默认：LLM自动分解成2-4条原子query
single_query_iter0=True          # 消融1：单query不分解，直接用原问题

# 维度2：Verifier充分性判断方式
rubric_judgment=True             # 默认：用rubric系统逐选项打分
rubric_judgment=False            # 消融2：无rubric，改为纯文本整体分析

# 维度3a：证据去重阈值
evidence_dedup_threshold=0.85    # 默认：BGE相似度≥0.85的证据去重
evidence_dedup_threshold=0.90    # 消融3a：更严格（0.85→0.90）的去重

# 维度3b：Query去重阈值
query_history_dedup_threshold=0.9   # 默认：Jaccard相似度≥0.9的query去重
query_history_dedup_threshold=0.99  # 消融3b：更严格（0.9→0.99）的去重，保留更多query变体

# 维度4：Planner是否接收Verifier反馈
ignore_verifier_signal=False     # 默认：Planner接收Verifier的verdict指导
ignore_verifier_signal=True      # 消融4：Planner无视Verifier反馈，盲目生成

# 其他保持主线配置
query_history_dedup_threshold=0.9         # query去重(Jaccard相似度)
query_evidence_dedup_threshold=0.70       # query-证据漂移去重(BGE相似度)
rubric_rerank=True               # 答题前用rubric重排证据
max_iter=3                        # 最多3轮迭代
coarse_top_k=8, final_top_k=8   # 检索top-k
```

---

## 📈 解读结果的方法

### Accuracy Delta 的含义

| Delta值 | 含义 | 建议 |
|--------|------|------|
| **+0.5% ~ +2%** | 消融改进了性能 | ✅ 优化点，考虑融入主线 |
| **-0.5% ~ +0.5%** | 无显著差异 | ⚪ 可选，看复杂度 |
| **-2% 以下** | 消融明显伤害性能 | ❌ 基线配置正确，说明该组件重要 |

### 例子解读

假设结果如下：
```
📌 veil_27b                    71.56%  baseline
📍 veil_27b_singlequery        70.30%  -1.26%    ← 维度1：单query损失1.26%，LLM分解有益
🔴 veil_27b_no_rubric_judge    69.12%  -2.44%    ← 维度2：无rubric损害2.44%，Rubric系统有效
📊 veil_27b_strict_dedup       71.22%  -0.34%    ← 维��3：严格去重略下降，0.85最优平衡
🔗 veil_27b_ignore_verifier    68.95%  -2.61%    ← 维度4：无Verifier反馈损害2.61%，循环很重要
```

**结论：**
- 维度1：LLM自动分解子问题很重要（单query损失1.26%）
- 维度2：Rubric系统确实有效（禁用损害2.44%，结构化打分重要）
- 维度3：默认阈值0.85最优（严格阈值略损害0.34%，不宜过严）
- 维度4：Verifier反馈循环很重要（Planner无反馈损害2.61%，指导信号关键）

---

## 🔄 后续实验计划

当前P1实验包括4个核心维度。根据P1结果，可考虑后续的精细化调优（如去重阈值微调、rubric形式优化等）。

---

## 📁 输出文件结构

```
outputs/results/videommeL/
├── veil_27b.jsonl                      # 基线（已有）
├── veil_27b_oracle.jsonl               # Oracle上界（已有）
│
├── veil_27b_singlequery.jsonl          # 维度1：单query不分解
├── veil_27b_no_rubric_judge.jsonl      # 维度2：Verifier不用rubric打分
├── veil_27b_strict_dedup.jsonl         # 维度3a：证据去重阈值0.85→0.90
├── veil_27b_high_query_threshold.jsonl # 维度3b：Query去重阈值0.9→0.99
└── veil_27b_ignore_verifier.jsonl      # 维度4：Planner无视Verifier反馈
```

每个JSONL文件包含：
```json
{
  "key": "videomme|video_id|sample_idx|pipeline",
  "benchmark": "videomme",
  "question_type": "topic",
  "video_id": "...",
  "question": "...",
  "candidates": ["A", "B", "C", "D"],
  "gold_answer": "A",
  "pred_letter": "B",
  "correct": false,
  "trace_iters": [...],
  "elapsed": 12.34,
  "error": null
}
```

---

## ⚙️ 命令行参数速查

所有实验脚本共享相同的CLI参数：

```bash
python experiments/veil_27b_*.py \
    --config CONFIG.yaml                    # 必需
    --memory-dir PATH                       # 可选：memory bank路径，默认${bench}_L_27b_27b
    --out OUTPUT.jsonl                      # 可选：输出文件，默认{pipeline}.jsonl
    --vlm-api-url http://localhost:8000    # VLM服务URL
    --llm-api-url http://localhost:8001    # LLM服务URL
    --bge-gpu cuda:3                        # BGE embedding GPU
    --workers 1                             # 并行workers数
    --sample-start 0                        # 样本范围（可用于分页运行）
    --sample-end 100
```

---

## 🛠️ 故障排除

### 问题：某个消融跑得很慢
**原因：** Rubric打分或Planner生成query耗时  
**解决：**
```bash
# 用小样本快速测试
--sample-start 0 --sample-end 50
```

### 问题：results目录已有该管线的结果，想重新跑
**解决：** 删除旧JSONL或改输出文件名
```bash
rm outputs/results/videommeL/veil_27b_no_rubric_judge.jsonl
# 或
--out outputs/results/videommeL/veil_27b_no_rubric_judge_v2.jsonl
```

### 问题：模型API不可用
**检查：**
```bash
curl http://localhost:8000/v1/models  # VLM
curl http://localhost:8001/v1/models  # LLM
```

---

## 📝 脚本差异速查

所有消融脚本的关键差异在run_sample函数中的kw dict（其余代码完全相同）：

```python
# veil_27b.py (baseline - 71.56%)
kw = dict(
    ...,
    query_history_dedup_threshold=0.9,
    evidence_dedup_threshold=0.85,       # 默认：BGE相似度≥0.85去重
    query_evidence_dedup_threshold=0.70,
    rubric_rerank=True,
    rubric_judgment=True,                # 默认：用rubric打分
    single_query_iter0=False,            # 默认：LLM自动分解
    ignore_verifier_signal=False,        # 默认：接收Verifier反馈
)

# veil_27b_singlequery.py (维度1)
kw = dict(
    ...,
    single_query_iter0=True,             # ← 改动：直接用原问题，不分解
)

# veil_27b_no_rubric_judge.py (维度2)
kw = dict(
    ...,
    rubric_judgment=False,               # ← 改动：无rubric，纯文本分析
)

# veil_27b_strict_dedup.py (维度3a - 证据去重)
kw = dict(
    ...,
    evidence_dedup_threshold=0.90,       # ← 改动：证据去重更严格(0.85→0.90)
)

# veil_27b_high_query_threshold.py (维度3b - Query去重)
kw = dict(
    ...,
    query_history_dedup_threshold=0.99,  # ← 改动：Query去重更严格(0.9→0.99)，保留更多变体
)

# veil_27b_ignore_verifier.py (维度4)
kw = dict(
    ...,
    ignore_verifier_signal=True,         # ← 改动：Planner无视Verifier反馈
)
```

---

## 📊 预期时间成本

假设：
- 单个样本平均耗时 10-15 秒（包括检索、Verifier、Planner）
- 900 题样本集
- 1 worker

| 管线 | 预计时间 | 备注 |
|------|---------|------|
| veil_27b | 2.5-3.75 小时 | baseline已完成 |
| veil_27b_no_rubric_judge | 2.5-3.75 小时 | 略快（无rubric打分） |
| veil_27b_singlequery | 2-3 小时 | 快（仅1个query） |
| veil_27b_verifier_text_only | 2.5-3.75 小时 | 与baseline相同 |
| veil_27b_ignore_verifier | 2.5-3.75 小时 | 与baseline相同 |

**总计：** 12.5-19 小时单机运行  
**建议：** 如果有多机，并行跑2-3个实验可缩短至 5-10 小时

---

## 🎯 推荐运行策略

### 分阶段运行（风险最小）

**Day 1:** 跑2个快速消融
```bash
python experiments/veil_27b_no_rubric_judge.py ...       # 无rubric判断
python experiments/veil_27b_singlequery.py ...           # 单query（快）
```

**Day 2:** 跑剩下的
```bash
python experiments/veil_27b_verifier_text_only.py ...    # 仅文本信号
python experiments/veil_27b_ignore_verifier.py ...       # 无Verifier反馈
```

**Day 3:** 分析结果，决定P2方向
```bash
python scripts/compare_ablations.py
```

### 全速运行（需基础设施）

若可用多GPU/多机，并行跑所有4个消融：
```bash
for script in no_rubric_judge singlequery verifier_text_only ignore_verifier; do
    PYTHONPATH=. python experiments/veil_27b_$script.py ... &
done
wait
python scripts/compare_ablations.py
```

---

## 下一步

1. ✅ 已创建 4 个消融脚本
2. ✅ 已创建运行和比较脚本
3. 📌 **待做：** 根据运行结果，选择赢家组合进行P2消融
4. 📌 **待做：** 若有显著改进，准备merge回主线

