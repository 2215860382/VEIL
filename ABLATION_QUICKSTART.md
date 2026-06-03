# VEIL 消融实验快速开始 ⚡

## 📂 已创建的文件

```
experiments/
├── veil_27b_singlequery.py              ← 维度1：不分解，直接用原问题
├── veil_27b_no_rubric_judge.py          ← 维度2：关闭rubric打分
├── veil_27b_strict_dedup.py             ← 维度3a：更严格的证据去重(0.85→0.90)
├── veil_27b_high_query_threshold.py     ← 维度3b：更严格的Query去重(0.9→0.99)
└── veil_27b_ignore_verifier.py          ← 维度4：Planner无视Verifier反馈

scripts/
├── run_ablations.sh                     ← 批量运行脚本
└── compare_ablations.py                 ← 结果对比分析

项目文档:
├── ABLATION_PLAN.md                     ← 详细规划（已更新）
└── ABLATION_EXPERIMENTS.md              ← 完整指南（已更新）
```

---

## 🚀 一键运行（推荐）

```bash
cd /home2/ycj/Project/VEIL

# 运行所有P1消融（自动批量）
bash scripts/run_ablations.sh

# 对比结果
python scripts/compare_ablations.py
```

**预计耗时：** 12-18小时（单GPU顺序运行所有5个消融）或 6-9小时（并行双GPU）

---

## 🎯 五个消融实验一览表

| 维度 | 脚本名称 | 参数改动 | 作用 | 预期影响 |
|-----|---------|--------|------|---------|
| 1️⃣ | `veil_27b_singlequery.py` | `single_query_iter0=True` | Iter-0不分解，直接用原问题 | ↓ 1-3% |
| 2️⃣ | `veil_27b_no_rubric_judge.py` | `rubric_judgment=False` | Verifier不用rubric，纯文本分析 | ↓ 2-5% |
| 3a️⃣ | `veil_27b_strict_dedup.py` | `evidence_dedup_threshold=0.90` | **证据**去重更严格（0.85→0.90） | ↓ 0.5-2% |
| 3b️⃣ | `veil_27b_high_query_threshold.py` | `query_history_dedup_threshold=0.99` | **Query**去重条件严格（0.9→0.99），保留变体 | -0.5 ~ +1% |
| 4️⃣ | `veil_27b_ignore_verifier.py` | `ignore_verifier_signal=True` | Planner无视Verifier反馈 | ↓ 3-5% |

---

## 💻 单个运行（手动）

```bash
# 模板命令（改最后的管线名）
PYTHONPATH=. python experiments/veil_27b_NO_RUBRIC_JUDGE.py \
    --config configs/videomme_memory_bank.yaml \
    --vlm-api-url http://localhost:8000 \
    --llm-api-url http://localhost:8001 \
    --bge-gpu cuda:3 \
    --workers 1
```

用你的管线名替换 `NO_RUBRIC_JUDGE`。

---

## 📊 查看结果

### 快速对比
```bash
python scripts/compare_ablations.py
```

### 单个管线深入分析
```bash
python3 << 'EOF'
import json
from pathlib import Path

path = Path("outputs/results/videommeL/veil_27b_no_rubric_judge.jsonl")
correct = sum(1 for line in path.open() if json.loads(line).get("correct"))
total = sum(1 for line in path.open())
print(f"{path.name}: {correct}/{total} = {100*correct/total:.2f}%")
EOF
```

---

## 🔧 高级选项

### 部分样本快速测试（20题，5分钟）
```bash
PYTHONPATH=. python experiments/veil_27b_singlequery.py \
    --config configs/videomme_memory_bank.yaml \
    ... \
    --sample-start 0 --sample-end 20
```

### 恢复中断的运行
脚本自动跳过已做过的样本（检查output JSONL中的key），直接运行：
```bash
PYTHONPATH=. python experiments/veil_27b_singlequery.py ...
# 自动继续之前未完成的样本
```

### 并行运行多个（需多GPU）
```bash
# 在后台运行，可同时进行3-4个
nohup python experiments/veil_27b_no_rubric_judge.py ... > log1.txt 2>&1 &
nohup python experiments/veil_27b_singlequery.py ... > log2.txt 2>&1 &
nohup python experiments/veil_27b_ignore_verifier.py ... > log3.txt 2>&1 &
wait
python scripts/compare_ablations.py
```

---

## 📈 解读结果

运行 `compare_ablations.py` 后，看这几列：

```
📌 veil_27b                       71.56%  baseline  
📍 veil_27b_singlequery           70.30%  -1.26%   ← 维度1：分解有益
🔴 veil_27b_no_rubric_judge       69.12%  -2.44%   ← 维度2：rubric系统重要
📊 veil_27b_strict_dedup          71.22%  -0.34%   ← 维度3a：证据阈值0.85最优
📝 veil_27b_high_query_threshold  71.89%  +0.33%   ← 维度3b：保留query变体略有帮助
🔗 veil_27b_ignore_verifier       68.95%  -2.61%   ← 维度4：反馈循环很重要
```

- **Delta > +1%** → ✅ 好的改进，考虑融入
- **Delta ≈ 0±0.5%** → ⚪ 可选
- **Delta < -1%** → ❌ 说明基线组件重要

---

## ❓ 常见问题

**Q: 为什么某个脚本更快/更慢？**  
A: 因为消融改变了Verifier/Planner的复杂度。例如：
- `singlequery` 快（仅1条query而不是2-4条，但可能精度下降）
- `no_rubric_judge` 快（无rubric打分，但精度下降）
- 其他维度复杂度相当，耗时类似

**Q: 可以只跑几个样本吗？**  
A: 可以，用 `--sample-end 100` 快速测试，确认脚本工作，再跑全量。

**Q: 某个脚本出错怎么办？**  
A: 检查日志（自动输出到console和error记录在output JSONL的error字段），常见原因：
- API不可用 → 检查VLM/LLM服务
- 内存不足 → 减少workers或关闭siglip (`--no-siglip`)
- 样本缺失 → memory bank路径不对

**Q: 如何保存结果便于后续分析？**  
A: 输出已在 `outputs/results/videommeL/`，可以：
```bash
cp outputs/results/videommeL/*.jsonl /backup/  # 备份
python scripts/compare_ablations.py > results_summary.txt
```

---

## 🎓 实验设计回顾

**目标：** 在主线71.56%基础上，通过消融验证：
1. Rubric系统是否重要？
2. 选项对齐query能否帮助？
3. 结构化vs纯文本信号哪个更好？
4. 剪枝已解决问题是否有益？

**方法：** 单参数消融，其他保持不变

**输出：** 4个新的JSONL结果文件 + 对比分析

---

## 📞 获取帮助

详见完整文档：
- `ABLATION_PLAN.md` — 详细规划和预期
- `ABLATION_EXPERIMENTS.md` — 完整实现指南

