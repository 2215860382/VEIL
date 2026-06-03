# VEIL 消融实验计划（2026-05-30）

基线：`veil_27b` 71.56% (644/900) | oracle上界：80.71%

---

## 优先级 P1：核心三维度（并行跑）

### 1️⃣ **Rubric vs No-Rubric**（维度2：证据充分性判断方式）
**目标：** 确认rubric系统是否有实际收益

| 参数 | 当前主线 | 实验对比 | 预期影响 |
|------|---------|---------|---------|
| `rubric_judgment` | `True` | `False` | Verifier不用rubric打分，改为整体判断。预期下降3-5% |
| 对应管线 | veil_27b | `veil_27b_no_rubric` | - |

**实现：** 新建 `veil_27b_no_rubric.py`，改一行参数即可

---

### 2️⃣ **Query分解策略**（维度1：iter-0初始化）
**目标：** 单query不分解 vs LLM自动分解（当前主线）

| 参数/模式 | 现状 | Variant | 预期 |
|---------|------|---------|------|
| 初始query方式 | LLM自动分解(n=2-4) | 单条query不分解 | 预期↓ 1-3% |
| 管线 | veil_27b | `veil_27b_singlequery` | - |

**实现：** 在run_veil中添加 `single_query_iter0` 参数，直接用原问题作为iter-0的唯一query

---

### 3️⃣ **Planner如何利用Verifier反馈**（维度5：是否接收反馈指导）
**目标：** 确认Verifier反馈循环是否有益，Planner是否需要指导

| 模式 | 当前主线 | 对比 | 预期 |
|------|--------|------|------|
| 信息流 | Planner接收Verifier verdict | Planner无视Verifier反馈，随意生成query | 无反馈预期↓ 3-5% |
| 管线 | veil_27b | `veil_27b_ignore_verifier` | - |

**实现：** `ignore_verifier_signal=True` 时，传入空verdict给Planner.plan_next()

---

## 优先级 P2：二阶消融（在P1赢家基础上）

### 4️⃣ **Verifier证据归因显隐式**（维度4）
- 隐式推理 vs 显式attribution vs 显式+剪干扰
- 对应参数：`explicit_attribution`, `prune_distractors`
- 预期影响：0-1%（精细优化）
- 管线：`veil_27b_explicit_attr`, `veil_27b_prune_distractors`

### 5️⃣ **广播策略**（维度6）
- 关键词触发广播 (当前) vs LLM自行决定
- 对应参数：`keyword_broadcast` (需添加)
- 预期影响：0-2%
- 管线：`veil_27b_llmbcast`

### 6️⃣ **Planner跳过已解决子问题**（维度7）
- 不跳过 (当前) vs 跳过（`prune_satisfied=True`）
- 对应参数：`prune_satisfied`
- 预期影响：+1-2%（减少重复搜索）
- 管线：`veil_27b_prune_satisfied`

### 7️⃣ **答题前证据重排**（维度9）
- Rubric指导重排 (当前: `rubric_rerank=True`)
- 无重排对比
- 对应参数：`rubric_rerank`
- 预期影响：-1-2%
- 管线：`veil_27b_no_rubric_rerank`（已有旧结果）

---

## 分组实验方案（推荐）

### **第一轮：P1三个消融并行** （预计3-4天）
```
veil_27b_no_rubric_judge       # Verifier不用rubric（维度2）
veil_27b_singlequery           # 单query不分解（维度1）
veil_27b_ignore_verifier       # Planner无视Verifier反馈（维度5）
```

**统计方式：** 同一样本集（300题）对比，确保可比性

---

### **第二轮：组合&微调** （基于P1结果）
- 赢家组合：e.g. `no_rubric + singleq + text_only` 
- P2消融：在赢家基础上逐个开关

---

## 实现检查清单

- [ ] 检查当前 Verifier 中 `rubric_judgment` 的控制点（verifier.py）
- [ ] 确认 Planner 中子问题分解入口（planner.py，搜 `_planner_decompose_iter0` 或相似）
- [ ] 确认 Verifier → Planner 的信息传递路径（verdict dict 构造）
- [ ] 为每个新管线创建 `veil_27b_{variant}.py` 脚本模板
- [ ] 添加运行命令到 scripts/ 目录

---

## 当前代码状态

**已有参数（run_veil函数签名）：**
```python
rubric_judgment: bool = True          # 维度2 ✓
prune_satisfied: bool = False          # 维度7 ✓
force_option_subquestions: bool = False # 维度1 (部分) ?
explicit_attribution: bool = False     # 维度4 ✓
prune_distractors: bool = False        # 维度4 ✓
rubric_rerank: bool = False            # 维度9 ✓（主线为True）
```

**待确认/待添加：**
- [ ] 查 Planner.plan() 签名，确认有无对应 "仅文本" 的参数
- [ ] 查是否有 `keyword_broadcast` 参数（维度6）

---

## 预期收益

| 维度 | 乐观 | 保守 | 关键性 |
|------|------|------|--------|
| No-rubric | -5% | -2% | 高（如果↑说明rubric有bug） |
| 单query | -3% | -1% | 中（分解是否必要） |
| 4选项 | +2% | 0% | 中（显式约束有无帮助） |
| 信号方式 | -3% | -1% | 中（信息有效性）  |

**保守预期：** 通过微调，可能达到 72-73%（<1% 改进）  
**乐观预期：** 通过组合优化，可能达到 74-75%（+3% 改进）

---

## 下一步行动

1. **确认优先级** — 是否同意P1三个维度，还是有其他关注点？
2. **运行环境** — vLLM/API配置是否就绪？可以并行跑5个实验吗？
3. **样本集选择** — 全量900题还是分组300题（快速反馈）？
