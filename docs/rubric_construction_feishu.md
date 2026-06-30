# VEIL · Rubric 构造管线

> 代码：`src/generate_rubric/construct/*.py`，脚本 `scripts/run_two_rounds.sh`，产物 `outputs/rubric/*.yaml`
> 只讲两件事：**rubric 如何引入（评什么、接在哪）** + **rubric 如何构造（七步管线）**。

---

## 一、Rubric 如何引入

### 1. 它评什么：evidence sufficiency，不是答案对错

VEIL 是「长视频 → 记忆库 → 迭代检索 → 回答」系统，verifier 的职责不是判断答案对不对，而是判断 **当前检索到的证据链，是否已经足够"验证或排除"每个选项**。够则停，不够则把缺口交给 planner 补检索。

所以 rubric 评的是 **证据充分性**。三条核心区分（写死在 `prompts.py:PAIRWISE_SYSTEM`，是整套方法的"宪法"）：

- rubric_score = 当前证据链能否 **verify or exclude 每个选项**
- rubric_score ≠ 答案正确性
- rubric_score ≠ 选项被支持的程度
- **一个错误选项也能拿高分**——只要证据足够把它排除

### 2. 它接在哪：verifier 的评分标准 + planner 的检索路由

```
question ─▶ planner ─▶ retriever ─▶ verifier〔用 rubric 逐选项打分〕
              ▲                          │
              │  不足: 按 failure_repair_action 给缺口   │ 充分
              └──────────────────────────┘           ▼
                     (迭代 ≤N 轮)                  answerer ─▶ option
```

rubric 在闭环里是 **双重角色**：
- **裁判**：verifier 让 LLM 对「每个选项 × 每条标准」打 0/0.5/1，加权平均后判 `SUFFICIENT/INSUFFICIENT`。
- **导航**：每条标准自带 `failure_repair_action`（refine_query / time_sorted / dense_sample / broadcast / asr_match），不满足时**直接告诉 planner 用哪种策略重检索**。

这是它和普通 LLM-as-judge rubric 最大的不同。

### 3. 它怎么被加载（`src/agents/verifier.py`）

- `_rubric_config()` 默认加载 `outputs/rubric/direct_answer_generated_v2.yaml`，可用环境变量 `VEIL_RUBRIC_PATH` 覆盖；每次 run 把 rubric 路径 + sha1 记进 `*.meta.json`，保证可复现。
- `get_rubric_dict(question, task_type)`：**general 标准恒生效** + 命中题型则**追加**该题型专属标准；scoring_rule / threshold 以题型为准，否则回落 general（默认 `average` / `0.80`）。

YAML 结构：
```yaml
general:        # 3 条通用标准 coverage / specificity / consistency
  scoring_rule: average
  sufficient_threshold: 0.80
type_aliases:   # 题型名 → 模板 key
templates:      # 12 个题型各自的专属标准
```

---

## 二、Rubric 如何构造

### 总览：七步管线 + 两轮增量

```
sample_dev → chains_from_result → make_pairs → extract_pair_criteria
           → aggregate_type_rubric → (weight_rubric) → backtest_report
```

理论根基（`README.md` / `PAIRWISE_SYSTEM` 明确引用）：
- **OnlineRubrics（成对增量诱导）**：从已有 rubric 出发，比较两条证据链，**只抽取尚未被覆盖的新差异**。
- **Chasing the Tail**：优先挖 **高质量链之间** 的差异（good-vs-great、great-vs-excellent），而非只盯 weak-vs-good——因为简单的系统已会做，收益在尾部。

| 步 | 文件 | 输入→输出 | 干什么 |
|----|------|-----------|--------|
| ① 采样 | `sample_dev.py` | 900 结果 → dev720/val90/test90 | 按题型分层 + 难题优先（答错/判不足/有 unknown 排前） |
| ② 造链 | `chains_from_result.py` | VEIL trace → weak/good/great 链 | 逐轮累积证据，按迭代位置+最终对错打质量标签 |
| ③ 配对 | `make_pairs.py` | 链 → 成对(≤3/题) | Chasing-the-Tail 优先级挑强弱对比 |
| ④ 抽标准 | `extract_pair_criteria.py` | 对+LLM → 候选标准 | 6 步差异分析→"未覆盖的新充分性标准" |
| ⑤ 聚合 | `aggregate_type_rubric.py` | 候选 → 题型 rubric | map-reduce 压缩去重，每题型 ≤7 条 |
| ⑥ 加权 | `weight_rubric.py` | rubric → 带权 | LLM 打 1–5 重要性 + 注入 groundedness 守门 |
| ⑦ 回测 | `backtest_report.py` | 全产物 → 统计 | 标出采样不足的题型 |

**两轮（`run_two_rounds.sh`）**：第二轮把第一轮 rubric 当"已有标准"喂回 ④，于是只抽出第一轮没覆盖的更细差异——OnlineRubrics 的在线增量在工程上的落地。最终主产物 = round1 ∪ round2 合并的 `round2_merged_runtime.yaml`。

### 逐步拆解

**① `sample_dev.py` — 分层 + 难题优先采样**
按 `question_type` 分 12 组；`_reasons` 从 900 结果读出每题"病征"（`system_wrong` / `final_insufficient` / `has_unknown_options` / `multi_round`），`sort_key` 把难题排前。→ 构造预算花在最有信息量的难题上。result 只用于排序，不泄题；val/test 干净留出。

**② `chains_from_result.py` — 从迭代轨迹造"质量分层"证据链**
遍历每题 `trace_iters`，**逐轮累积** chunk id，每轮生成一条"截止到此轮"的链；质量标签靠迭代位置推断（`_chain_quality`）：第 0 轮=**weak**、中间=**good**、末轮且答对=**great**（答错降 good）。
> 关键：利用"同题里越往后累积越充分"这个天然偏序，**零额外标注**就得到"弱→强"的质量阶梯。这是整套方法免标注的根。

**③ `make_pairs.py` — Chasing-the-Tail 配对**
按质量 rank 排序后按固定优先级挑对（每题 ≤3）：
```
weak_vs_excellent              # 跨度最大，奠基
good_vs_great / great_vs_excellent   # ★ 尾部差异，最有价值
excellent_vs_excellent_diverse # 同级多样性
weak_vs_good                   # 兜底
```

**④ `extract_pair_criteria.py` — 成对差异 → 新标准（核心 LLM 步）**
对每对（弱链 vs 强链），`PAIRWISE_USER_TEMPLATE` 让 LLM 按固定 6 步推理：① 弱链能判什么 → ② 弱链不能判什么 → ③ 强链补了什么 → ④ 为何使某些选项可判 → ⑤ 哪些差异**已被现有 rubric 覆盖**(丢) → ⑥ 哪些**未覆盖**→成为新标准。
输出严格 JSON，每条标准带 `name / description / score_1 / score_half / score_0 / repair_action / source_observation`。
两条硬约束（防偷懒）：必须**扎根本对真实差异、禁止世界知识**；禁止"证据相关/完整/清晰"这类泛泛标准。
> `--rubric-path` 是两轮开关：第二轮把 round1 rubric 作为 `initial_rubric` 传入，第 5 步"已覆盖"判断就会过滤重复，逼出尾部细标准。工程：ThreadPool 并发、按 `pair_id` 断点续跑、多 vLLM endpoint 轮询。

**⑤ `aggregate_type_rubric.py` — map-reduce 聚合**
候选成百上千，需压到每题型 ≤7 条，两段式防超长：
- **map**：每题型候选切 24 条/块，每块 LLM 压成 ≤4 条；
- **reduce**：各块再合并成 ≤7 条，去重、合并重叠、删泛泛。
保留原则：只留**能解释多个失败、能区分质量阶梯、能帮 verifier 输出 true/false/unknown、能指导 planner 重检索**的标准。
`_render_final_yaml` 组装成 verifier 可加载结构（general + type_aliases + 12 templates）。
工程鲁棒性：可断点续跑的 `.state.json`（崩了能接着跑）、`--render-only` 不调 LLM 重渲染、`--base-rubric` 合并各轮 delta、`_quote_scalars`/`_safe_parse_block` 自动修复 LLM 非法 YAML（裸冒号补引号 + 坏块跳过不污染全局）、`--agg-workers` 多题型并发。

**⑥ `weight_rubric.py` — 加权 + 反作弊守门（可选）**
LLM 对每条标准就"对充分性判断有多关键"打 1–5，写回 `weight`；并往 `general` 注入正向打分的 `evidence_groundedness` 守门标准——惩罚"靠世界知识/选项措辞回声/凭空脑补"做出的判断（1=全基于真实证据，0=脑补），保证加权平均仍在 [0,1]。产物用 verifier loader 实跑校验。

**⑦ `backtest_report.py` — 诊断**
统计各题型题数/链质量/对类型/标准数，标出 `pairs<5 或 criteria<3` 的采样不足题型，指导下一轮加采样。

### 产物示例（`round2_merged_runtime.yaml`，temporal_reasoning）

```yaml
- name: multi_entity_temporal_completeness
  description: 必须明确锚定并打时间戳给选项里 ALL distinct entities/topics/actions，
    才能构造完整时序；只锚一部分则缺失项相对顺序模糊，无法排除仅差该项位置的选项。
  score_1: 每个选项里提到的项都有引入时间 → 可定全序，排除所有错误排列
  score_half: 只锚部分(如 4 选 2) → 缺失项顺序仍模糊
  score_0: 一个关键项都没锚 → 任何顺序都无法判定
  failure_repair_action: dense_sample          # 不满足→指导 planner 这样补检索
  source_pair_examples: [124:weak_vs_great:..., 142:good_vs_great:...]  # 可追溯到哪几对长出来
```
三档锚点（1/0.5/0）描述具体可操作；`failure_repair_action` 回流 planner；`source_pair_examples` 保留可追溯性。

---

## 三、方法图（论文用）

横向泳道，从左到右；要视觉强调 ① 质量阶梯（免标注信号）② two-round 回环（OnlineRubrics）③ 每条标准的 `failure_repair_action` 回流 planner：

```
┌─────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌─────────────────────┐
│ VEIL traces │──▶│ ② chains_from_   │──▶│ ③ make_pairs │──▶│ ④ pairwise          │
│ 900题+迭代  │   │   result         │   │  Chasing-    │   │   extraction (LLM)  │
│   轨迹      │   │  weak/good/great │   │  the-Tail    │   │  6步差异→新标准      │
└─────────────┘   └──────────────────┘   └──────────────┘   └──────────┬──────────┘
   │ ① sample_dev (分层+难题优先)                                       │
   │        ┌──── two-round 回环 (round1 rubric 喂回 ④, 过滤已覆盖) ────┤
   │        ▼                                                           ▼
   │   ┌──────────────────────────────────────────────────────────────────┐
   └──▶│ ⑤ aggregate (map-reduce, ≤7/题型) → ⑥ weight + groundedness guard │
       │         → round2_merged_runtime.yaml (general + 12 题型 templates) │
       └──────────────────────────────────────────────────────────────────┘
```

---

## 附：文件索引

| 用途 | 路径 |
|------|------|
| 七步管线 | `src/generate_rubric/construct/*.py` |
| 核心 prompt（宪法） | `src/generate_rubric/construct/prompts.py` |
| 数据结构/初始粗 rubric | `src/generate_rubric/construct/schema.py` |
| 两轮脚本 | `scripts/run_two_rounds.sh` |
| 主产物 | `outputs/rubric/round2_merged_runtime.yaml` |
| 运行时默认 rubric | `outputs/rubric/direct_answer_generated_v2.yaml` |
| verifier 消费 | `src/agents/verifier.py`（`_rubric_config`/`get_rubric_dict`） |
