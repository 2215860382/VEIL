# VEIL Experiment Changelog — VideoMME-L (900 questions, Qwen3.5-27B)

## 结果文件命名规范

格式：`videommeL_{方法}_{回答模型}_{内容类型}[_{视觉}].jsonl`

**内容类型**：
- `caption` — memory_text 为 VLM 生成的视觉描述（caption 模式）
- `summary` — memory_text 为 VLM 生成的摘要（summary 模式，当前新实验）

**视觉**：
- 无后缀 — 答案模型只看文字
- `_kf` — 答案模型能看关键帧图像

---

## 当前结果文件

| 文件 | 方法 | 题数 | 全集准确率 | 306题准确率 | 说明 |
|------|------|------|-----------|------------|------|
| `coarse64_27b_caption.jsonl` | coarse64 | 900 | 62.44% | 69.93% | caption，无关键帧 |
| `coarse64_27b_caption_kf.jsonl` | coarse64 | 900 | 64.00% | 71.24% | caption，答案可见关键帧 |
| `coarse8_27b_caption_kf.jsonl` | coarse8 | 900 | 61.67% | 69.61% | caption，答案可见关键帧 |
| `direct_27b.jsonl` | direct | 900 | 65.89% | 71.57% | thinking 关闭 |
| `veil8_27b_summary_kf.jsonl` | veil8 | 进行中 | — | — | summary，全模型可见关键帧（改进版 verifier+planner） |
| `coarse8_27b_summary.jsonl` | coarse8 | 进行中 | — | — | summary，无关键帧 |

**备份**：`veil8_27b_kfall_legacyverifier.jsonl`（旧版 legacy verifier，64.67% / 71.24%）

---

## 改动历史

### caption nokf 基线 → caption kfall（veil8 准确率 62.89% → 64.67%，+1.78%）

**Bug fix（此前已做）**: `--answer-evidence-k 8` 截断了 iter1+ 的所有证据，使迭代形同虚设。删除后 veil8 从 59.78% → 62.89%（+3.11%）。

**Verifier 重构（此前已做）**: rubric 从自由文本升级为三层结构化 YAML（critical gates + rubric criteria + reasoning），不同题型可通过 type_aliases / keyword_rules 路由到不同模板。

**证据去重（此前已做）**: 跨 iter 新增 chunk 若与已有证据 v_semantic cos_sim ≥ 0.85 则丢弃，避免同质证据占用上下文。

**Planner 漂移检测（此前已做）**: 新 query embedding 与已有证据向量 max cos_sim ≥ 0.70 时提前终止迭代。

**视觉通道 iter1+**: `vq = vis_query if it == 0 else ""` 改为 `vq = vis_query if it == 0 else current_query`，iter1+ 的 SigLIP 粗检索也走视觉通道。

**关键帧全模型注入（nokf → kfall）**:
- 每轮检索后加载对应关键帧 JPEG（路径由代码重建，不用 chunk.keyframe_path）
- 跨帧视觉去重：v_visual cos_sim ≥ 0.92 的帧跳过，无数量上限
- 关键帧以 base64 image_url 格式注入 planner / verifier / answerer 消息（API 模式生效）
- 分类改善：OCR +14.3%，Counting +10.4%，Spatial +4.5%，全部 10 类提升

**coarse_rag 同步**: `coarse_rag.py` 加入视觉去重（v_visual cos_sim ≥ 0.92）+ keyframe_cap=8，关键帧传给 answerer，与 veil 保持可比基线。coarse64 另加 max_evidence_chars=18000（避免 64 块文本超 vLLM 8192 token 上限）。

### Rubric 重设计 + 空答案 retry（当前改动）

**核心问题**：`answer_disambiguation` 标准太宽松，只要证据提到了相关话题就能得 1.0，不需要实际排除错误选项。导致 verifier FP 率 25.5%（sufficient 判对率仅 74.5%）。

**Rubric 重设计**（`reasoning/rubric_templates.yaml`）：
- 所有模板中 `answer_disambiguation` 替换为 `wrong_options_excluded`，要求证据明确矛盾/排除至少两个错误选项，1.0 的门槛更高。
- 视觉/动作类新增 `answer_option_mentioned` gate：至少有一个选项的关键实体出现在证据中。
- `scoring_rule: min` 用于 temporal/counting/ocr（所有标准必须同时满足）。
- 阈值全面提高：object_visual 0.35→0.75，action 0.40→0.75，synopsis 0.40→0.75，ocr 0.45→0.75，counting 0.70→0.75，temporal 0.60→0.49(min)，default 0.50→0.65。
- Verifier `max_new_tokens` 300→512（新 rubric 输出更多 criteria 字段）。

**空答案 retry**（`reasoning/answerer.py`）：
- TextAnswerer 和 VLAnswerer 均加 retry：若首次 answer 为空，用加强指令重试一次（128 tokens，明确要求输出 A/B/C/D 之一）。

### caption kfall → summary kf（进行中，改进版 Verifier + Planner）

**关键 Bug 修复**：旧 veil.py 导入 `get_rubric`（返回字符串→Legacy Verifier），改为 `get_rubric_dict`（结构化三层 rubric）。旧版所有 veil 实验实际使用的是简单 legacy verifier。

**Per-type rubric 阈值**：object_visual=0.35，action/synopsis=0.40，ocr=0.45，default=0.50，temporal=0.60，counting=0.70。不同题型用不同充足性门槛，避免对简单视觉题过度迭代、对时序题迭代不足。

**Planner 改进**：
- `_extract_covered_times()` 从已有证据中提取已覆盖时间段，提示 planner 去检索空白时间段
- `_MISSING_TYPE_HINTS` 字典：按缺失类型（temporal/visual/count/identity/causal/comparative）提供具体查询格式指引
- `missing_type` 字段传入 planner，生成更有针对性的 follow-up query

**完整 trace 记录**：`iterations` 列表新增 score、gate_failure、criteria、reasoning、missing_evidence 字段，方便后续分析。

---

## Log Files

| 文件 | 用途 |
|------|------|
| `logs/videommeL_veil_v2_20260512_2144.log` | veil8 nokf 完整运行日志 |
| `logs/videommeL_v3_veil8_20260513_0148.log` | veil8 kfall (legacy verifier) 运行日志 |
| `logs/videommeL_coarse8_20260513_1053.log` | coarse8 caption_kf 运行日志 |
| `logs/videommeL_direct27b_v3_20260513_1047.log` | direct 运行日志 |
| `logs/videommeL_coarse64_27b_v3_20260513_1156.log` | coarse64 caption_kf 运行日志 |
| `logs/videommeL_veil8_27b_kfall_20260513_1427.log` | veil8 summary_kf（改进版，当前运行）|
| `logs/vllm_gpu4_p8780_20260513_1048.log` | vLLM GPU4:8780 |
| `logs/vllm_gpu1_p8777_20260513_1047.log` | vLLM GPU1:8777 |
| `logs/vllm_gpu5_p8781_20260513_1153.log` | vLLM GPU5:8781（veil8 专用）|
