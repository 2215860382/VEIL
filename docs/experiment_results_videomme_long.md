# VEIL 实验结果分析 — VideoMME Long Duration

**Benchmark**: VideoMME Long（300 视频，900 题，4 选 1）  
**模型**: Qwen3.5-27B（VLM + LLM + Embedder: BGE-M3）  
**记忆库**: similarity-grouped bank（`videomme_L_27B`），L2 范数 BGE-M3 检索 + SigLIP 视觉去重  

---

## 1. 总体结果

| 实验配置 | N | Overall | 备注 |
|---|---|---|---|
| **kf32 + strict verifier** | 900 | **74.44%** | 当前主线最优 |
| main baseline | 900 | 73.89% | 无 kf cap 限制的早期版本 |
| kf=∞ (无上限) | 900 | 73.00% | 帧越多反而更差 |
| kf32 + loose verifier | 900 | 72.89% | loose 不如 strict |
| text-first kf32 | 900 | 70.67% | 92% 不看图，掉明显 |
| **oracle kf=16** (偷看答案) | 900 | **77.33%** | 上界参考（旧库） |
| oracle kf=∞ | 900 | 76.56% | oracle 帧过多也变差 |
| oracle kf=32 | 877 | ~76.4% | 接近 kf=16 oracle |

**Oracle gap**：主线与 oracle 差距约 **2.9pp**，主要来自检索召回和 verifier 判断。

---

## 2. 消融实验（ablation）

| 去掉的模块 | Overall | vs 主线 |
|---|---|---|
| 完整主线 (kf32 strict) | 74.44% | — |
| 去掉 verifier（直接 3 轮后答题） | 71.67% | -2.77pp |
| 去掉 rubric judge | 71.56% | -2.88pp |
| 只用 1 次 query（不迭代） | 69.78% | -4.66pp |
| 粗粒度 24 帧检索（无细粒度） | 66.78% | -7.66pp |
| 纯 LLM 迭代（无视觉） | 61.56% | -12.88pp |

---

## 3. 关键配置对比

### 3.1 Keyframe Cap

| kf cap | Overall | 结论 |
|---|---|---|
| kf = 32 | **74.44%** | 最优 |
| kf = ∞ | 73.00% | -1.44pp，VLM 注意力被稀释 |

VLM answerer 同时处理过多帧时注意力分散，32 帧是当前平衡点。

### 3.2 Verifier 严格度

| Verifier | Overall | 结论 |
|---|---|---|
| Strict（保守，更多 unknown） | **74.44%** | 最优 |
| Loose（激进，减少 unknown） | 72.89% | -1.55pp |

Strict verifier 保留更多 unknown → 更多 ANSWER_SUFFICIENT 判定 → 迭代继续收集证据。

### 3.3 Text-first（先文后图）

| 策略 | Overall | 看图比例 |
|---|---|---|
| 正常（直接用帧） | 74.44% | 100% |
| Text-first v1 | 70.67% | 7.8% |

92% 的题直接用文本回答，视觉信息严重不足。按题型看损失最大的：

| 题型 | 正常 kf32 | Text-first | 损失 |
|---|---|---|---|
| Spatial Perception | 66.7% | 33.3% | **-33.4pp** |
| Temporal Perception | 66.7% | 50.0% | -16.7pp |
| Attribute Perception | 77.8% | 66.7% | -11.1pp |
| Action Recognition | 65.1% | 57.1% | -7.9pp |
| OCR Problems | 71.4% | **78.6%** | +7.1pp ✅ |
| Counting Problem | 45.8% | **52.1%** | +6.3pp ✅ |

> OCR 和 Counting 题文本已足够，看图反而引入噪声。

---

## 4. 各题型详细对比

| 题型 | main | **kf32** | kf=∞ | loose | textfirst | oracle kf16 |
|---|---|---|---|---|---|---|
| Action Reasoning | 73.3% | 72.2% | 73.3% | 72.2% | 69.4% | **80.0%** |
| Action Recognition | 63.5% | **65.1%** | 61.9% | 55.6% | 57.1% | 61.9% |
| Attribute Perception | 77.8% | 77.8% | 77.8% | 74.1% | 66.7% | **81.5%** |
| Counting Problem | 43.8% | 45.8% | 43.8% | 39.6% | **52.1%** | 45.8% |
| Information Synopsis | 87.7% | **88.3%** | 86.5% | 86.5% | 82.2% | 89.6% |
| OCR Problems | 71.4% | 71.4% | 71.4% | 71.4% | **78.6%** | 85.7% |
| Object Reasoning | 76.7% | **77.9%** | 74.2% | 75.8% | 73.8% | 77.1% |
| Object Recognition | 63.0% | 66.7% | 64.8% | **68.5%** | 63.0% | 74.1% |
| Spatial Perception | 66.7% | 66.7% | 66.7% | 66.7% | 33.3% | 66.7% |
| Spatial Reasoning | 81.8% | **90.9%** | 90.9% | 90.9% | 90.9% | 90.9% |
| Temporal Perception | 66.7% | 66.7% | 66.7% | 66.7% | 50.0% | 50.0% |
| Temporal Reasoning | **71.4%** | 69.2% | 70.3% | 72.5% | 68.1% | 78.0% |
| **Overall** | 73.9% | **74.4%** | 73.0% | 72.9% | 70.7% | **77.3%** |

---

## 5. 结论与下一步

### 有效的配置
- ✅ **kf cap = 32**（vs 无限制）：+1.44pp，帧数限制防注意力稀释
- ✅ **Strict verifier**（vs loose）：+1.55pp，保守判断促使更多迭代

### 无效 / 有害的配置
- ❌ **kf = ∞**：-1.44pp，帧太多 VLM 注意力分散
- ❌ **Loose verifier**：-1.55pp，过早停止迭代
- ❌ **Text-first（少看图）**：-3.77pp，视觉信息关键不可省略

### 待验证
- 🔄 **Pyramid L1 记忆库**（新建库 + 固定 10s chunk + VLM 多图 caption）：实验进行中
- 🔄 **Text-first v2**（更激进触发看图）：等待运行
- 📌 **多粒度检索**（L1+L2+L3 分层检索）：待实现
