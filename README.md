# VEIL — Video Evidence Iterative Loop

Prototype for evidence-grounded long-video reasoning:

```
long video
   ├── memory builder (VLM + SigLIP + BGE — checkpoints you choose)
   │       └── memory bank (per-segment captions / events / ASR-aligned text)
   └── question
           ├── planner (inside VEIL pipeline; text LLM)  →  search_query
           ├── retriever (BGE-M3 dense + optional cross-encoder rerank)
           ├── verifier (text LLM) →  is_sufficient ? next_query
           │      ↺ iterate up to N rounds
           └── answerer (VLM or text)  →  final option
```

The point of the prototype: validate that memory bank + iterative retrieval + sufficiency check beats both
the "watch the whole video" baseline and a naive single-shot RAG.

## Layout

Each top-level directory has one purpose. Read this table once and you can navigate the repo.

| Dir            | Purpose (one line) |
|----------------|-------------------|
| `experiments/` | Eval: **`experiments/configs/*.yaml`**, **`experiments/pipelines/*.py`**, **`experiments/run_experiments.py`** (entry; `--pipelines` selects a pipeline). |
| `data/`        | Benchmark loaders → typed `*Sample` records. |
| `models/`      | Thin wrappers around external models (VLM, LLM, BGE, reranker). |
| `memory/`      | Build memory banks from video (similarity grouping or fixed time windows). |
| `reasoning/`   | Verifier + answerer (text / VL). Planner prompts live in **`experiments/pipelines/veil.py`**. |
| `eval/`        | Answer parsing + accuracy metrics. |
| `utils/`       | Config loader (`utils/config.py`), logging — no domain logic. |
| `scripts/`     | Downloads, cluster helpers, `analyze_traces.py`; main eval entry is **`experiments/`**. |
| `outputs/`     | Generated artifacts (memory banks, eval JSONL, logs). |

There is **no** top-level `retrieval/` package: coarse / rerank / VEIL retrieval run inside **`experiments/pipelines/`** (and helpers under `memory/` / `models/` as needed).

### File map

```
experiments/configs/
  mlvu_memory_bank.yaml              # MLVU + bank-backed pipelines
  mlvu_direct_fullvideo.yaml         # MLVU + full-video direct VLM
  videomme_memory_bank.yaml
  videomme_direct_fullvideo.yaml

experiments/run_experiments.py       # benchmark × pipelines → JSONL

experiments/pipelines/
  direct_videoQA.py   # full video frames → VLM → option (--pipelines direct)
  coarse_rag.py       # BGE top-k → answerer (coarse8 / coarse64)
  rerank_rag.py       # BGE pool + rerank top-k → answerer (rerank_rag8, llm_rerank8)
  veil.py             # iterative planner / retrieval / verifier (veil_*)

data/
  load_mlvu.py
  load_videomme.py

models/
  llm_client.py
  vlm_client.py
  embedder.py        # BGE-M3
  reranker.py        # cross-encoder + optional LLM listwise rerank

memory/
  schema.py                  # MemoryChunk / MemoryBank
  build_specs.py             # build-time benchmark paths + default cache dirs (see below)
  build_similarity_memory.py # SigLIP grouping + VLM/API + BGE (CLI)
  build_fixedframe_memory.py # fixed wall-clock chunks + VLM bank (CLI)

reasoning/
  verifier.py
  answerer.py

eval/
  parse_answer.py
  compute_accuracy.py

utils/
  config.py          # YAML inherit + ${var} interpolation
  logging.py

scripts/
  *.sh, analyze_traces.py, …
```

### `memory/build_specs.py` in one sentence

It returns **small specification dicts** (dataset roots, default `outputs/memory/...` cache dir names, embedder fields, fixed-frame chunk defaults) so the memory **build** CLIs share one source of truth instead of hard-coding paths. **“Specs”** = these declarative build-time configs, not the eval YAMLs under `experiments/configs/`.

## Setup

Conda env `veil` (Python 3.11) is already created with torch 2.11+cu128.

```bash
conda activate veil
# pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# pip install -r requirements.txt
```

**Model and data paths for eval** are whatever you set in the YAML you pass to `experiments/run_experiments.py` (for example `experiments/configs/mlvu_memory_bank.yaml`: `models.*.model_path`, `benchmark.*`).

**Memory build** uses `memory/build_specs.py` for benchmark locations when you pass `--benchmark mlvu|videomme`; BGE path comes from the dict returned there (or from `--config` YAML). **VLM and SigLIP checkpoints are not defaulted** in `build_similarity_memory` — pass `--vlm-model` and `--siglip-model` explicitly (and `--api-model` whenever `--api-url` is set).

## Data

MLVU-Dev layout is under `benchmark.json_dir` / `benchmark.video_dir` in your YAML (often `/home/Dataset/Dataset/MLVU/MLVU/`). Video-MME paths are the `parquet_path`, `video_dir`, and optional `subtitle_dir` fields in `videomme_memory_bank.yaml`.

## Running

```bash
cd /home2/ycj/Project/VEIL

# 1) Similarity-group memory banks (SigLIP + VLM + BGE). Example: require explicit VLM/SigLIP paths.
PYTHONPATH=. python -m memory.build_similarity_memory \
  --benchmark mlvu \
  --vlm-model /path/to/your/Qwen3-VL \
  --siglip-model /path/to/your/siglip \
  --vlm-gpu cuda:0 --bge-gpu cuda:0 --siglip-gpu cuda:0

# Fixed time-window banks (event / dense):
PYTHONPATH=. python -m memory.build_fixedframe_memory --benchmark mlvu --modes event dense

# 2) Baselines
PYTHONPATH=. python experiments/run_experiments.py \
  --config experiments/configs/mlvu_direct_fullvideo.yaml --pipelines direct
PYTHONPATH=. python experiments/run_experiments.py \
  --config experiments/configs/mlvu_memory_bank.yaml --pipelines rerank_rag8 \
  --memory-dir outputs/memory/mlvu_similarity

# 3) VEIL
PYTHONPATH=. python experiments/run_experiments.py \
  --config experiments/configs/mlvu_memory_bank.yaml --pipelines veil_rerank8 \
  --memory-dir outputs/memory/mlvu_similarity
```

Results go to the config's `eval.output_dir` when set. In this repo, the VideoMME-L config writes to `outputs/results/videommeL/`; otherwise `run_experiments.py` falls back to `outputs/results/<benchmark>/`. Similarity banks default to `outputs/memory/mlvu_similarity` or `outputs/memory/videomme_L_27b_27b` when using `--benchmark` (see `memory/build_specs.py`). **`run_experiments.py`** defaults `--memory-dir` to `outputs/memory/<benchmark>_fixed` if omitted — for similarity banks, pass `--memory-dir` explicitly as above.

## What to verify first (Stage-1 goals)

1. memory bank > directly watching the whole video
2. retrieval reduces information dilution
3. verifier catches insufficient evidence
4. iterative loop fills evidence gaps

Look for `direct < rerank_rag < veil` accuracy on a small slice (1 task type, 1 video) before scaling up.
