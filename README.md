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

| Dir              | Purpose (one line) |
|------------------|-------------------|
| `configs/`       | Eval YAMLs (one per benchmark × pipeline family). |
| `experiments/`   | Eval driver `run_experiments.py` + standalone analysis scripts (no business logic). |
| `scripts/`       | Shell helpers: launch vLLM servers, run ablation sweeps, etc. |
| `src/`           | Python package: all business code lives here. |
| `outputs/`       | Generated artifacts (memory banks, eval JSONL, logs). |

Inside `src/`:

| Dir                  | Purpose |
|----------------------|---------|
| `src/config.py`      | YAML inherit + `${var}` interpolation. |
| `src/dataloader/`    | Benchmark loaders → typed `*Sample` records. |
| `src/eval/`          | Answer parsing + accuracy metrics. |
| `src/memory/`        | Memory-bank build CLIs (`similarity.py`, `fixedframe.py`). Core primitives live in `src/memory/core/`. |
| `src/models/`        | Thin wrappers around external models (VLM, LLM, BGE, reranker, SigLIP). |
| `src/pipelines/`     | The four pipeline families: direct, coarse_rag, rerank_rag, veil. Planner prompts live in `src/reasoning/planner.py`. |
| `src/reasoning/`     | Planner + Verifier + Answerer + rubric templates. |
| `src/utils/`         | Logging, JSON helpers, GPU lock, jsonl utilities — no domain logic. |

There is **no** top-level `retrieval/` package: coarse / rerank / VEIL retrieval run inside **`src/pipelines/`** (and use helpers from `src/memory/core/` and `src/models/` as needed).

### File map

```
configs/
  mlvu_memory_bank.yaml              # MLVU + bank-backed pipelines
  mlvu_direct_fullvideo.yaml         # MLVU + full-video direct VLM
  videomme_memory_bank.yaml
  videomme_direct_fullvideo.yaml

experiments/
  veil_27b.py                        # main veil pipeline runner
  veil_27b_*.py                      # ablations (singlequery, no_rubric_judge, oracle, ignore_verifier)
  coarse24.py
  llm_iter.py
  core/                              # importable pipeline libs (veil.py, coarse_rag.py, rerank_rag.py, _keyframes.py, summarize_jsonl.py)

scripts/
  *.sh                               # vLLM launchers, ablation sweeps

src/
  config.py                          # YAML inherit + ${var} interpolation
  dataloader/
    mlvu.py
    videomme.py
    longvideobench.py
  eval/
    parse_answer.py
    compute_accuracy.py
  memory/
    similarity.py                    # SigLIP grouping + VLM/API + BGE (CLI)
    fixedframe.py                    # fixed wall-clock chunks + VLM bank (CLI)
    core/
      schema.py                      # MemoryChunk / MemoryBank
      specs.py                       # build-time benchmark paths + default cache dirs
      sample_frames.py
      dynamic_grouper.py
      consolidate.py
  clients/
    llm_client.py
    vlm_client.py
    embedder.py                      # BGE-M3
    reranker.py                      # cross-encoder + optional LLM listwise rerank
    siglip_embedder.py
  pipelines/
    direct_video_qa.py               # full video frames → VLM → option (--pipelines direct)
    coarse_rag.py                    # BGE top-k → answerer (coarse8 / coarse64)
    rerank_rag.py                    # BGE pool + rerank top-k → answerer (rerank_rag8, llm_rerank8)
    veil.py                          # iterative retrieval loop (veil_*)
    _keyframes.py
  agents/
    planner.py                       # iter-0 decomposition + iter≥1 unified planner
    verifier.py                      # rubric-guided evidence verification
    answerer.py
  rubric/
    templates/
      legacy.yaml                    # earlier experiment rubric
      generated_v2.yaml              # current generated rubric
    generation/
      generate_instance.py           # per-question rubric generation
      distill.py                      # question-type/general distillation
      prompts/
    artifacts/                       # ignored generation outputs and logs
  utils/
    logging.py
    jsonx.py
    gpu_lock.py
    strip_pipeline_from_jsonl.py
```

### `src/memory/core/specs.py` in one sentence

It returns **small specification dicts** (dataset roots, default `outputs/memory/...` cache dir names, embedder fields, fixed-frame chunk defaults) so the memory **build** CLIs share one source of truth instead of hard-coding paths. **“Specs”** = these declarative build-time configs, not the eval YAMLs under `configs/`.

## Setup

Conda env `veil` (Python 3.11) is already created with torch 2.11+cu128.

```bash
conda activate veil
# pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# pip install -r requirements.txt
```

**Model and data paths for eval** are whatever you set in the YAML you pass to `experiments/run_experiments.py` (for example `configs/mlvu_memory_bank.yaml`: `models.*.model_path`, `benchmark.*`).

**Memory build** uses `src/memory/core/specs.py` for benchmark locations when you pass `--benchmark mlvu|videomme`; BGE path comes from the dict returned there (or from `--config` YAML). **VLM and SigLIP checkpoints are not defaulted** in `src.memory.similarity` — pass `--vlm-model` and `--siglip-model` explicitly (and `--api-model` whenever `--api-url` is set).

## Data

MLVU-Dev layout is under `benchmark.json_dir` / `benchmark.video_dir` in your YAML (often `/home/Dataset/Dataset/MLVU/MLVU/`). Video-MME paths are the `parquet_path`, `video_dir`, and optional `subtitle_dir` fields in `videomme_memory_bank.yaml`.

## Running

```bash
cd /home2/ycj/Project/VEIL

# 1) Similarity-group memory banks (SigLIP + VLM + BGE). Example: require explicit VLM/SigLIP paths.
PYTHONPATH=. python -m src.memory.similarity \
  --benchmark mlvu \
  --vlm-model /path/to/your/Qwen3-VL \
  --siglip-model /path/to/your/siglip \
  --vlm-gpu cuda:0 --bge-gpu cuda:0 --siglip-gpu cuda:0

# Fixed time-window banks (event / dense):
PYTHONPATH=. python -m src.memory.fixedframe --benchmark mlvu --modes event dense

# 2) Baselines
PYTHONPATH=. python experiments/run_experiments.py \
  --config configs/mlvu_direct_fullvideo.yaml --pipelines direct
PYTHONPATH=. python experiments/run_experiments.py \
  --config configs/mlvu_memory_bank.yaml --pipelines rerank_rag8 \
  --memory-dir outputs/memory/mlvu_similarity

# 3) VEIL
PYTHONPATH=. python experiments/run_experiments.py \
  --config configs/mlvu_memory_bank.yaml --pipelines veil_rerank8 \
  --memory-dir outputs/memory/mlvu_similarity
```

Results go to the config's `eval.output_dir` when set. In this repo, the VideoMME-L config writes to `outputs/results/videommeL/`; otherwise `run_experiments.py` falls back to `outputs/results/<benchmark>/`. Similarity banks default to `outputs/memory/mlvu_similarity` or `outputs/memory/videomme_L_27b_27b` when using `--benchmark` (see `src/memory/core/specs.py`). **`run_experiments.py`** defaults `--memory-dir` to `outputs/memory/<benchmark>_fixed` if omitted — for similarity banks, pass `--memory-dir` explicitly as above.

## What to verify first (Stage-1 goals)

1. memory bank > directly watching the whole video
2. retrieval reduces information dilution
3. verifier catches insufficient evidence
4. iterative loop fills evidence gaps

Look for `direct < rerank_rag < veil` accuracy on a small slice (1 task type, 1 video) before scaling up.
