# VEIL — Video Evidence Iterative Loop

Prototype for evidence-grounded long-video reasoning:

```
long video
   ├── memory builder (Qwen3-VL-8B)
   │       └── memory bank (per-segment captions / events / OCR)
   └── question
           ├── planner (Qwen3-8B, no-think)  →  search_query
           ├── retriever (BGE-M3 dense + bge-reranker-v2-m3 cross-encoder)
           ├── verifier (Qwen3-8B, no-think) →  is_sufficient ? next_query
           │      ↺ iterate up to N rounds
           └── answerer (Qwen3-VL-8B)        →  final option
```

The point of the prototype: validate that memory bank + iterative retrieval + sufficiency check beats both
the "watch the whole video" baseline and a naive single-shot RAG.

## Layout

Each top-level directory has one purpose. Read this table once and you can navigate the repo.

| Dir          | Purpose (one line)                                                 |
|--------------|--------------------------------------------------------------------|
| `configs/`   | YAML run/benchmark configs (`base.yaml`, `mlvu.yaml`, ...)         |
| `data/`      | Benchmark JSON loaders → typed `*Sample` records                    |
| `models/`    | Thin wrappers around external models — swap backends in one place  |
| `memory/`    | VEIL core: build memory bank from a video                          |
| `retrieval/` | VEIL core: vector store + retrieval helpers                        |
| `reasoning/` | VEIL core: planner / verifier / answerer                           |
| `pipelines/` | Orchestration: `direct_video_qa` / `naive_rag` / `veil_iterative`  |
| `eval/`      | Answer parsing + accuracy metrics                                  |
| `utils/`     | Pure infrastructure (config loader, logging) — nothing else        |
| `scripts/`   | CLI entry points (`build_memory.py`, `run_eval.py`)                |
| `outputs/`   | Generated artifacts (memory banks, eval results, logs)             |

Detailed file map:

```
configs/
  base.yaml          mlvu.yaml          videomme.yaml

data/
  load_mlvu.py       load_videomme.py    # stub

models/                                  # all "load weights, expose .chat() / .encode()"
  llm_client.py      # Qwen3-8B (or 30B-A3B) text chat
  vlm_client.py      # Qwen3-VL frame/video chat
  embedder.py        # BGE-M3 dense
  reranker.py        # bge-reranker-v2-m3 cross-encoder

memory/
  schema.py          # MemoryChunk / MemoryBank pydantic models
  segment_video.py   # decord-based fixed-window cutting
  build_memory.py    # VLM → per-segment structured memory_text

retrieval/
  vector_store.py    # numpy-cosine in-memory store

reasoning/
  planner.py         # question → search query (JSON)
  verifier.py        # evidence sufficiency + next_query (JSON)
  answerer.py        # final A/B/C/D (JSON, VL or text-only)

pipelines/
  direct_video_qa.py # baseline 1: full video → VLM → answer
  naive_rag.py       # baseline 2: bank → BGE retrieve+rerank → VLAnswerer
  veil_iterative.py  # ours: planner → retrieve → verifier → loop → answer

eval/
  parse_answer.py    compute_accuracy.py

utils/
  config.py          # YAML with inherit + ${var} interpolation
  logging.py

scripts/
  build_memory.py    run_eval.py
```

## Setup

Conda env `veil` (Python 3.11) is already created with torch 2.11+cu128.

```bash
conda activate veil
# Existing wheels — only re-install if you blow the env away.
# pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# pip install -r requirements.txt
```

Model paths (already populated):

| role          | path                                              |
|---------------|---------------------------------------------------|
| VLM           | `/home/Dataset/Models/Qwen/Qwen3-VL-8B-Instruct`  |
| LLM           | `/home/Dataset/Models/Qwen/Qwen3-8B`              |
| embedder      | `/home/Dataset/Models/BAAI/bge-m3`                |
| reranker      | `/home/Dataset/Models/BAAI/bge-reranker-v2-m3`    |

To swap planner/verifier to the MoE Instruct-2507, change `models.llm.model_path` in `configs/base.yaml` to
`/home/Dataset/Models/Qwen/Qwen3-30B-A3B-Instruct-2507`.

## Data

MLVU-Dev is at `/home/Dataset/Dataset/MLVU/MLVU/` (json + video). MLVU-Test is gated; once granted, it
will live at `/home/Dataset/Dataset/MLVU/MLVU_Test/`.

## Running

```bash
# 1) build memory banks (once per video)
python scripts/build_memory.py --config configs/mlvu.yaml \
    --task-types plotQA --max-videos 1

# 2) baselines
python scripts/run_eval.py --config configs/mlvu.yaml --pipeline direct \
    --task-types plotQA --max-videos 1
python scripts/run_eval.py --config configs/mlvu.yaml --pipeline naive_rag \
    --task-types plotQA --max-videos 1

# 3) VEIL
python scripts/run_eval.py --config configs/mlvu.yaml --pipeline veil \
    --task-types plotQA --max-videos 1
```

Results land in `outputs/results/mlvu/<pipeline>.json`, memory banks in `outputs/memory/mlvu/<video_id>.json`.

## What to verify first (Stage-1 goals)

1. memory bank > directly watching the whole video
2. retrieval reduces information dilution
3. verifier catches insufficient evidence
4. iterative loop fills evidence gaps

Look for `direct < naive_rag < veil` accuracy on a small slice (1 task type, 1 video) before scaling up.
