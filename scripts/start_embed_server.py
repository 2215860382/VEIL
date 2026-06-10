"""HTTP server hosting BGE-M3 + SigLIP — load once, share across experiments.

Usage:
    PYTHONPATH=. python scripts/start_embed_server.py \\
        --port 9000 --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clients.embedder import BGEM3Embedder
from src.clients.siglip_embedder import SigLIPEmbedder

app = FastAPI(title="VEIL Embedding Server")

BGE: BGEM3Embedder | None = None
SIGLIP: SigLIPEmbedder | None = None


class TextRequest(BaseModel):
    texts: List[str]


class PathRequest(BaseModel):
    paths: List[str]


@app.get("/health")
def health():
    return {"status": "ok", "bge": BGE is not None, "siglip": SIGLIP is not None}


@app.post("/bge/encode")
def bge_encode(req: TextRequest):
    assert BGE is not None
    return {"vectors": BGE.encode(req.texts).tolist()}


@app.post("/siglip/encode_images")
def siglip_encode_images(req: PathRequest):
    assert SIGLIP is not None
    return {"vectors": SIGLIP.encode_images(req.paths).tolist()}


@app.post("/siglip/encode_text")
def siglip_encode_text(req: TextRequest):
    assert SIGLIP is not None
    return {"vectors": SIGLIP.encode_text(req.texts).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bge-model", default="/home2/ycj/Models/BAAI/bge-m3")
    ap.add_argument("--siglip-model", default="/home2/ycj/Models/google/siglip-large-patch16-384")
    args = ap.parse_args()

    global BGE, SIGLIP
    print(f"[embed-server] loading BGE-M3 on {args.device}...", flush=True)
    BGE = BGEM3Embedder(model_path=args.bge_model, use_fp16=True, device=args.device)
    print(f"[embed-server] loading SigLIP on {args.device}...", flush=True)
    SIGLIP = SigLIPEmbedder(model_path=args.siglip_model, device=args.device)
    print(f"[embed-server] ready, listening on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
