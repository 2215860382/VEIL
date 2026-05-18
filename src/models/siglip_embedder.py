"""SigLIP-based frame embedder for dynamic visual grouping.

Uses google/siglip-large-patch16-384 to embed video frames into
L2-normalized vectors for cosine-similarity grouping.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

_DEFAULT_MODEL = "/home2/ycj/Models/google/siglip-large-patch16-384"


class SigLIPEmbedder:
    def __init__(self, model_path: str = _DEFAULT_MODEL, device: str = "cuda:0"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model.to(device).eval()

    @torch.no_grad()
    def encode_images(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        """Return (N, D) L2-normalized image embeddings."""
        all_vecs = []
        for i in range(0, len(image_paths), batch_size):
            batch = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_vecs.append(feats.float().cpu().numpy())
        return np.concatenate(all_vecs, axis=0)

    @torch.no_grad()
    def encode_text(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Return (N, D) L2-normalized text embeddings."""
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            inputs = self.processor(text=texts[i:i+batch_size], return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_vecs.append(feats.float().cpu().numpy())
        return np.concatenate(all_vecs, axis=0)
