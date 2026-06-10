"""SigLIP-based frame embedder — local model or HTTP API client."""
from __future__ import annotations

from typing import List

import numpy as np

_DEFAULT_MODEL = "/home2/ycj/Models/google/siglip-large-patch16-384"


class SigLIPEmbedder:
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        device: str = "cuda:0",
        api_url: str | None = None,
    ):
        self.api_url = (api_url or "").rstrip("/") if api_url else None
        if self.api_url:
            import requests
            self._session = requests.Session()
            return
        import torch
        from transformers import AutoProcessor, AutoModel
        self._torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model.to(device).eval()

    def encode_images(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        """Return (N, D) L2-normalized image embeddings."""
        if self.api_url:
            resp = self._session.post(
                f"{self.api_url}/siglip/encode_images",
                json={"paths": list(image_paths)},
                timeout=600,
            )
            resp.raise_for_status()
            return np.asarray(resp.json()["vectors"], dtype=np.float32)
        from PIL import Image
        torch = self._torch
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
                feats = self.model.get_image_features(**inputs)
                if hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_vecs.append(feats.float().cpu().numpy())
        return np.concatenate(all_vecs, axis=0)

    def encode_text(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Return (N, D) L2-normalized text embeddings."""
        if self.api_url:
            resp = self._session.post(
                f"{self.api_url}/siglip/encode_text",
                json={"texts": list(texts)},
                timeout=300,
            )
            resp.raise_for_status()
            return np.asarray(resp.json()["vectors"], dtype=np.float32)
        torch = self._torch
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                inputs = self.processor(text=texts[i:i+batch_size], return_tensors="pt",
                                        padding=True, truncation=True).to(self.device)
                feats = self.model.get_text_features(**inputs)
                if hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_vecs.append(feats.float().cpu().numpy())
        return np.concatenate(all_vecs, axis=0)
