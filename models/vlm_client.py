"""Qwen3-VL client — supports frames-list and full video-file modes.

Frames mode is used by:
  * memory builder (per-segment frames),
  * answerer when reading evidence-clip frames.

Video-file mode is used by Direct Video QA baseline only.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from PIL import Image


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}


def _to_pil(frame, max_pixels: int | None = None) -> Image.Image:
    if isinstance(frame, Image.Image):
        img = frame
    elif isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        raise TypeError(f"Unsupported frame type: {type(frame)}")
    if max_pixels is not None:
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return img


class VLMClient:
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        attn_impl: str = "sdpa",
        max_new_tokens: int = 512,
    ):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_path = model_path
        self.default_max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=_DTYPE_MAP.get(dtype, torch.bfloat16),
            attn_implementation=attn_impl,
            device_map=device,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def chat_with_frames(
        self,
        frames: Sequence,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        max_pixels: int | None = None,
    ) -> str:
        """Multi-image chat. `frames` is a sequence of np arrays or PIL images."""
        pil_frames = [_to_pil(f, max_pixels=max_pixels) for f in frames]
        content = [{"type": "image", "image": img} for img in pil_frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return self._generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)

    @torch.inference_mode()
    def chat_with_video(
        self,
        video_path: str,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        fps: float = 1.0,
        max_pixels: int = 128 * 28 * 28,    # qwen_vl_utils video default
        min_pixels: int = 64 * 28 * 28,
    ) -> str:
        """Single-video chat using qwen_vl_utils for frame sampling/preprocessing.

        Note: qwen_vl_utils enforces max_pixels >= min_pixels. For very small max_pixels
        we also lower min_pixels to keep the assertion happy.
        """
        if max_pixels < min_pixels:
            min_pixels = max(28 * 28, max_pixels // 2)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages, max_new_tokens=max_new_tokens, temperature=temperature, with_video=True)

    def _generate(self, messages, max_new_tokens=None, temperature=0.0, with_video=False) -> str:
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
        if image_inputs:
            proc_kwargs["images"] = image_inputs
        if video_inputs:
            proc_kwargs["videos"] = video_inputs
        inputs = self.processor(**proc_kwargs).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        out = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded.strip()
