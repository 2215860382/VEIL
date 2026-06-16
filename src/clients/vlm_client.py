"""Vision–language model wrapper. Checkpoint comes from ``model_path`` (or API id via ``api_model``).

**Frames** (``chat_with_frames``): list of images — memory builder, answerer over evidence frames.

**Whole video file** (``chat_with_video``): local path; uses ``qwen_vl_utils`` + HF processor stack
as implemented today (swap implementation if you move to a different VL family).

**API** (``api_url``): JPEG base64 ``image_url`` messages to a vLLM/OpenAI-compatible server.
"""
from __future__ import annotations

import base64
import threading
import io
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


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class VLMClient:
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        attn_impl: str = "sdpa",
        max_new_tokens: int = 512,
        api_url: str | None = None,
        api_model: str | None = None,
    ):
        self.model_path = model_path
        self.default_max_new_tokens = max_new_tokens
        self._api_endpoints: list[str] | None = None
        self._api_rr = 0
        self._api_lock = threading.Lock()
        self._api_model = api_model or model_path
        self.model = None
        self.processor = None

        if api_url:
            import requests as _req
            self._requests = _req
            bases = [b.strip() for b in api_url.split(",") if b.strip()]
            self._api_endpoints = [b.rstrip("/") + "/v1/chat/completions" for b in bases]
            if not api_model:
                try:
                    r = _req.get(bases[0].rstrip("/") + "/v1/models", timeout=10)
                    self._api_model = r.json()["data"][0]["id"]
                except Exception:
                    self._api_model = model_path
        else:
            from transformers import AutoProcessor, AutoModelForImageTextToText
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
        enable_thinking: bool = False,
    ) -> str:
        """Multi-image chat. `frames` is a sequence of np arrays or PIL images."""
        if self._api_endpoints:
            return self._chat_with_frames_api(frames, prompt, max_new_tokens, temperature, max_pixels, enable_thinking)
        pil_frames = [_to_pil(f, max_pixels=max_pixels) for f in frames]
        content = [{"type": "image", "image": img} for img in pil_frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return self._generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)

    def _chat_with_frames_api(
        self,
        frames: Sequence,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        max_pixels: int | None = None,
        enable_thinking: bool = False,
    ) -> str:
        pil_frames = [_to_pil(f, max_pixels=max_pixels) for f in frames]
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(img)}"}}
            for img in pil_frames
        ]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self._api_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_new_tokens or self.default_max_new_tokens,
            "temperature": temperature if temperature > 0 else 0.0,
            # Qwen3-family models default to thinking-on; disable unless caller opts in.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        with self._api_lock:
            ep = self._api_endpoints[self._api_rr % len(self._api_endpoints)]
            self._api_rr += 1
        resp = self._requests.post(ep, json=payload, timeout=300)
        if resp.status_code in (400, 500) and pil_frames:
            import logging
            logging.getLogger(__name__).warning(
                "VLM API %s error with %d frames — retrying without images",
                resp.status_code, len(pil_frames),
            )
            payload["messages"] = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            with self._api_lock:
                ep = self._api_endpoints[self._api_rr % len(self._api_endpoints)]
                self._api_rr += 1
            resp = self._requests.post(ep, json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"]
        if "</think>" in result:
            result = result.split("</think>", 1)[1]
        return result.strip()

    def chat_with_content(
        self,
        content: list,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> str:
        """Send a pre-built mixed content list (text + image_url blocks) to the API.

        ``content`` is a list of OpenAI-style content items:
          {"type": "text", "text": "..."}
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        The caller is responsible for encoding images to base64.
        """
        if not self._api_endpoints:
            raise NotImplementedError("chat_with_content is only supported in API mode")
        payload = {
            "model": self._api_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_new_tokens or self.default_max_new_tokens,
            "temperature": temperature if temperature > 0 else 0.0,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        with self._api_lock:
            ep = self._api_endpoints[self._api_rr % len(self._api_endpoints)]
            self._api_rr += 1
        resp = self._requests.post(ep, json=payload, timeout=300)
        if resp.status_code in (400, 500):
            import logging
            logging.getLogger(__name__).warning(
                "VLM API %s error with mixed content — retrying text-only", resp.status_code)
            text_only = [item for item in content if item.get("type") == "text"]
            payload["messages"] = [{"role": "user", "content": text_only}]
            with self._api_lock:
                ep = self._api_endpoints[self._api_rr % len(self._api_endpoints)]
                self._api_rr += 1
            resp = self._requests.post(ep, json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"]
        if "</think>" in result:
            result = result.split("</think>", 1)[1]
        return result.strip()

    @torch.inference_mode()
    def chat_with_video(
        self,
        video_path: str,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        fps: float = 1.0,
        max_pixels: int = 128 * 28 * 28,
        min_pixels: int = 64 * 28 * 28,
    ) -> str:
        """Single-video chat: delegates frame packing to ``qwen_vl_utils`` + processor (current VL stack).

        ``qwen_vl_utils`` requires ``max_pixels >= min_pixels``; we shrink ``min_pixels`` when needed.
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
        # API mode: forward to API endpoint
        if self._api_endpoints:
            payload = {
                "model": self._api_model,
                "messages": messages,
                "max_tokens": max_new_tokens or self.default_max_new_tokens,
                "temperature": temperature if temperature > 0 else 0.0,
            }
            with self._api_lock:
                ep = self._api_endpoints[self._api_rr % len(self._api_endpoints)]
                self._api_rr += 1
            resp = self._requests.post(ep, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"]
            if "</think>" in result:
                result = result.split("</think>", 1)[1]
            return result.strip()

        # Local mode: use processor + model
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
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
            repetition_penalty=1.3,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        out = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded.strip()
