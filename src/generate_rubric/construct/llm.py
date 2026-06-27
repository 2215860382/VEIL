"""OpenAI-compatible LLM helper with URL health filtering."""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass

import requests


@dataclass
class LLMRouter:
    bases: list[str]
    model: str = "Qwen3.5-27B"
    timeout: int = 600

    @classmethod
    def from_urls(cls, urls: str, model: str = "Qwen3.5-27B") -> "LLMRouter":
        bases = [u.strip().rstrip("/") for u in urls.split(",") if u.strip()]
        live = []
        session = requests.Session()
        session.trust_env = False
        for base in bases:
            try:
                r = session.get(base + "/v1/models", timeout=5)
                if r.status_code == 200:
                    live.append(base)
            except Exception:
                continue
        if not live:
            raise RuntimeError(f"no live vLLM endpoints from: {urls}")
        return cls(live, model=model)

    def __post_init__(self) -> None:
        self._cycle = itertools.cycle(self.bases)
        self._lock = threading.Lock()

    def chat(self, system: str, user: str, max_tokens: int = 1800, temperature: float = 0.0) -> str:
        last_error = None
        for _ in range(len(self.bases)):
            with self._lock:
                base = next(self._cycle)
            try:
                session = requests.Session()
                session.trust_env = False
                r = session.post(
                    base + "/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                if "</think>" in text:
                    text = text.split("</think>", 1)[1].strip()
                return text.strip()
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"all LLM endpoints failed: {last_error}")
