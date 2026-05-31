"""Layer 1: base reasoning via local Ollama.

Talks to Ollama's OpenAI-compatible /v1/chat/completions endpoint. We ask for
`logprobs=true, top_logprobs=K` so we can recover per-token entropy and an
approximate cross-entropy of the response — both feed the monitoring layer.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import requests


@dataclass
class TokenInfo:
    position: int
    token: str
    chosen_logprob: float | None
    entropy: float | None


@dataclass
class LLMResponse:
    text: str
    tokens: list[TokenInfo] = field(default_factory=list)
    avg_entropy: float | None = None
    cross_entropy: float | None = None
    raw: dict | None = None


def _entropy_from_top(top: list[dict]) -> float | None:
    if not top:
        return None
    logps = [t.get("logprob") for t in top if t.get("logprob") is not None]
    if not logps:
        return None
    probs = [math.exp(lp) for lp in logps]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    return -sum(p * math.log(p + 1e-12) for p in probs)


class OllamaClient:
    def __init__(self, host: str, default_model: str, timeout_s: int = 180):
        self.host = host.rstrip("/")
        self.default_model = default_model
        self.timeout_s = timeout_s
        # Match the byte-for-byte shape of `curl` requests: raw UTF-8 (no
        # \uXXXX escapes), no gzip, no kept-alive connection from previous
        # calls. Some Ollama + GPU combos produce wrong logits when given
        # ASCII-escaped Chinese, presumably because the parser/tokenizer takes
        # a different path.
        self._headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

    def _post(self, url: str, payload: dict) -> requests.Response:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return requests.post(url, data=body, headers=self._headers, timeout=self.timeout_s)

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
        top_logprobs: int = 5,  # kept for API compatibility; /api/chat doesn't support logprobs
    ) -> LLMResponse:
        # Use native /api/chat — the OpenAI-compat /v1/chat/completions endpoint crashes
        # with CUDA illegal memory access on Ollama 0.24 + RTX 2070 (Turing SM 7.5).
        url = f"{self.host}/api/chat"
        payload: dict = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        r = self._post(url, payload)
        if not r.ok:
            raise RuntimeError(f"Ollama 调用失败 ({r.status_code}): {r.text[:300]}")
        data = r.json()
        text = (data.get("message") or {}).get("content") or ""

        # /api/chat doesn't expose per-token logprobs; entropy metrics will be None.
        return LLMResponse(
            text=text,
            tokens=[],
            avg_entropy=None,
            cross_entropy=None,
            raw=data,
        )

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        url = f"{self.host}/api/embed"
        r = self._post(url, {"model": model, "input": texts})
        if not r.ok:
            raise RuntimeError(f"Ollama embed 失败 ({r.status_code}): {r.text[:300]}")
        data = r.json()
        return data.get("embeddings") or []
