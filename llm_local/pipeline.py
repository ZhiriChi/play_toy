"""End-to-end Q&A pipeline tying Layer 1 + 2 + 3 together."""
from __future__ import annotations

import re
import time
from pathlib import Path

from .config import load_config
from .layer1_base import OllamaClient
from .layer2_rag import RAGLayer
from .layer3_feedback import FeedbackLayer
from .storage import Storage

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_thinking(text: str) -> tuple[str | None, str]:
    """For reasoning models that emit <think>…</think>, return (thinking, answer)."""
    m = _THINK_RE.search(text)
    if not m:
        return None, text
    thinking = m.group(1).strip()
    answer = (text[: m.start()] + text[m.end():]).strip()
    return thinking, answer


class Pipeline:
    def __init__(self, config_path: str | Path | None = None):
        self.cfg = load_config(config_path) if config_path else load_config()
        self.client = OllamaClient(
            host=self.cfg["ollama"]["host"],
            default_model=self.cfg["ollama"]["default_model"],
            timeout_s=self.cfg["ollama"]["request_timeout_s"],
        )
        self.storage = Storage(self.cfg["paths"]["db_path"])
        self.rag = RAGLayer(self.cfg, self.client, self.storage)
        self.feedback = FeedbackLayer(self.storage)

    def available_models(self) -> list[dict]:
        return list(self.cfg["ollama"]["models"])

    def model_info(self, name: str) -> dict | None:
        for m in self.cfg["ollama"]["models"]:
            if m["name"] == name:
                return m
        return None

    def ingest(self, path: str | Path) -> dict:
        return self.rag.ingest_path(path)

    def list_documents(self) -> list[dict]:
        return self.storage.list_documents()

    def delete_document(self, doc_id: str) -> None:
        self.rag.delete_document(doc_id)

    def ask(self, query: str, model: str | None = None) -> dict:
        use_model = model or self.cfg["ollama"]["default_model"]
        info = self.model_info(use_model) or {}
        is_reasoning = bool(info.get("reasoning"))

        t0 = time.time()
        hits = self.rag.retrieve(query)

        # Keep the prompt minimal — on some GPUs (e.g. RTX 2070 on current
        # Ollama) long multi-line Chinese system blocks produce numerically
        # unstable logits and the model degrades into garbage. We add only the
        # retrieved materials, when present, and let the instruct-tuned model
        # handle the rest from the bare user query.
        if hits:
            context = self.rag.build_context_block(hits)
            user_content = f"参考资料:\n{context}\n\n问题: {query}"
        else:
            user_content = query
        messages = [{"role": "user", "content": user_content}]
        gen = self.cfg["generation"]
        resp = self.client.chat(
            messages=messages,
            model=use_model,
            temperature=gen["temperature"],
            max_tokens=gen["max_tokens"],
            top_logprobs=gen["logprobs_top_k"],
            num_ctx=gen.get("num_ctx", 2048),
            num_gpu=gen.get("num_gpu", 0),
            keep_alive=gen.get("keep_alive", 600),
        )
        latency_ms = (time.time() - t0) * 1000

        thinking, answer = split_thinking(resp.text) if is_reasoning else (None, resp.text)

        token_rows = [
            (t.position, t.token, t.chosen_logprob, t.entropy) for t in resp.tokens
        ]
        conv_id = self.storage.log_conversation(
            query=query,
            response=answer,
            latency_ms=latency_ms,
            avg_entropy=resp.avg_entropy,
            cross_entropy=resp.cross_entropy,
            retrieved_chunk_ids=[h["chunk_id"] for h in hits],
            token_rows=token_rows,
            model=use_model,
            thinking=thinking,
        )
        return {
            "conversation_id": conv_id,
            "model": use_model,
            "response": answer,
            "thinking": thinking,
            "hits": hits,
            "avg_entropy": resp.avg_entropy,
            "cross_entropy": resp.cross_entropy,
            "latency_ms": latency_ms,
            "tokens": [
                {"position": t.position, "token": t.token,
                 "chosen_logprob": t.chosen_logprob, "entropy": t.entropy}
                for t in resp.tokens
            ],
        }

    def mark_clear(self, conversation_id: int, clear: bool = True) -> dict:
        return self.feedback.record(conversation_id, clear)
