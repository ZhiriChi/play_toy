"""End-to-end Q&A pipeline tying Layer 1 + 2 + 3 together."""
from __future__ import annotations

import time
from pathlib import Path

from .config import load_config
from .layer1_base import OllamaClient
from .layer2_rag import RAGLayer
from .layer3_feedback import FeedbackLayer
from .storage import Storage


class Pipeline:
    def __init__(self, config_path: str | Path | None = None):
        self.cfg = load_config(config_path) if config_path else load_config()
        self.client = OllamaClient(
            host=self.cfg["ollama"]["host"],
            model=self.cfg["ollama"]["chat_model"],
            timeout_s=self.cfg["ollama"]["request_timeout_s"],
        )
        self.storage = Storage(self.cfg["paths"]["db_path"])
        self.rag = RAGLayer(self.cfg, self.client, self.storage)
        self.feedback = FeedbackLayer(self.storage)

    def ingest(self, path: str | Path) -> dict:
        return self.rag.ingest_path(path)

    def list_documents(self) -> list[dict]:
        return self.storage.list_documents()

    def delete_document(self, doc_id: str) -> None:
        self.rag.delete_document(doc_id)

    def ask(self, query: str) -> dict:
        t0 = time.time()
        hits = self.rag.retrieve(query)
        context = self.rag.build_context_block(hits)

        messages = [
            {"role": "system", "content": self.cfg["generation"]["system_prompt"]},
            {
                "role": "user",
                "content": f"参考材料:\n{context}\n\n用户问题:\n{query}",
            },
        ]
        resp = self.client.chat(
            messages=messages,
            temperature=self.cfg["generation"]["temperature"],
            max_tokens=self.cfg["generation"]["max_tokens"],
            top_logprobs=self.cfg["generation"]["logprobs_top_k"],
        )
        latency_ms = (time.time() - t0) * 1000

        token_rows = [
            (t.position, t.token, t.chosen_logprob, t.entropy) for t in resp.tokens
        ]
        conv_id = self.storage.log_conversation(
            query=query,
            response=resp.text,
            latency_ms=latency_ms,
            avg_entropy=resp.avg_entropy,
            cross_entropy=resp.cross_entropy,
            retrieved_chunk_ids=[h["chunk_id"] for h in hits],
            token_rows=token_rows,
        )
        return {
            "conversation_id": conv_id,
            "response": resp.text,
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
