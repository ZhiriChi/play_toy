"""Layer 2: retrieval-augmented context from user-uploaded materials.

Each chunk carries a `reward_score` in SQLite that is bumped whenever the user
marks an answer as "clearly understood" (Layer 3). Retrieval re-ranks by
`similarity + reward_weight * tanh(reward_score / 3)` so good chunks rise.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .ingest import chunk_id_for, chunk_text, doc_id_for, extract_document
from .layer1_base import OllamaClient
from .storage import Storage


class RAGLayer:
    def __init__(self, cfg: dict, client: OllamaClient, storage: Storage):
        self.cfg = cfg
        self.client = client
        self.storage = storage
        self._chroma = chromadb.PersistentClient(
            path=cfg["paths"]["vector_dir"],
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._chroma.get_or_create_collection(
            name="materials",
            metadata={"hnsw:space": "cosine"},
        )
        self.embed_model = cfg["ollama"]["embed_model"]

        embed_cfg = cfg.get("embedding", {})
        self._embed_backend = embed_cfg.get("backend", "ollama")
        self._st_model = None
        if self._embed_backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            st_name = embed_cfg.get("st_model", "paraphrase-multilingual-MiniLM-L12-v2")
            self._st_model = SentenceTransformer(st_name, device="cpu")

    @staticmethod
    def _is_embeddable(text: str) -> bool:
        # Skip blank / punctuation-only chunks that some embedding models
        # (e.g. bge-m3 on Turing) return NaN for.
        if not text or len(text.strip()) < 3:
            return False
        return bool(re.search(r"[\w一-鿿]", text))

    @staticmethod
    def _embedding_is_finite(vec: list[float]) -> bool:
        return all(math.isfinite(v) for v in vec) and any(v != 0.0 for v in vec)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embed_backend == "sentence-transformers":
            vecs = self._st_model.encode(texts, normalize_embeddings=True)
            return vecs.tolist()
        # Ollama path: try batch first; fall back to per-text on NaN/error.
        try:
            vecs = self.client.embed(texts, model=self.embed_model)
            if len(vecs) == len(texts) and all(self._embedding_is_finite(v) for v in vecs):
                return vecs
        except Exception:
            pass

        out: list[list[float]] = []
        for t in texts:
            try:
                v = self.client.embed([t], model=self.embed_model)
                vec = v[0] if v else []
                if vec and self._embedding_is_finite(vec):
                    out.append(vec)
                    continue
            except Exception:
                pass
            out.append([0.0] * (len(out[0]) if out else 1024))
        return out

    def ingest_path(self, path: str | Path) -> dict:
        ocr_cfg = self.cfg.get("ocr", {})
        doc = extract_document(
            path,
            ocr_enabled=ocr_cfg.get("enabled", True),
            dpi=ocr_cfg.get("dpi", 200),
        )
        res = self.ingest_text(Path(path).name, doc["text"], source=str(path))
        res["ocr_pages"] = doc.get("ocr_pages", 0)
        res["method"] = doc.get("method")
        return res

    def ingest_text(self, filename: str, text: str, source: str | None = None) -> dict:
        raw_chunks = chunk_text(
            text,
            chunk_size=self.cfg["rag"]["chunk_size"],
            overlap=self.cfg["rag"]["chunk_overlap"],
        )
        chunks = [c for c in raw_chunks if self._is_embeddable(c)]
        if not chunks:
            return {"doc_id": None, "n_chunks": 0, "filename": filename}

        d_id = doc_id_for(filename, text)
        chunk_ids = [chunk_id_for(d_id, i) for i in range(len(chunks))]
        embeddings = self._embed(chunks)

        existing = set(self.collection.get(ids=chunk_ids).get("ids") or [])
        if existing:
            self.collection.delete(ids=list(existing))

        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"doc_id": d_id, "filename": filename, "idx": i} for i in range(len(chunks))],
        )

        self.storage.add_document(d_id, filename, len(chunks))
        self.storage.add_chunks([
            (cid, d_id, content, source or filename)
            for cid, content in zip(chunk_ids, chunks)
        ])
        return {"doc_id": d_id, "n_chunks": len(chunks), "filename": filename}

    def delete_document(self, doc_id: str) -> None:
        got = self.collection.get(where={"doc_id": doc_id})
        ids = got.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        self.storage.delete_document(doc_id)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.cfg["rag"]["top_k"]
        min_sim = self.cfg["rag"].get("min_similarity", 0.0)
        reward_w = self.cfg["rag"].get("reward_weight", 0.3)

        if self.collection.count() == 0:
            return []

        emb = self._embed([query])
        if not emb:
            return []

        res = self.collection.query(
            query_embeddings=emb,
            n_results=max(k * 3, k),
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        reward_map = self.storage.get_chunk_rewards(ids)

        ranked: list[dict] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            sim = 1.0 - float(dist)
            if sim < min_sim:
                continue
            reward = reward_map.get(cid, 0.0)
            boost = reward_w * math.tanh(reward / 3.0)
            ranked.append({
                "chunk_id": cid,
                "content": doc,
                "metadata": meta or {},
                "similarity": sim,
                "reward_score": reward,
                "final_score": sim + boost,
            })

        ranked.sort(key=lambda r: r["final_score"], reverse=True)
        top = ranked[:k]
        self.storage.bump_chunk_usage([r["chunk_id"] for r in top])
        return top

    def build_context_block(self, hits: list[dict]) -> str:
        if not hits:
            return "（暂无相关材料,请告知用户基础模型可能不掌握专门信息）"
        parts = []
        for i, h in enumerate(hits, 1):
            src = (h.get("metadata") or {}).get("filename", "unknown")
            parts.append(f"[材料{i} · {src}]\n{h['content']}")
        return "\n\n".join(parts)
