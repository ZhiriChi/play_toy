"""Monitoring + analytics built on top of SQLite logs.

Per-query: entropy/cross-entropy already stored in `conversations` and `token_logs`.
Weekly: topic clustering on past queries + token frequency + behavioral drift.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

from .layer1_base import OllamaClient
from .storage import Storage

_CN_STOP = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "和", "与", "或", "也", "都",
    "就", "还", "但", "而", "如", "如果", "因为", "所以", "什么", "怎么", "怎样", "为什么",
    "可以", "需要", "应该", "请", "帮", "我们", "你们", "他们", "这个", "那个", "这些",
    "那些", "什么样", "如何", "哪些", "哪个", "一个", "一些", "没有", "不是", "知道",
    "告诉", "说明", "解释",
}
_EN_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "of", "to", "in", "on", "at",
    "for", "with", "as", "by", "from", "this", "that", "these", "those", "it", "its",
    "what", "how", "why", "when", "where", "who", "which", "i", "you", "we", "they",
    "he", "she", "can", "could", "should", "would", "will", "would", "if", "so",
    "about", "into", "than", "then", "your", "my", "our", "tell", "explain", "please",
}

_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}|[一-鿿]{2,3}")


def extract_tokens(text: str) -> list[str]:
    raw = _TOKEN_RE.findall(text.lower())
    out = []
    for tok in raw:
        if tok in _EN_STOP or tok in _CN_STOP:
            continue
        out.append(tok)
    return out


def week_ago_ts(weeks: int = 1) -> float:
    return time.time() - weeks * 7 * 86400


def top_tokens(storage: Storage, since_ts: float | None = None, n: int = 20) -> list[tuple[str, int]]:
    convs = storage.conversations_since(since_ts or 0)
    counter: Counter[str] = Counter()
    for c in convs:
        counter.update(extract_tokens(c["query"]))
    return counter.most_common(n)


def cluster_topics(
    storage: Storage,
    client: OllamaClient,
    embed_model: str,
    since_ts: float | None = None,
    max_k: int = 5,
) -> list[dict]:
    """KMeans on past query embeddings; label each cluster with its top tokens."""
    from sklearn.cluster import KMeans

    convs = storage.conversations_since(since_ts or 0)
    queries = [c["query"] for c in convs if c.get("query")]
    if len(queries) < 3:
        return []

    embeddings = client.embed(queries, model=embed_model)
    X = np.array(embeddings, dtype=np.float32)
    k = max(2, min(max_k, len(queries) // 4 + 1))
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(X)

    clusters: list[dict] = []
    for cid in range(k):
        members = [q for q, lab in zip(queries, labels) if lab == cid]
        if not members:
            continue
        tok_counter: Counter[str] = Counter()
        for q in members:
            tok_counter.update(extract_tokens(q))
        top = [t for t, _ in tok_counter.most_common(4)]
        label = " · ".join(top) if top else f"topic {cid}"
        clusters.append({
            "id": cid,
            "label": label,
            "count": len(members),
            "samples": members[:3],
        })
    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters


def daily_volume(storage: Storage, days: int = 14) -> list[dict]:
    since = time.time() - days * 86400
    convs = storage.conversations_since(since)
    buckets: dict[str, int] = {}
    for c in convs:
        day = datetime.fromtimestamp(c["ts"]).strftime("%Y-%m-%d")
        buckets[day] = buckets.get(day, 0) + 1
    out = []
    for i in range(days):
        d = (datetime.now() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        out.append({"date": d, "count": buckets.get(d, 0)})
    return out


def quality_metrics(storage: Storage, since_ts: float | None = None) -> dict:
    convs = storage.conversations_since(since_ts or 0)
    n = len(convs)
    if n == 0:
        return {"n": 0}

    entropies = [c["avg_entropy"] for c in convs if c.get("avg_entropy") is not None]
    ces = [c["cross_entropy"] for c in convs if c.get("cross_entropy") is not None]
    lats = [c["latency_ms"] for c in convs if c.get("latency_ms") is not None]

    fb = storage.feedback_stats(since_ts)
    clarity = (fb["n_clear"] / fb["n"]) if fb["n"] else None

    return {
        "n": n,
        "avg_entropy": float(np.mean(entropies)) if entropies else None,
        "p90_entropy": float(np.percentile(entropies, 90)) if entropies else None,
        "avg_cross_entropy": float(np.mean(ces)) if ces else None,
        "avg_latency_ms": float(np.mean(lats)) if lats else None,
        "feedback_n": fb["n"],
        "clarity_rate": clarity,
    }


def by_model_breakdown(storage: Storage, since_ts: float | None = None) -> list[dict]:
    """Per-model: count, avg entropy, avg cross-entropy, avg latency."""
    convs = storage.conversations_since(since_ts or 0)
    buckets: dict[str, list[dict]] = {}
    for c in convs:
        m = c.get("model") or "unknown"
        buckets.setdefault(m, []).append(c)

    out: list[dict] = []
    for m, rows in buckets.items():
        ents = [r["avg_entropy"] for r in rows if r.get("avg_entropy") is not None]
        ces = [r["cross_entropy"] for r in rows if r.get("cross_entropy") is not None]
        lats = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        out.append({
            "model": m,
            "n": len(rows),
            "avg_entropy": float(np.mean(ents)) if ents else None,
            "avg_cross_entropy": float(np.mean(ces)) if ces else None,
            "avg_latency_ms": float(np.mean(lats)) if lats else None,
        })
    out.sort(key=lambda r: r["n"], reverse=True)
    return out


def high_uncertainty_segments(conv: dict, top_pct: float = 0.2) -> list[dict]:
    """Return tokens whose entropy is in the top X% — useful for highlighting
    where the model was least confident.
    """
    tokens = conv.get("tokens") or []
    ents = [t["entropy"] for t in tokens if t.get("entropy") is not None]
    if not ents:
        return []
    cutoff = float(np.quantile(ents, 1 - top_pct))
    return [t for t in tokens if (t.get("entropy") or 0) >= cutoff]
