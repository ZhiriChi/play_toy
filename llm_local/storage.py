import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    uploaded_at REAL NOT NULL,
    n_chunks INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    reward_score REAL NOT NULL DEFAULT 0,
    n_used INTEGER NOT NULL DEFAULT 0,
    n_rewarded INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    avg_entropy REAL,
    cross_entropy REAL,
    retrieved_chunk_ids TEXT
);

CREATE TABLE IF NOT EXISTS token_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    token TEXT NOT NULL,
    chosen_logprob REAL,
    entropy REAL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    marked_clear INTEGER NOT NULL,
    ts REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversations(ts);
CREATE INDEX IF NOT EXISTS idx_tok_conv ON token_logs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class Storage:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_document(self, doc_id: str, filename: str, n_chunks: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO documents (id, filename, uploaded_at, n_chunks) VALUES (?, ?, ?, ?)",
                (doc_id, filename, time.time(), n_chunks),
            )

    def add_chunks(self, rows: list[tuple[str, str, str, str]]) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO chunks (id, doc_id, content, source) VALUES (?, ?, ?, ?)",
                rows,
            )

    def list_documents(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, filename, uploaded_at, n_chunks FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def get_chunk_rewards(self, chunk_ids: list[str]) -> dict[str, float]:
        if not chunk_ids:
            return {}
        with self._conn() as c:
            placeholders = ",".join("?" * len(chunk_ids))
            rows = c.execute(
                f"SELECT id, reward_score FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        return {r["id"]: r["reward_score"] for r in rows}

    def bump_chunk_usage(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        with self._conn() as c:
            c.executemany(
                "UPDATE chunks SET n_used = n_used + 1 WHERE id = ?",
                [(cid,) for cid in chunk_ids],
            )

    def reward_chunks(self, chunk_ids: list[str], delta: float = 1.0) -> None:
        if not chunk_ids:
            return
        with self._conn() as c:
            c.executemany(
                "UPDATE chunks SET reward_score = reward_score + ?, n_rewarded = n_rewarded + 1 WHERE id = ?",
                [(delta, cid) for cid in chunk_ids],
            )

    def log_conversation(
        self,
        query: str,
        response: str,
        latency_ms: float,
        avg_entropy: float | None,
        cross_entropy: float | None,
        retrieved_chunk_ids: list[str],
        token_rows: list[tuple[int, str, float | None, float | None]],
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO conversations
                   (ts, query, response, latency_ms, avg_entropy, cross_entropy, retrieved_chunk_ids)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    query,
                    response,
                    latency_ms,
                    avg_entropy,
                    cross_entropy,
                    json.dumps(retrieved_chunk_ids),
                ),
            )
            cid = cur.lastrowid
            if token_rows:
                c.executemany(
                    "INSERT INTO token_logs (conversation_id, position, token, chosen_logprob, entropy) VALUES (?, ?, ?, ?, ?)",
                    [(cid, p, t, lp, e) for (p, t, lp, e) in token_rows],
                )
        return cid

    def get_conversation(self, conv_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["retrieved_chunk_ids"] = json.loads(d.get("retrieved_chunk_ids") or "[]")
            d["tokens"] = [
                dict(r) for r in c.execute(
                    "SELECT position, token, chosen_logprob, entropy FROM token_logs WHERE conversation_id = ? ORDER BY position",
                    (conv_id,),
                ).fetchall()
            ]
        return d

    def recent_conversations(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, ts, query, latency_ms, avg_entropy, cross_entropy FROM conversations ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def conversations_since(self, since_ts: float) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, ts, query, response, latency_ms, avg_entropy, cross_entropy FROM conversations WHERE ts >= ? ORDER BY ts",
                (since_ts,),
            ).fetchall()
        return [dict(r) for r in rows]

    def record_feedback(self, conv_id: int, marked_clear: bool) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO feedback (conversation_id, marked_clear, ts) VALUES (?, ?, ?)",
                (conv_id, 1 if marked_clear else 0, time.time()),
            )

    def feedback_stats(self, since_ts: float | None = None) -> dict:
        with self._conn() as c:
            where = "WHERE ts >= ?" if since_ts else ""
            args = (since_ts,) if since_ts else ()
            row = c.execute(
                f"SELECT COUNT(*) AS n, COALESCE(SUM(marked_clear), 0) AS n_clear FROM feedback {where}",
                args,
            ).fetchone()
        return {"n": row["n"], "n_clear": row["n_clear"]}

    def top_chunks_by_reward(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT c.id, c.content, c.source, c.reward_score, c.n_used, c.n_rewarded, d.filename
                   FROM chunks c LEFT JOIN documents d ON c.doc_id = d.id
                   WHERE c.n_used > 0
                   ORDER BY c.reward_score DESC, c.n_rewarded DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
