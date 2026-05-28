"""Layer 3: user feedback feeds reward into Layer 2.

When a user marks an answer as "clearly understood", every retrieved chunk that
helped produce that answer gets its reward_score bumped. Future retrievals
re-rank by similarity + reward, so the materials that consistently lead to clear
answers float to the top.
"""
from __future__ import annotations

from .storage import Storage


class FeedbackLayer:
    def __init__(self, storage: Storage, reward_delta: float = 1.0):
        self.storage = storage
        self.reward_delta = reward_delta

    def record(self, conversation_id: int, marked_clear: bool) -> dict:
        conv = self.storage.get_conversation(conversation_id)
        if not conv:
            return {"ok": False, "reason": "conversation_not_found"}

        self.storage.record_feedback(conversation_id, marked_clear)
        rewarded: list[str] = []
        if marked_clear:
            chunk_ids = conv.get("retrieved_chunk_ids") or []
            if chunk_ids:
                self.storage.reward_chunks(chunk_ids, delta=self.reward_delta)
                rewarded = chunk_ids
        return {"ok": True, "rewarded_chunks": rewarded}
