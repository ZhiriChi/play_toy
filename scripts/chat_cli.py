"""CLI: interactive chat with the pipeline.

Usage:
    python scripts/chat_cli.py
Commands inside the prompt:
    /quit            exit
    /clear N         mark conversation N as 'clearly understood' (reward Layer 2)
    /clear           mark last conversation
    /docs            list ingested docs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_local import Pipeline  # noqa: E402


def _fmt(x: float | None, spec: str = ".2f") -> str:
    if x is None:
        return "—"
    return format(x, spec)


def main() -> int:
    pipe = Pipeline()
    print("本地 LLM 已就绪。/quit 退出。")
    last_id: int | None = None
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q == "/quit":
            return 0
        if q == "/docs":
            for d in pipe.list_documents():
                print(f"  {d['filename']}  ({d['n_chunks']} chunks)  id={d['id']}")
            continue
        if q.startswith("/clear"):
            parts = q.split()
            cid = int(parts[1]) if len(parts) > 1 else last_id
            if cid is None:
                print("(没有可标记的对话)")
                continue
            r = pipe.mark_clear(cid, True)
            print(f"  → 已奖励 {len(r.get('rewarded_chunks', []))} 个材料块。")
            continue
        try:
            out = pipe.ask(q)
        except Exception as e:
            print(f"[错误] {e}")
            continue
        last_id = out["conversation_id"]
        print(f"\n{out['response']}")
        print(
            f"\n  [#{out['conversation_id']}  avg_H={_fmt(out['avg_entropy'])}  "
            f"CE={_fmt(out['cross_entropy'])}  latency={out['latency_ms']:.0f}ms  "
            f"hits={len(out['hits'])}]"
        )


if __name__ == "__main__":
    sys.exit(main())
