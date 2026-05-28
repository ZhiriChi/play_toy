"""CLI: ingest documents into the vector store.

Usage:
    python scripts/ingest_cli.py path1 [path2 ...]
    python scripts/ingest_cli.py ./my_docs    # directory: ingests recognized files
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_local import Pipeline  # noqa: E402

SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def main(args: list[str]) -> int:
    if not args:
        print(__doc__)
        return 1
    pipe = Pipeline()
    files: list[Path] = []
    for raw in args:
        p = Path(raw)
        if p.is_dir():
            files.extend(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
        elif p.is_file():
            files.append(p)
        else:
            print(f"跳过不存在的路径: {p}")
    for f in files:
        try:
            info = pipe.ingest(f)
            print(f"[OK] {f.name}: {info['n_chunks']} chunks  doc_id={info['doc_id']}")
        except Exception as e:
            print(f"[ERR] {f.name}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
