import hashlib
import re
from pathlib import Path


def _extract_pdf(path: Path) -> tuple[str, str]:
    """Try multiple PDF backends in order. Returns (text, backend_used)."""
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            text = "\n\n".join((p.extract_text() or "") for p in pdf.pages)
        if text.strip():
            return text, "pdfplumber"
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        if text.strip():
            return text, "pypdf"
    except Exception:
        pass
    return "", "none"


def parse_file(path: str | Path) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".txt", ".md", ".markdown"):
        return p.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        text, backend = _extract_pdf(p)
        if not text.strip():
            raise ValueError(
                "PDF 无法抽取文字 — 大概率是扫描版(图片型)。"
                "请先用 OCR 工具(Adobe Acrobat / WPS / ocrmypdf)转成文字版,再上传。"
            )
        return text
    if ext in (".docx",):
        import docx
        d = docx.Document(str(p))
        return "\n".join(par.text for par in d.paragraphs)
    raise ValueError(f"不支持的文件类型: {ext}")


_SPLIT_RE = re.compile(r"(?<=[。！？!?\.\n])\s+")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SPLIT_RE.split(text) if s.strip()]
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= chunk_size:
            buf = (buf + " " + s).strip() if buf else s
        else:
            if buf:
                chunks.append(buf)
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                buf = (tail + " " + s).strip()
            else:
                buf = s
    if buf:
        chunks.append(buf)
    return chunks


def doc_id_for(filename: str, content: str) -> str:
    h = hashlib.sha1()
    h.update(filename.encode("utf-8"))
    h.update(b"::")
    h.update(content[:4096].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def chunk_id_for(doc_id: str, idx: int) -> str:
    return f"{doc_id}:{idx:04d}"
