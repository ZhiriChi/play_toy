import hashlib
import re
from pathlib import Path

# Lazy module-level singleton so the (heavyish) OCR model loads only once.
_OCR_ENGINE = None
_OCR_LOAD_FAILED = False


def _get_ocr_engine():
    """Return a cached RapidOCR engine, or None if the dependency is missing."""
    global _OCR_ENGINE, _OCR_LOAD_FAILED
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    if _OCR_LOAD_FAILED:
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
        return _OCR_ENGINE
    except Exception:
        _OCR_LOAD_FAILED = True
        return None


def ocr_available() -> bool:
    return _get_ocr_engine() is not None


def _ocr_png_bytes(png_bytes: bytes) -> str:
    engine = _get_ocr_engine()
    if engine is None:
        return ""
    import cv2
    import numpy as np

    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    result, _ = engine(img)
    if not result:
        return ""
    # RapidOCR returns [[box, text, score], ...]
    return "\n".join(item[1] for item in result)


def _extract_pdf_pymupdf(path: Path, ocr_enabled: bool, dpi: int) -> tuple[str, int]:
    """Per-page: use the text layer; OCR pages that have none. Returns (text, ocr_pages)."""
    import fitz  # PyMuPDF

    parts: list[str] = []
    ocr_pages = 0
    doc = fitz.open(str(path))
    try:
        for page in doc:
            txt = (page.get_text() or "").strip()
            if txt:
                parts.append(txt)
            elif ocr_enabled and ocr_available():
                try:
                    pix = page.get_pixmap(dpi=dpi)
                    page_txt = _ocr_png_bytes(pix.tobytes("png"))
                except Exception:
                    page_txt = ""
                if page_txt.strip():
                    parts.append(page_txt)
                    ocr_pages += 1
    finally:
        doc.close()
    return "\n\n".join(parts), ocr_pages


def _extract_pdf_legacy(path: Path) -> str:
    """Text-layer-only fallback when PyMuPDF isn't installed (no OCR)."""
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            text = "\n\n".join((p.extract_text() or "") for p in pdf.pages)
        if text.strip():
            return text
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        if text.strip():
            return text
    except Exception:
        pass
    return ""


def extract_document(path: str | Path, ocr_enabled: bool = True, dpi: int = 200) -> dict:
    """Extract text from a file. Returns {text, method, ocr_pages}.

    For PDFs: try PyMuPDF's text layer first, OCR any image-only pages with
    RapidOCR, and fall back to pdfplumber/pypdf if PyMuPDF is unavailable.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".txt", ".md", ".markdown"):
        return {"text": p.read_text(encoding="utf-8", errors="ignore"), "method": "text", "ocr_pages": 0}

    if ext == ".docx":
        import docx
        d = docx.Document(str(p))
        return {"text": "\n".join(par.text for par in d.paragraphs), "method": "docx", "ocr_pages": 0}

    if ext == ".pdf":
        text, ocr_pages, method = "", 0, "none"
        try:
            text, ocr_pages = _extract_pdf_pymupdf(p, ocr_enabled, dpi)
            method = "pymupdf+ocr" if ocr_pages else "pymupdf"
        except ImportError:
            text = _extract_pdf_legacy(p)
            method = "legacy"
        if not text.strip():
            if ocr_enabled and not ocr_available():
                raise ValueError(
                    "PDF 无法抽取文字 — 看起来是扫描版(图片型),而 OCR 引擎未安装。"
                    "请运行:pip install pymupdf rapidocr-onnxruntime  之后重新上传,即可自动识别。"
                )
            raise ValueError(
                "PDF 无法抽取文字 — 扫描版且 OCR 也未识别出内容。"
                "请确认文件不是加密/空白扫描件。"
            )
        return {"text": text, "method": method, "ocr_pages": ocr_pages}

    raise ValueError(f"不支持的文件类型: {ext}")


def parse_file(path: str | Path) -> str:
    """Backward-compatible wrapper returning only the extracted text."""
    return extract_document(path)["text"]


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
