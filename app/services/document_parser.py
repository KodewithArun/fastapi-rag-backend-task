import os
import re
import tempfile
from typing import BinaryIO

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document


def _temp_write(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _normalize_pdf_text(text: str) -> str:
    """Reading order (sort=False) + line-wise collapse fixes column-gap space runs."""
    text = text.replace("\ufffd", "•")
    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line.strip())
        if line:
            lines.append(line)
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _apply_source(docs: list[Document], source: str | None) -> list[Document]:
    if source is None:
        return docs
    for doc in docs:
        doc.metadata["source"] = source
    return docs


def load_documents(
    content_type: str,
    file: BinaryIO,
    *,
    source: str | None = None,
) -> list[Document]:
    """Load the upload from a temp file via LangChain loaders.

    PDF: `PyMuPDFLoader` in single-document mode, then replacement-char fix and
    whitespace normalization.
    """
    if content_type == "text/plain":
        suffix = ".txt"
    elif content_type == "application/pdf":
        suffix = ".pdf"
    else:
        raise ValueError(
            f"Unsupported document format: {content_type}. Only .txt and .pdf are allowed."
        )

    path = _temp_write(file.read(), suffix)
    try:
        if content_type == "text/plain":
            docs = TextLoader(path, encoding="utf-8").load()
        else:
            docs = PyMuPDFLoader(path, mode="single").load()
            for d in docs:
                d.page_content = _normalize_pdf_text(d.page_content)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    return _apply_source(docs, source)
