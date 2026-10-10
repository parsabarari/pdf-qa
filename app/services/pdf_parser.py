"""Step 1-2 of the pipeline: PDF bytes -> cleaned text, page by page."""

import logging
import re
from dataclasses import dataclass

import pymupdf

logger = logging.getLogger(__name__)

MIN_TOTAL_TEXT_CHARS = 20  # below this we consider the PDF "no usable text" (e.g. a scan)


class PDFParseError(Exception):
    """The file could not be opened as a PDF (malformed, encrypted, zero pages)."""


class NoExtractableTextError(Exception):
    """The PDF is valid but contains no (or almost no) extractable text."""


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based, matches what a human sees in a PDF viewer
    text: str


@dataclass(frozen=True)
class ParsedPDF:
    num_pages: int  # total pages, including empty ones
    pages: list[PageText]  # only pages that contain text


def clean_text(text: str) -> str:
    """Light, deterministic cleanup. No NLP."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\f\v]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)  # strip spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()


def parse_pdf(data: bytes) -> ParsedPDF:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFParseError("File could not be opened as a PDF") from exc

    with doc:
        if doc.needs_pass:
            raise PDFParseError("Password-protected PDFs are not supported")
        if doc.page_count == 0:
            raise PDFParseError("PDF has no pages")

        pages: list[PageText] = []
        for index in range(doc.page_count):
            try:
                raw = doc.load_page(index).get_text("text")
            except Exception:
                logger.warning("page_extract_failed page=%d", index + 1)
                continue
            text = clean_text(raw)
            if text:
                pages.append(PageText(page=index + 1, text=text))
        num_pages = doc.page_count

    if sum(len(p.text) for p in pages) < MIN_TOTAL_TEXT_CHARS:
        raise NoExtractableTextError("PDF contains no extractable text (is it a scanned document?)")

    return ParsedPDF(num_pages=num_pages, pages=pages)
