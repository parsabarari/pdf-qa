"""Step 3 of the pipeline: split page text into overlapping, deterministic chunks."""

from dataclasses import dataclass

from app.services.pdf_parser import PageText

# Preferred places to cut, best first. We only look in the second half of the window,
# so a chunk is never shorter than roughly chunk_size / 2 (except the last one).
_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    chunk_index: int  # global index within the document
    page: int  # 1-based page the chunk came from
    text: str


def _find_cut(text: str, start: int, end: int, chunk_size: int) -> int:
    low = start + chunk_size // 2
    for sep in _SEPARATORS:
        idx = text.rfind(sep, low, end)
        if idx != -1:
            return idx + len(sep)
    return end  # no natural boundary: hard cut


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    n = len(text)
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _find_cut(text, start, end, chunk_size)

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break

        # Step back by `chunk_overlap`, but always make real progress.
        next_start = max(end - chunk_overlap, start + chunk_size // 4)
        # Avoid starting in the middle of a word (look ahead a few characters only).
        for offset in range(min(40, end - next_start)):
            if text[next_start + offset - 1].isspace():
                next_start += offset
                break
        start = next_start
    return chunks


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk each page separately (so every chunk has exactly one page number)."""
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(Chunk(chunk_index=len(chunks), page=page.page, text=piece))
    return chunks
