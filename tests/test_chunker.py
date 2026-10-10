import pytest

from app.services.chunker import chunk_pages, split_text
from app.services.pdf_parser import PageText, clean_text


def test_short_text_is_one_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n  ", 100, 10) == []


def test_chunks_respect_size_and_are_never_empty():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, 100, 20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)
    assert all(len(c) <= 100 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = split_text(text, 100, 30)
    # the end of one chunk reappears at the start of the next
    tail_words = chunks[0].split()[-2:]
    assert all(w in chunks[1] for w in tail_words)


def test_chunking_is_deterministic():
    text = "Sentence one. " * 100
    assert split_text(text, 120, 20) == split_text(text, 120, 20)


def test_all_text_is_covered():
    words = [f"w{i}" for i in range(300)]
    chunks = split_text(" ".join(words), 100, 20)
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        split_text("abc " * 100, 50, 50)


def test_chunk_pages_keeps_page_numbers_and_global_index():
    pages = [PageText(1, "alpha " * 60), PageText(3, "beta " * 10)]
    chunks = chunk_pages(pages, 100, 20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].page == 1
    assert chunks[-1].page == 3
    assert {c.page for c in chunks} == {1, 3}


def test_clean_text():
    assert clean_text("a  b\t c\n\n\n\nd inter-\nnational") == "a b c\n\nd international"
