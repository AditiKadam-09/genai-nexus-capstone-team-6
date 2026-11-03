"""Tests for document chunking, section building, and overlap packing."""

import pytest
from ingestion.chunker import (
    build_sections,
    split_table,
    recursive_split,
    pack_with_overlap,
    DocumentChunker,
)


def test_recursive_split_respects_size():
    long_text = "Sentence one. " * 50  # ~700 chars
    pieces = recursive_split(long_text, chunk_size=200)

    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)


def test_pack_with_overlap():
    pieces = [
        "First section with important details.",
        "Second section expanding upon the rules.",
        "Third section finalizing the cancellation policy.",
    ]
    packed = pack_with_overlap(pieces, chunk_size=80, chunk_overlap=20)
    assert len(packed) >= 2


def test_split_table_preserves_headers():
    headers = ["Class", "Cost"]
    rows = [[f"Class_{i}", f"Rs_{i*50}"] for i in range(20)]

    chunks = split_table(headers, rows, chunk_size=150)
    assert len(chunks) > 1

    # Every chunk must have the table headers
    for c in chunks:
        assert "| Class | Cost |" in c
        assert "|---|---|" in c


def test_document_chunker_metadata():
    items = [
        {"type": "heading", "text": "Cancellation Rules", "page": 2},
        {"type": "paragraph", "text": "Cancellation before 48 hours is subject to minimum clerkage.", "page": 2},
    ]

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_document("irctc_refund_rules_jan2026.pdf", items)

    assert len(chunks) == 1
    c = chunks[0]
    assert c["doc"] == "irctc_refund_rules_jan2026.pdf"
    assert c["page"] == 2
    assert "Cancellation Rules" in c["section_title"]
    assert "Document: " in c["embedding_text"]
    assert "Section: " in c["embedding_text"]
    assert c["metadata"]["is_html"] is False


def test_document_chunker_html_page_none():
    items = [
        {"type": "heading", "text": "HTML Rules", "page": None},
        {"type": "paragraph", "text": "General refund rules text.", "page": None},
    ]

    chunker = DocumentChunker()
    chunks = chunker.chunk_document("indianrail_refund_rules_2015.html", items)

    assert len(chunks) == 1
    c = chunks[0]
    assert c["page"] is None
    assert c["metadata"]["is_html"] is True

