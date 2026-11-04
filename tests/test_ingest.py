"""Tests for helpdesk.ingest module exports and pipeline interface."""

import pytest
from pathlib import Path

from helpdesk.ingest import (
    PDFExtractor,
    HTMLExtractor,
    DocumentChunker,
    DocumentEmbedder,
    ChromaIndexer,
    IngestionPipeline,
    clean_text,
    table_to_text,
    clean_items,
    ingest_documents,
)


def test_helpdesk_ingest_exports():
    """Verify all required classes and functions are exported by helpdesk.ingest."""
    assert PDFExtractor is not None
    assert HTMLExtractor is not None
    assert DocumentChunker is not None
    assert DocumentEmbedder is not None
    assert ChromaIndexer is not None
    assert IngestionPipeline is not None
    assert callable(clean_text)
    assert callable(table_to_text)
    assert callable(clean_items)
    assert callable(ingest_documents)


def test_cleaner_functionality():
    """Verify clean_text properly cleans artifacts."""
    raw = "Refund\u00a0rules   cancel   ticket.\r\n"
    cleaned = clean_text(raw)
    assert cleaned == "Refund rules cancel ticket."


def test_chunker_basic():
    """Verify DocumentChunker chunks text properly."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    items = [
        {"type": "heading", "level": 1, "text": "Cancellation Fees", "page": 1},
        {"type": "paragraph", "text": "Flat cancellation charge of Rs 240 applies for AC First Class tickets cancelled 48 hours before departure.", "page": 1},
    ]
    chunks = chunker.chunk_document(doc_name="test.pdf", items=items)
    assert len(chunks) >= 1
    assert chunks[0]["doc"] == "test.pdf"
    assert chunks[0]["page"] == 1
    assert "Cancellation Fees" in chunks[0]["section_title"]
