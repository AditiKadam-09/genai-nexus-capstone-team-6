"""Tests for persistent ChromaDB indexing."""

import pytest
import numpy as np
from pathlib import Path

from ingestion.indexer import ChromaIndexer


def test_indexer_lifecycle(tmp_path: Path):
    indexer = ChromaIndexer(persist_dir=tmp_path / "test_chroma", collection_name="test_col")
    assert indexer.count() == 0

    chunks = [
        {
            "chunk_id": "test_0001",
            "doc": "test_doc.pdf",
            "page": 1,
            "text": "Rule for advance reservations.",
            "section_title": "Advance Reservation",
            "section_path": "Doc > Advance Reservation",
            "split_method": "whole_section",
        },
        {
            "chunk_id": "test_0002",
            "doc": "test_doc.html",
            "page": None,
            "text": "Refund clerkage rules.",
            "section_title": "Refunds",
            "section_path": "Doc > Refunds",
            "split_method": "whole_section",
        },
    ]

    embeddings = np.random.randn(2, 384).astype(np.float32)

    # First index
    indexed = indexer.index_chunks(chunks, embeddings)
    assert indexed == 2
    assert indexer.count() == 2

    # Test idempotency (re-indexing same chunk IDs should not duplicate)
    indexed_repeat = indexer.index_chunks(chunks, embeddings)
    assert indexed_repeat == 2
    assert indexer.count() == 2

    # Clear collection
    indexer.clear()
    assert indexer.count() == 0


def test_indexer_skips_empty_chunks(tmp_path: Path):
    indexer = ChromaIndexer(persist_dir=tmp_path / "test_chroma2", collection_name="test_col2")
    chunks = [
        {"chunk_id": "c1", "doc": "d1.pdf", "page": 1, "text": "Valid text"},
        {"chunk_id": "c2", "doc": "d1.pdf", "page": 1, "text": "   "},  # empty
    ]
    embeddings = np.random.randn(2, 384).astype(np.float32)

    indexed = indexer.index_chunks(chunks, embeddings)
    assert indexed == 1
    assert indexer.count() == 1

