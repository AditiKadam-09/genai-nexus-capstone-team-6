"""Document Ingestion & Index Building Module for Railway Help Desk.

Provides end-to-end extraction (Docling + RapidOCR for scanned PDFs, BeautifulSoup for HTML),
structure-preserving text cleaning and footnote binding, hierarchical contextual chunking,
dense embeddings with BAAI/bge-small-en-v1.5, and persistent ChromaDB indexing.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from helpdesk.ingestion.config import (
    DOCS_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    DOC_METADATA_MAP,
)
from helpdesk.ingestion.cleaner import clean_text, table_to_text, clean_items
from helpdesk.ingestion.pdf_extractor import PDFExtractor
from helpdesk.ingestion.html_extractor import HTMLExtractor
from helpdesk.ingestion.chunker import (
    DocumentChunker,
    build_sections,
    split_table,
    recursive_split,
    pack_with_overlap,
)
from helpdesk.ingestion.embedder import DocumentEmbedder
from helpdesk.ingestion.indexer import ChromaIndexer
from helpdesk.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

__all__ = [
    "PDFExtractor",
    "HTMLExtractor",
    "clean_text",
    "table_to_text",
    "clean_items",
    "DocumentChunker",
    "build_sections",
    "split_table",
    "recursive_split",
    "pack_with_overlap",
    "DocumentEmbedder",
    "ChromaIndexer",
    "IngestionPipeline",
    "ingest_documents",
    "DOCS_DIR",
    "CHROMA_PERSIST_DIR",
    "CHROMA_COLLECTION_NAME",
]


def ingest_documents(
    docs_dir: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    reindex: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedder: Optional[DocumentEmbedder] = None,
) -> Dict[str, Any]:
    """High-level function to ingest documents and build the persistent vector index.

    Args:
        docs_dir: Directory containing railway circulars and HTML rules (defaults to configured DOCS_DIR).
        persist_dir: Storage directory for ChromaDB (defaults to configured CHROMA_PERSIST_DIR).
        collection_name: ChromaDB collection name.
        reindex: Whether to clear the collection before re-indexing.
        chunk_size: Max character chunk size.
        chunk_overlap: Overlap between consecutive chunks.
        embedder: Optional DocumentEmbedder instance.

    Returns:
        Summary dictionary with indexing results.
    """
    pipeline = IngestionPipeline(
        docs_dir=docs_dir or DOCS_DIR,
        persist_dir=persist_dir or CHROMA_PERSIST_DIR,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedder=embedder,
    )
    return pipeline.run(reindex=reindex)


if __name__ == "__main__":
    import argparse
    import sys

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ingest documents and build vector index.")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--persist-dir", type=Path, default=CHROMA_PERSIST_DIR)
    parser.add_argument("--collection", type=str, default=CHROMA_COLLECTION_NAME)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ingest_documents(
        docs_dir=args.docs_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        reindex=args.reindex,
    )
