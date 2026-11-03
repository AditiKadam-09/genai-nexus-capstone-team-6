"""End-to-end ingestion pipeline orchestrator.

Coordinates extraction (Docling + RapidOCR / HTML), cleaning,
chunking, embedding (BGE-small), and persistent indexing in ChromaDB.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ingestion.config import (
    DOCUMENTS_DIR,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from ingestion.cleaner import clean_items
from ingestion.pdf_extractor import PDFExtractor
from ingestion.html_extractor import HTMLExtractor
from ingestion.chunker import DocumentChunker
from ingestion.embedder import DocumentEmbedder
from ingestion.indexer import ChromaIndexer

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates document ingestion and index building."""

    def __init__(
        self,
        docs_dir: Path = DOCUMENTS_DIR,
        persist_dir: Path = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.docs_dir = Path(docs_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.pdf_extractor = PDFExtractor(use_docling=True, use_ocr=True)
        self.html_extractor = HTMLExtractor()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = DocumentEmbedder(model_name=model_name)
        self.indexer = ChromaIndexer(persist_dir=persist_dir, collection_name=collection_name)

    def run(self, reindex: bool = False) -> Dict[str, Any]:
        """Run the complete ingestion pipeline on all documents in docs_dir.

        Args:
            reindex: If True, clears existing collection and re-indexes all documents.

        Returns:
            Dictionary containing execution statistics and chunk counts.
        """
        start_time = time.time()
        logger.info(f"Starting document ingestion pipeline from: {self.docs_dir}")

        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

        if reindex:
            logger.info("Reindex requested. Clearing existing ChromaDB collection...")
            self.indexer.clear()

        # Gather document files
        pdf_files = sorted(list(self.docs_dir.glob("*.pdf")))
        html_files = sorted(list(self.docs_dir.glob("*.html")))
        all_files = pdf_files + html_files

        if not all_files:
            logger.warning(f"No PDF or HTML files found in {self.docs_dir}")
            return {"status": "warning", "message": "No documents found", "total_chunks": 0}

        logger.info(f"Discovered {len(all_files)} documents ({len(pdf_files)} PDFs, {len(html_files)} HTML files)")

        all_chunks: List[Dict[str, Any]] = []
        doc_stats: Dict[str, Dict[str, Any]] = {}

        for file_path in all_files:
            doc_name = file_path.name
            logger.info(f"--- Processing {doc_name} ---")
            t0 = time.time()

            try:
                # 1. Extraction
                if file_path.suffix.lower() == ".pdf":
                    raw_items = self.pdf_extractor.extract(file_path)
                elif file_path.suffix.lower() == ".html":
                    raw_items = self.html_extractor.extract(file_path)
                else:
                    logger.warning(f"Skipping unsupported file format: {file_path}")
                    continue

                if not raw_items:
                    logger.warning(f"No items extracted from {doc_name}. Skipping.")
                    doc_stats[doc_name] = {"items": 0, "chunks": 0, "status": "empty"}
                    continue

                # 2. Cleaning and footnote binding
                cleaned_items = clean_items(raw_items)

                # 3. Chunking
                doc_chunks = self.chunker.chunk_document(doc_name, cleaned_items)
                elapsed = time.time() - t0

                doc_stats[doc_name] = {
                    "items": len(cleaned_items),
                    "chunks": len(doc_chunks),
                    "time_seconds": round(elapsed, 2),
                    "status": "success",
                }
                logger.info(f"Extracted {len(cleaned_items)} items -> {len(doc_chunks)} chunks in {elapsed:.2f}s")
                all_chunks.extend(doc_chunks)

            except Exception as e:
                logger.error(f"Error processing {doc_name}: {e}", exc_info=True)
                doc_stats[doc_name] = {"status": "error", "error": str(e)}

        if not all_chunks:
            logger.error("No valid chunks were created from any documents.")
            return {"status": "error", "doc_stats": doc_stats, "total_chunks": 0}

        # 4. Embedding Generation
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks using BAAI/bge-small-en-v1.5...")
        embed_start = time.time()
        embedding_texts = [c["embedding_text"] for c in all_chunks]
        embeddings = self.embedder.embed_passages(embedding_texts)
        embed_time = time.time() - embed_start
        logger.info(f"Embeddings generated in {embed_time:.2f}s. Shape: {embeddings.shape}")

        # 5. ChromaDB Indexing
        index_start = time.time()
        indexed_count = self.indexer.index_chunks(all_chunks, embeddings)
        index_time = time.time() - index_start
        logger.info(f"Indexed {indexed_count} chunks in {index_time:.2f}s.")

        total_duration = time.time() - start_time
        logger.info(f"Pipeline completed in {total_duration:.2f}s. Total indexed: {indexed_count}")

        return {
            "status": "success",
            "total_documents": len(all_files),
            "total_chunks": indexed_count,
            "collection_count": self.indexer.count(),
            "doc_stats": doc_stats,
            "duration_seconds": round(total_duration, 2),
        }

