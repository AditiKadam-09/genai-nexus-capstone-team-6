"""Build index script for Indian Railways Help Desk.

Extracts, cleans, chunks, embeds, and indexes railway documents into a persistent ChromaDB vector store.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from helpdesk.ingest import IngestionPipeline, DocumentEmbedder
from helpdesk.ingestion.config import (
    DOCS_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build persistent vector index from railway circulars and rules.")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR, help="Path to directory containing documents")
    parser.add_argument("--persist-dir", type=Path, default=CHROMA_PERSIST_DIR, help="Path to ChromaDB storage directory")
    parser.add_argument("--collection", type=str, default=CHROMA_COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--reindex", action="store_true", help="Clear existing collection and re-index all documents")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Maximum chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Overlap between chunks in characters")
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL_NAME, help="SentenceTransformers embedding model")
    parser.add_argument("--device", type=str, default=EMBEDDING_DEVICE, help="Embedding device (cpu or cuda)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Initializing embedding model: %s on %s", args.model, args.device)
    embedder = DocumentEmbedder(model_name=args.model, device=args.device)

    pipeline = IngestionPipeline(
        docs_dir=args.docs_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedder=embedder,
    )

    result = pipeline.run(reindex=args.reindex)
    logger.info("Index building complete: %d chunks across %d documents indexed in collection '%s'.",
                result["total_chunks_indexed"], len(result["documents_processed"]), args.collection)


if __name__ == "__main__":
    main()
