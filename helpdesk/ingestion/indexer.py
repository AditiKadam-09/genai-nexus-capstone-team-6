"""Persistent ChromaDB indexing module for railway documents.

Manages collection lifecycle, metadata serialization, duplicate detection,
and persistent vector indexing compatible with the retrieval module.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings

from ingestion.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    HNSW_SPACE,
)

logger = logging.getLogger(__name__)


class ChromaIndexer:
    """Manages persistent ChromaDB vector index."""

    def __init__(
        self,
        persist_dir: Path = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(f"Initializing persistent ChromaDB client at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": HNSW_SPACE},
        )
        logger.info(f"Collection '{self.collection_name}' ready. Initial count: {self.collection.count()}")

    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> int:
        """Upsert chunks and embeddings into ChromaDB collection.

        Prevents duplicate errors and ensures valid metadata.
        Returns total number of chunks indexed.
        """
        if not chunks:
            logger.warning("No chunks provided to index.")
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        valid_ids: List[str] = []
        valid_docs: List[str] = []
        valid_embeddings: List[List[float]] = []
        valid_metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "").strip()

            if not chunk_id or not text:
                logger.warning(f"Skipping invalid or empty chunk at index {i}")
                continue

            doc_name = chunk.get("doc", "")
            raw_page = chunk.get("page")
            is_html = chunk.get("metadata", {}).get("is_html", False) or doc_name.endswith(".html")

            # ChromaDB requires primitive metadata types (str, int, float, bool)
            meta: Dict[str, Any] = {
                "doc": str(doc_name),
                "page": int(raw_page) if (raw_page is not None and not is_html) else -1,
                "is_html": bool(is_html),
                "section_title": str(chunk.get("section_title", "")),
                "section_path": str(chunk.get("section_path", "")),
                "split_method": str(chunk.get("split_method", "")),
            }

            valid_ids.append(chunk_id)
            valid_docs.append(text)
            valid_embeddings.append(embeddings[i].tolist())
            valid_metadatas.append(meta)

        total_to_index = len(valid_ids)
        logger.info(f"Indexing {total_to_index} valid chunks into collection '{self.collection_name}'...")

        # Upsert in batches
        for start_idx in range(0, total_to_index, batch_size):
            end_idx = min(start_idx + batch_size, total_to_index)
            self.collection.upsert(
                ids=valid_ids[start_idx:end_idx],
                embeddings=valid_embeddings[start_idx:end_idx],
                documents=valid_docs[start_idx:end_idx],
                metadatas=valid_metadatas[start_idx:end_idx],
            )

        final_count = self.collection.count()
        logger.info(f"Indexing complete. Collection '{self.collection_name}' now holds {final_count} records.")
        return total_to_index

    def clear(self) -> None:
        """Clear all entries in the collection for a clean rebuild."""
        logger.warning(f"Resetting collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": HNSW_SPACE},
        )
        logger.info(f"Collection '{self.collection_name}' has been reset.")

    def count(self) -> int:
        """Return the current number of records in the collection."""
        return self.collection.count()
