"""Embedding generation module using BAAI/bge-small-en-v1.5.

Encodes passage chunks and queries with cosine normalization
and query instruction prefixes.
"""

import os
import logging
from typing import List, Union
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
from ingestion.config import EMBEDDING_MODEL_NAME, QUERY_INSTRUCTION

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Generates dense vector embeddings using SentenceTransformer."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {self.model_name}")
        # Try local first, fall back if needed
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully.")

    def embed_passages(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of passage texts for storage in vector database."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string with the BGE query instruction prefix."""
        prefixed_query = QUERY_INSTRUCTION + query.strip()
        query_embedding = self.model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return query_embedding
