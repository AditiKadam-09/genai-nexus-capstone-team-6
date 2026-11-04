"""Document ingestion package for Railway Passenger Help Desk."""

from ingestion.config import (
    DOCUMENTS_DIR,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

__all__ = [
    "DOCUMENTS_DIR",
    "CHROMA_PERSIST_DIRECTORY",
    "CHROMA_COLLECTION_NAME",
    "EMBEDDING_MODEL_NAME",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
]
