"""Advanced Hybrid Retrieval Engine for Railway Passenger Help Desk.

Combines:
1. Dense semantic search (BAAI/bge-small-en-v1.5 + ChromaDB)
2. BM25 keyword search (rank_bm25 with railway code preservation)
3. Dedicated exact-code matching (Rules, train numbers, paragraphs, circulars, station codes)
4. Reciprocal Rank Fusion (RRF)
5. Cross-Encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("helpdesk.search")

DEFAULT_INDEX_DIR = Path("index")
DEFAULT_CHROMA_DIR = DEFAULT_INDEX_DIR / "chroma_db"
DEFAULT_COLLECTION_NAME = "railway_documents"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RRF_K = 60

# Global singletons to avoid repeated reloads
_embedding_model = None
_reranker_model = None
_chroma_client = None
_chroma_collection = None
_bm25_index = None
_bm25_chunks: List[Dict[str, Any]] = []


def tokenize_railway_text(text: str) -> List[str]:
    """Tokenize text while preserving railway codes, rule numbers, and train numbers."""
    # Normalize lower case but preserve patterns like TC-II/2015, Rule 25, 12002
    pattern = r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*|\d{5}|\bRule\s*\d+\b"
    tokens = re.findall(pattern, text, re.IGNORECASE)
    return [t.lower() for t in tokens if len(t) > 1]


def extract_exact_railway_codes(query: str) -> Dict[str, Set[str]]:
    """Extract domain-specific exact railway codes from a query."""
    codes: Dict[str, Set[str]] = {
        "rules": set(),
        "train_numbers": set(),
        "paragraphs": set(),
        "annexures": set(),
        "circulars": set(),
        "stations": set(),
    }

    # 1. Rule numbers: "Rule 25", "Rule 6", "Rule 49"
    rules = re.findall(r"\brule\s*(\d+[a-z\(\)]*)", query, re.IGNORECASE)
    for r in rules:
        codes["rules"].add(f"rule {r.lower()}")

    # 2. 5-digit train numbers (e.g. 12002, 12951, 22691)
    train_nums = re.findall(r"\b(1\d{4}|2\d{4}|0\d{4})\b", query)
    for tn in train_nums:
        codes["train_numbers"].add(tn)

    # 3. Paragraph numbers: "Para 3.2", "Paragraph 5"
    paras = re.findall(r"\bpara(?:graph)?\s*(\d+(?:\.\d+)?)", query, re.IGNORECASE)
    for p in paras:
        codes["paragraphs"].add(f"para {p.lower()}")

    # 4. Annexure references: "Annexure A", "Annexure-I"
    annex = re.findall(r"\bannexure\s*[-–—]?\s*([a-z0-9]+)", query, re.IGNORECASE)
    for a in annex:
        codes["annexures"].add(f"annexure {a.lower()}")

    # 5. Circular IDs: "TC-II/2015", "Commercial Circular 65"
    circs = re.findall(r"\b(tc[-–—/][a-z0-9/.\-]+|circular\s*(?:no\.?)?\s*\d+)", query, re.IGNORECASE)
    for c in circs:
        codes["circulars"].add(c.lower())

    # 6. Station codes (prominent uppercase 3-4 letter railway abbreviations)
    potential_stations = re.findall(r"\b([A-Z]{3,4})\b", query)
    known_stations = {
        "NDLS", "HWH", "CSMT", "MAS", "SBC", "PUNE", "ADI", "BCT", "BPL", "CNB",
        "GKP", "PNBE", "RNC", "GHY", "BBS", "SC", "HYB", "MYS", "TVC", "LKO"
    }
    for st in potential_stations:
        if st in known_stations:
            codes["stations"].add(st)

    return codes


def matches_exact_code(chunk_text: str, codes: Dict[str, Set[str]]) -> bool:
    """Check if chunk text contains any of the extracted exact codes."""
    lower_text = chunk_text.lower()
    for cat, items in codes.items():
        for item in items:
            if item in lower_text:
                return True
    return False


def get_embedding_model():
    """Lazy-load the BAAI/bge-small-en-v1.5 sentence-transformer model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return _embedding_model
    except Exception as exc:
        logger.warning("Could not load SentenceTransformer (%s). Dense retrieval will use fallback.", exc)
        return None


def get_reranker_model():
    """Lazy-load the CrossEncoder reranker model."""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder model: %s", RERANKER_MODEL_NAME)
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        return _reranker_model
    except Exception as exc:
        logger.warning("Could not load CrossEncoder (%s). Reranking will use fused scores.", exc)
        return None


def get_chroma_collection(
    index_dir: Path = DEFAULT_INDEX_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Retrieve or initialize persistent ChromaDB collection."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    chroma_path = index_dir / "chroma_db"
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        _chroma_collection = _chroma_client.get_or_create_collection(name=collection_name)
        return _chroma_collection
    except Exception as exc:
        logger.debug("ChromaDB not available or failed: %s", exc)
        return None


def load_chunks(index_dir: Path = DEFAULT_INDEX_DIR) -> List[Dict[str, Any]]:
    """Load chunks from index/chunks.json."""
    chunks_path = index_dir / "chunks.json"
    if not chunks_path.exists():
        logger.warning("Index file '%s' does not exist. Run build_index.py first.", chunks_path)
        return []
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_bm25_index(chunks: List[Dict[str, Any]]):
    """Build or retrieve the BM25 index for the supplied chunks."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None and len(_bm25_chunks) == len(chunks):
        return _bm25_index

    if not chunks:
        return None

    try:
        from rank_bm25 import BM25Okapi
        corpus = [tokenize_railway_text(c.get("text", "") + " " + c.get("section_path", "")) for c in chunks]
        _bm25_index = BM25Okapi(corpus)
        _bm25_chunks = chunks
        return _bm25_index
    except Exception as exc:
        logger.error("Failed to construct BM25 index: %s", exc)
        return None


def retrieve_dense(
    query: str,
    chunks: List[Dict[str, Any]],
    k: int = 10,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> List[Tuple[Dict[str, Any], float]]:
    """Retrieve top-k candidates using dense embedding search."""
    if not query.strip() or not chunks:
        return []

    collection = get_chroma_collection(index_dir)
    model = get_embedding_model()

    # 1. Primary path: ChromaDB vector store
    if collection is not None and model is not None and collection.count() > 0:
        try:
            query_emb = model.encode([query], normalize_embeddings=True).tolist()
            res = collection.query(
                query_embeddings=query_emb,
                n_results=min(k, collection.count()),
                include=["metadatas", "documents", "distances"],
            )
            results: List[Tuple[Dict[str, Any], float]] = []
            chunk_map = {c["chunk_id"]: c for c in chunks}
            ids = res["ids"][0] if res["ids"] else []
            distances = res["distances"][0] if res["distances"] else []

            for cid, dist in zip(ids, distances):
                score = max(0.0, 1.0 - dist)  # Convert cosine distance to similarity
                if cid in chunk_map:
                    results.append((chunk_map[cid], score))
                else:
                    meta = res["metadatas"][0][ids.index(cid)] if res.get("metadatas") else {}
                    doc_text = res["documents"][0][ids.index(cid)] if res.get("documents") else ""
                    chunk_obj = {
                        "chunk_id": cid,
                        "text": doc_text,
                        "metadata": meta,
                        "doc": meta.get("doc", ""),
                        "page": meta.get("page", 0),
                        "date": meta.get("date", "Unknown"),
                        "section_path": meta.get("section_path", ""),
                    }
                    results.append((chunk_obj, score))
            return results
        except Exception as exc:
            logger.debug("Chroma query failed: %s. Falling back to in-memory cosine.", exc)

    # 2. Fallback path: In-memory cosine similarity if model is loaded
    if model is not None:
        try:
            import numpy as np
            q_emb = model.encode(query, normalize_embeddings=True)
            corpus_texts = [c.get("embedding_text") or c.get("text", "") for c in chunks]
            doc_embs = model.encode(corpus_texts, normalize_embeddings=True)
            sims = np.dot(doc_embs, q_emb)
            top_indices = np.argsort(sims)[::-1][:k]
            return [(chunks[idx], float(sims[idx])) for idx in top_indices]
        except Exception as exc:
            logger.debug("In-memory embedding similarity failed: %s", exc)

    # 3. Graceful fallback if embeddings unavailable: simple term overlap
    q_tokens = set(tokenize_railway_text(query))
    scored = []
    for c in chunks:
        c_tokens = set(tokenize_railway_text(c.get("text", "")))
        overlap = len(q_tokens & c_tokens) / (len(q_tokens) or 1)
        scored.append((c, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def retrieve_bm25(
    query: str,
    chunks: List[Dict[str, Any]],
    k: int = 10,
) -> List[Tuple[Dict[str, Any], float]]:
    """Retrieve top-k candidates using BM25 keyword matching."""
    if not query.strip() or not chunks:
        return []

    bm25 = get_bm25_index(chunks)
    if bm25 is None:
        return []

    q_tokens = tokenize_railway_text(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], float(scores[i])) for i in ranked_indices if scores[i] > 0]


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Dict[str, Any], float]],
    bm25_results: List[Tuple[Dict[str, Any], float]],
    exact_match_ids: Set[str],
    k_rrf: int = DEFAULT_RRF_K,
    exact_bonus: float = 0.05,
    all_chunks_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """Fuse dense and BM25 candidate lists using Reciprocal Rank Fusion with exact code bonus."""
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    # Accumulate dense ranks
    for rank, (chunk, _) in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k_rrf + rank))

    # Accumulate BM25 ranks
    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k_rrf + rank))

    # Ensure any exact match chunk is included even if outside top-k dense/BM25 pool
    if all_chunks_map:
        for cid in exact_match_ids:
            if cid not in chunk_map and cid in all_chunks_map:
                chunk_map[cid] = all_chunks_map[cid]
                scores[cid] = 0.0

    # Apply exact-code match bonus
    for cid in exact_match_ids:
        if cid in scores:
            scores[cid] += exact_bonus

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[cid], score) for cid, score in sorted_items]


def rerank_with_cross_encoder(
    query: str, candidates: List[Dict[str, Any]], k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:
    """Rerank candidates using CrossEncoder ms-marco-MiniLM-L-6-v2."""
    if not candidates:
        return []

    reranker = get_reranker_model()
    if reranker is None:
        # Fallback to candidate order if reranker model is unavailable
        return [(c, 1.0 / (idx + 1)) for idx, c in enumerate(candidates[:k])]

    pairs = [[query, c.get("text", "")] for c in candidates]
    try:
        scores = reranker.predict(pairs)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(candidates[i], float(scores[i])) for i in ranked_indices]
    except Exception as exc:
        logger.warning("Reranking failed: %s. Using default candidate order.", exc)
        return [(c, 1.0 / (idx + 1)) for idx, c in enumerate(candidates[:k])]


def search(
    question: str,
    k: int = 5,
    rerank: bool = True,
    use_exact: bool = True,
    mode: str = "hybrid",
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> List[Dict[str, Any]]:
    """Execute search over railway document chunks across experimental configurations.

    Supported modes:
    - "dense": Dense only
    - "bm25": BM25 only
    - "dense_bm25": Linear combination
    - "rrf": Dense + BM25 + Reciprocal Rank Fusion
    - "hybrid_exact": Dense + BM25 + RRF + Exact code match
    - "hybrid": Dense + BM25 + RRF + Exact code + CrossEncoder rerank (default)

    Returns:
        List of dicts formatted as:
        {
          "chunk_id": "...",
          "text": "...",
          "metadata": {
            "doc": "...",
            "page": 0,
            "date": "...",
            "section_path": "..."
          },
          "rank": 1,
          "score": 0.0
        }
    """
    if not question.strip():
        return []

    chunks = load_chunks(index_dir)
    if not chunks:
        logger.warning("No chunks available for search in '%s'.", index_dir)
        return []

    pool_k = max(k * 4, 20)
    chunks_map = {c["chunk_id"]: c for c in chunks}

    # 1. Exact-code extraction
    exact_codes = extract_exact_railway_codes(question) if use_exact else {}
    exact_match_cids: Set[str] = set()
    if use_exact:
        for c in chunks:
            if matches_exact_code(c.get("text", "") + " " + c.get("section_path", ""), exact_codes):
                exact_match_cids.add(c["chunk_id"])

    # Mode 1: BM25 only
    if mode == "bm25":
        bm25_res = retrieve_bm25(question, chunks, k=k)
        ranked_chunks = bm25_res

    # Mode 2: Dense only
    elif mode == "dense":
        dense_res = retrieve_dense(question, chunks, k=k, index_dir=index_dir)
        ranked_chunks = dense_res

    # Mode 3: Dense + BM25 (linear score merge)
    elif mode == "dense_bm25":
        dense_res = retrieve_dense(question, chunks, k=pool_k, index_dir=index_dir)
        bm25_res = retrieve_bm25(question, chunks, k=pool_k)
        # Linear merge
        merged: Dict[str, float] = {}
        c_map: Dict[str, Dict[str, Any]] = {}
        for c, s in dense_res:
            cid = c["chunk_id"]
            merged[cid] = merged.get(cid, 0.0) + s
            c_map[cid] = c
        for c, s in bm25_res:
            cid = c["chunk_id"]
            merged[cid] = merged.get(cid, 0.0) + (s / 10.0)  # scale bm25
            c_map[cid] = c
        sorted_m = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:k]
        ranked_chunks = [(c_map[cid], score) for cid, score in sorted_m]

    # Mode 4: Dense + BM25 + RRF
    elif mode == "rrf":
        dense_res = retrieve_dense(question, chunks, k=pool_k, index_dir=index_dir)
        bm25_res = retrieve_bm25(question, chunks, k=pool_k)
        fused = reciprocal_rank_fusion(
            dense_res, bm25_res, exact_match_ids=set(), k_rrf=DEFAULT_RRF_K, all_chunks_map=chunks_map
        )
        ranked_chunks = fused[:k]

    # Mode 5: Hybrid + exact-code matching (no CrossEncoder)
    elif mode == "hybrid_exact":
        dense_res = retrieve_dense(question, chunks, k=pool_k, index_dir=index_dir)
        bm25_res = retrieve_bm25(question, chunks, k=pool_k)
        fused = reciprocal_rank_fusion(
            dense_res, bm25_res, exact_match_ids=exact_match_cids, k_rrf=DEFAULT_RRF_K, all_chunks_map=chunks_map
        )
        ranked_chunks = fused[:k]

    # Mode 6: Full Hybrid + exact-code + CrossEncoder rerank (default)
    else:
        dense_res = retrieve_dense(question, chunks, k=pool_k, index_dir=index_dir)
        bm25_res = retrieve_bm25(question, chunks, k=pool_k)
        fused = reciprocal_rank_fusion(
            dense_res, bm25_res, exact_match_ids=exact_match_cids, k_rrf=DEFAULT_RRF_K, all_chunks_map=chunks_map
        )
        candidate_chunks = [c for c, _ in fused[:pool_k]]

        if rerank:
            reranked = rerank_with_cross_encoder(question, candidate_chunks, k=k)
            ranked_chunks = reranked
        else:
            ranked_chunks = fused[:k]

    # Format output according to mandatory contract with deduplication & metadata preservation
    output: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    current_rank = 1

    for chunk, score in ranked_chunks:
        cid = chunk.get("chunk_id", "")
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)

        meta = chunk.get("metadata", {})
        doc = chunk.get("doc") or meta.get("doc", "")
        page = chunk.get("page") if chunk.get("page") is not None else meta.get("page", 0)
        date = chunk.get("date") or meta.get("date", "Unknown")
        section_path = chunk.get("section_path") or meta.get("section_path", "")

        output.append({
            "chunk_id": cid,
            "text": chunk.get("text", ""),
            "metadata": {
                "doc": doc,
                "page": page,
                "date": date,
                "section_path": section_path,
            },
            "rank": current_rank,
            "score": round(float(score), 4),
        })
        current_rank += 1
        if len(output) >= k:
            break

    return output


def reset_search_state():
    """Reset cached search indices and models (useful for unit testing)."""
    global _embedding_model, _reranker_model, _chroma_client, _chroma_collection, _bm25_index, _bm25_chunks
    _embedding_model = None
    _reranker_model = None
    _chroma_client = None
    _chroma_collection = None
    _bm25_index = None
    _bm25_chunks = []

