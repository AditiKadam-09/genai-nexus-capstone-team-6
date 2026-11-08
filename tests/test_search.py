"""Comprehensive Unit and Regression Tests for Advanced Hybrid Retrieval Engine.

Covers:
1. Railway tokenization & domain code preservation
2. Exact-code extraction (Rules, 5-digit trains, paras, annexures, circulars, station codes)
3. Negative filtering (no false train numbers on 48 hours, 120 Rs, 2024 year)
4. BM25 keyword retrieval on synthetic corpus
5. Reciprocal Rank Fusion (RRF) mathematical formula verification
6. Exact-code boost contribution in RRF
7. All 6 retrieval modes:
   - Dense only
   - BM25 only
   - Dense + BM25 linear combination
   - Dense + BM25 + RRF
   - Hybrid + exact-code matching
   - Hybrid + exact-code + CrossEncoder reranking
8. Output schema and metadata preservation (doc, page, date, section_path)
9. Boundary handling (empty queries, missing index directory, whitespace)
"""

import json
from pathlib import Path
import pytest

from helpdesk.search import (
    tokenize_railway_text,
    extract_exact_railway_codes,
    matches_exact_code,
    reciprocal_rank_fusion,
    retrieve_bm25,
    search,
    reset_search_state,
    DEFAULT_RRF_K,
)


@pytest.fixture(autouse=True)
def clean_search_state():
    """Reset singleton state before and after each test."""
    reset_search_state()
    yield
    reset_search_state()


@pytest.fixture
def synthetic_index_dir(tmp_path: Path) -> Path:
    """Create a temporary index directory populated with realistic synthetic railway chunks."""
    chunks = [
        {
            "chunk_id": "chunk_rule25_p4",
            "doc": "2015_refund_rules.pdf",
            "page": 4,
            "date": "2015-11-04",
            "section_path": "Refund Rules > Rule 25 > Unforeseen Circumstances",
            "text": "Rule 25: Refund on Cancellation of Trains due to accidents, breaches or floods. Full refund of fare paid shall be granted within 72 hours of scheduled departure.",
            "embedding_text": "Refund on Cancellation of Trains due to accidents floods Rule 25 72 hours full refund",
        },
        {
            "chunk_id": "chunk_ac_cancellation_p1",
            "doc": "2015_refund_rules.pdf",
            "page": 1,
            "date": "2015-11-04",
            "section_path": "Refund Rules > Cancellation Charges > Confirmed Tickets",
            "text": "If a confirmed ticket is cancelled more than 48 hours before scheduled departure, flat cancellation charge per passenger is Rs. 240 for AC First Class / Executive Class.",
            "embedding_text": "confirmed ticket cancelled more than 48 hours AC First Class flat charge Rs 240",
        },
        {
            "chunk_id": "chunk_sleeper_cancellation_p1",
            "doc": "2015_refund_rules.pdf",
            "page": 1,
            "date": "2015-11-04",
            "section_path": "Refund Rules > Cancellation Charges > Confirmed Tickets",
            "text": "If a confirmed ticket is cancelled more than 48 hours before departure, flat cancellation charge per passenger is Rs. 120 for Sleeper Class and Rs. 180 for AC 3 Tier.",
            "embedding_text": "cancellation charge sleeper class Rs 120 AC 3 tier Rs 180 48 hours",
        },
        {
            "chunk_id": "chunk_train_12002_p1",
            "doc": "commercial_circular_65.pdf",
            "page": 1,
            "date": "2018-06-20",
            "section_path": "Commercial Circulars > Special Trains > Shatabdi",
            "text": "Commercial Circular 65: Catering charges and dynamic pricing for train 12002 New Delhi Bhopal Shatabdi departing from NDLS station under Annexure A.",
            "embedding_text": "train 12002 new delhi bhopal shatabdi NDLS commercial circular 65 Annexure A dynamic pricing",
        },
        {
            "chunk_id": "chunk_tatkal_p3",
            "doc": "circular_tatkal_rules.pdf",
            "page": 3,
            "date": "2016-01-15",
            "section_path": "Tatkal Scheme > Paragraph 4.2 > Refund Conditions",
            "text": "Under Paragraph 4.2 of Tatkal Scheme, no refund of fare shall be granted on cancellation of confirmed Tatkal tickets except in cases of train cancellation or delay exceeding 3 hours.",
            "embedding_text": "tatkal scheme paragraph 4.2 no refund confirmed tatkal tickets train cancelled delay 3 hours",
        },
    ]

    chunks_file = tmp_path / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return tmp_path


# ==============================================================================
# 1. Domain Tokenization & Preservation
# ==============================================================================

def test_tokenize_railway_text_preserves_codes():
    text = "Under Rule 25 and circular TC-II/2015, train 12002 charges NDLS to HWH."
    tokens = tokenize_railway_text(text)
    assert any("rule" in t for t in tokens)
    assert "12002" in tokens
    assert "ndls" in tokens
    assert "hwh" in tokens


# ==============================================================================
# 2. Exact-Code Extraction
# ==============================================================================

def test_extract_exact_railway_codes_all_categories():
    query = (
        "What does Rule 25 say about train 12951 running late into NDLS "
        "under Annexure A and Commercial Circular 65 according to Para 3.2?"
    )
    codes = extract_exact_railway_codes(query)

    assert "rule 25" in codes["rules"]
    assert "12951" in codes["train_numbers"]
    assert "annexure a" in codes["annexures"]
    assert "NDLS" in codes["stations"]
    assert "para 3.2" in codes["paragraphs"]
    assert any("circular 65" in c or "commercial circular 65" in c for c in codes["circulars"])


def test_extract_exact_railway_codes_negative_filtering():
    """Verify that regular non-code numbers (hours, prices, years) are not falsely parsed as train numbers."""
    query = "Cancelled 48 hours before departure with fee of 240 rupees in the year 2024"
    codes = extract_exact_railway_codes(query)

    assert len(codes["train_numbers"]) == 0
    assert len(codes["rules"]) == 0
    assert len(codes["annexures"]) == 0


def test_matches_exact_code():
    codes = {
        "rules": {"rule 25"},
        "train_numbers": {"12002"},
        "paragraphs": {"para 4.2"},
        "annexures": {"annexure a"},
        "circulars": set(),
        "stations": {"NDLS"},
    }

    assert matches_exact_code("As per Rule 25, passengers are entitled to refund.", codes)
    assert matches_exact_code("Train 12002 departs NDLS at 06:00.", codes)
    assert matches_exact_code("Refer to Annexure A for charges.", codes)
    assert matches_exact_code("See Para 4.2 for details.", codes)
    assert not matches_exact_code("General waiting list rules for second class.", codes)


# ==============================================================================
# 3. Reciprocal Rank Fusion Mathematics & Code Boosting
# ==============================================================================

def test_reciprocal_rank_fusion_math():
    dense = [
        ({"chunk_id": "c1"}, 0.95),
        ({"chunk_id": "c2"}, 0.85),
    ]
    bm25 = [
        ({"chunk_id": "c2"}, 6.0),
        ({"chunk_id": "c3"}, 4.0),
    ]

    fused = reciprocal_rank_fusion(dense, bm25, exact_match_ids=set(), k_rrf=60)
    # c2 is at rank 2 in dense (1/62) and rank 1 in bm25 (1/61) -> ~0.0325
    # c1 is only at rank 1 in dense (1/61) -> ~0.0164
    # c3 is only at rank 2 in bm25 (1/62) -> ~0.0161
    assert fused[0][0]["chunk_id"] == "c2"
    assert fused[1][0]["chunk_id"] == "c1"
    assert fused[2][0]["chunk_id"] == "c3"

    score_c2 = fused[0][1]
    expected_c2 = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
    assert pytest.approx(score_c2, rel=1e-4) == expected_c2


def test_reciprocal_rank_fusion_exact_bonus():
    """Verify that an exact-code match gets an additive bonus pushing it up the ranking."""
    dense = [
        ({"chunk_id": "c1"}, 0.95),
        ({"chunk_id": "c2"}, 0.85),
    ]
    bm25 = [
        ({"chunk_id": "c1"}, 5.0),
        ({"chunk_id": "c2"}, 4.0),
    ]

    # Without bonus, c1 wins unanimously
    fused_no_bonus = reciprocal_rank_fusion(dense, bm25, exact_match_ids=set(), k_rrf=60)
    assert fused_no_bonus[0][0]["chunk_id"] == "c1"

    # With exact match on c2 and bonus=0.05, c2 gains enough to surpass c1:
    # c1 score = 1/61 + 1/61 = ~0.0328
    # c2 score = 1/62 + 1/62 + 0.05 = ~0.0822
    fused_with_bonus = reciprocal_rank_fusion(
        dense, bm25, exact_match_ids={"c2"}, k_rrf=60, exact_bonus=0.05
    )
    assert fused_with_bonus[0][0]["chunk_id"] == "c2"


# ==============================================================================
# 4. Isolated BM25 Retrieval
# ==============================================================================

def test_retrieve_bm25_synthetic_corpus(synthetic_index_dir: Path):
    with open(synthetic_index_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Query targeting Tatkal cancellation
    results = retrieve_bm25("Tatkal ticket cancellation refund", chunks, k=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk["chunk_id"] == "chunk_tatkal_p3"
    assert score > 0.0


# ==============================================================================
# 5. Verification of All Six Retrieval Modes
# ==============================================================================

@pytest.mark.parametrize("mode", [
    "dense",
    "bm25",
    "dense_bm25",
    "rrf",
    "hybrid_exact",
    "hybrid",
])
def test_all_retrieval_modes_return_contract_schema(synthetic_index_dir: Path, mode: str):
    query = "What is the cancellation charge for AC First Class 48 hours before departure?"
    results = search(query, k=3, mode=mode, index_dir=synthetic_index_dir)

    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 3

    for rank_idx, res in enumerate(results, start=1):
        assert "chunk_id" in res
        assert "text" in res
        assert "metadata" in res
        assert "rank" in res
        assert "score" in res

        assert res["rank"] == rank_idx
        assert isinstance(res["score"], float)

        # Validate metadata preservation
        meta = res["metadata"]
        assert "doc" in meta and meta["doc"].endswith(".pdf")
        assert "page" in meta and isinstance(meta["page"], int)
        assert "date" in meta and isinstance(meta["date"], str)
        assert "section_path" in meta and isinstance(meta["section_path"], str)


def test_search_retrieves_exact_rule_with_hybrid_exact(synthetic_index_dir: Path):
    query = "According to Rule 25, how is refund calculated if a train is cancelled?"
    results = search(query, k=2, mode="hybrid_exact", index_dir=synthetic_index_dir)

    assert len(results) > 0
    top_result = results[0]
    assert top_result["chunk_id"] == "chunk_rule25_p4"
    assert "Rule 25" in top_result["text"]
    assert top_result["metadata"]["doc"] == "2015_refund_rules.pdf"
    assert top_result["metadata"]["page"] == 4


# ==============================================================================
# 6. Edge Cases & Boundary Handling
# ==============================================================================

def test_reciprocal_rank_fusion_exact_match_outside_pool():
    """Verify that an exact-code match not present in dense/BM25 pool is still recovered via all_chunks_map."""
    dense = [({"chunk_id": "c1"}, 0.9)]
    bm25 = [({"chunk_id": "c1"}, 5.0)]
    all_chunks = {
        "c1": {"chunk_id": "c1", "text": "General refund rules"},
        "c_rule25": {"chunk_id": "c_rule25", "text": "Rule 25 emergency cancellation"},
    }

    fused = reciprocal_rank_fusion(
        dense, bm25, exact_match_ids={"c_rule25"}, k_rrf=60, exact_bonus=0.05, all_chunks_map=all_chunks
    )
    # c_rule25 gets 0 + 0.05 = 0.05, surpassing c1 (1/61 + 1/61 = ~0.0328)
    assert fused[0][0]["chunk_id"] == "c_rule25"


def test_search_deduplication_and_metadata_preservation(tmp_path: Path):
    """Verify that search deduplicates results and preserves all metadata fields."""
    duplicate_chunks = [
        {
            "chunk_id": "c_dup",
            "doc": "test_refund.pdf",
            "page": 2,
            "date": "2020-01-01",
            "section_path": "Refunds > Duplicate Test",
            "text": "First copy of ticket cancellation rule.",
            "embedding_text": "First copy of ticket cancellation rule",
        },
        {
            "chunk_id": "c_dup",
            "doc": "test_refund.pdf",
            "page": 2,
            "date": "2020-01-01",
            "section_path": "Refunds > Duplicate Test",
            "text": "Second duplicate copy of ticket cancellation rule.",
            "embedding_text": "Second duplicate copy of ticket cancellation rule",
        },
        {
            "chunk_id": "c_unique",
            "doc": "test_refund.pdf",
            "page": 3,
            "date": "2020-01-01",
            "section_path": "Refunds > Unique Section",
            "text": "Unique rule for unreserved passengers.",
            "embedding_text": "Unique rule for unreserved passengers",
        },
    ]
    chunks_file = tmp_path / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(duplicate_chunks, f)

    results = search("cancellation rule", k=5, index_dir=tmp_path)
    chunk_ids = [r["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs returned in search results"
    for r in results:
        assert r["metadata"]["doc"] == "test_refund.pdf"
        assert isinstance(r["metadata"]["page"], int)
        assert r["metadata"]["date"] == "2020-01-01"
        assert r["metadata"]["section_path"] != ""


def test_empty_and_whitespace_query_search():
    assert search("") == []
    assert search("   ") == []
    assert search("\n\t  \r") == []


def test_missing_index_dir_handled_gracefully(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist"
    results = search("train ticket cancellation", index_dir=nonexistent)
    assert results == []
