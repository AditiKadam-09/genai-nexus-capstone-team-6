"""Controlled Synthetic Retrieval Benchmark.

Executes all 6 retrieval configurations on a controlled synthetic railway corpus
to measure and compare Hit@5, MRR@5, and Recall@5.

Used specifically for experimental validation and ablation analysis in absence of
the official rail_capstone_pack.zip dataset.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from helpdesk.search import search, reset_search_state

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark_synthetic")

SYNTHETIC_DIR = WORKSPACE_ROOT / "eval" / "synthetic_fixture"
SYNTHETIC_CHUNKS_FILE = SYNTHETIC_DIR / "chunks.json"


SYNTHETIC_CORPUS = [
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
    {
        "chunk_id": "chunk_clerkage_p2",
        "doc": "2015_refund_rules.pdf",
        "page": 2,
        "date": "2015-11-04",
        "section_path": "Refund Rules > Clerkage Charge > Unreserved and RAC",
        "text": "Clerkage charge of Rs. 60 per passenger is deducted for cancellation of RAC and waitlisted tickets. For unreserved second class, clerkage is Rs. 30 per passenger.",
        "embedding_text": "clerkage charge Rs 60 RAC waitlisted ticket cancellation unreserved Rs 30",
    },
    {
        "chunk_id": "chunk_time_window_50pct_p2",
        "doc": "2015_refund_rules.pdf",
        "page": 2,
        "date": "2015-11-04",
        "section_path": "Refund Rules > Cancellation Charges > Time Windows",
        "text": "Where a confirmed ticket is cancelled less than 12 hours and up to 4 hours before the scheduled departure of train or chart preparation, 50% of the fare subject to minimum flat charge is deducted.",
        "embedding_text": "cancelled less than 12 hours up to 4 hours before departure chart preparation 50 percent fare deduction",
    },
    {
        "chunk_id": "chunk_annexure_b_catering_p2",
        "doc": "catering_policy_2017.pdf",
        "page": 2,
        "date": "2017-02-27",
        "section_path": "Catering Policy > Annexure B > Meal Tariffs",
        "text": "Standard tea and coffee rates under Annexure B: Standard tea in cups (150 ml) Rs 10, packaged drinking water (1000 ml) Rail Neer Rs 15.",
        "embedding_text": "catering policy Annexure B standard tea coffee meal tariff Rail Neer Rs 15",
    },
]


BENCHMARK_QUERIES = [
    {
        "id": "q1",
        "question": "What is the cancellation charge on a confirmed AC First Class ticket cancelled more than 48 hours before departure?",
        "relevant_sources": [{"doc": "2015_refund_rules.pdf", "page": 1}],
    },
    {
        "id": "q2",
        "question": "What is the flat cancellation charge on a Sleeper Class ticket cancelled more than 48 hours in advance?",
        "relevant_sources": [{"doc": "2015_refund_rules.pdf", "page": 1}],
    },
    {
        "id": "q3",
        "question": "Under Railway Rule 25, under what conditions can a passenger claim full refund if train is cancelled?",
        "relevant_sources": [{"doc": "2015_refund_rules.pdf", "page": 4}],
    },
    {
        "id": "q4",
        "question": "What does Commercial Circular 65 specify regarding dynamic pricing for train 12002 from NDLS under Annexure A?",
        "relevant_sources": [{"doc": "commercial_circular_65.pdf", "page": 1}],
    },
    {
        "id": "q5",
        "question": "What are the refund rules for confirmed Tatkal ticket cancellation under Paragraph 4.2?",
        "relevant_sources": [{"doc": "circular_tatkal_rules.pdf", "page": 3}],
    },
    {
        "id": "q6",
        "question": "What clerkage fee is deducted on cancellation of waitlisted or RAC tickets?",
        "relevant_sources": [{"doc": "2015_refund_rules.pdf", "page": 2}],
    },
    {
        "id": "q7",
        "question": "How much refund is given if a confirmed ticket is cancelled between 12 hours and 4 hours before departure?",
        "relevant_sources": [{"doc": "2015_refund_rules.pdf", "page": 2}],
    },
    {
        "id": "q8",
        "question": "What is the price of Rail Neer drinking water under Annexure B of the catering policy?",
        "relevant_sources": [{"doc": "catering_policy_2017.pdf", "page": 2}],
    },
    {
        "id": "q9",
        "question": "Can I get my fare back if severe flooding or accidents prevent the train from running?",
        "relevant_sources": [{"doc": "2015_refund_rules.pdf", "page": 4}],
    },
    {
        "id": "q10",
        "question": "What is the refund penalty if I cancel within 48 hours up to 12 hours before train departure?",
        "relevant_sources": [{"doc": "2015_refund_rules.pdf", "page": 1}],
    },
]


def setup_synthetic_fixture():
    """Write synthetic corpus to synthetic fixture directory."""
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(SYNTHETIC_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(SYNTHETIC_CORPUS, f, indent=2)


def is_relevant(chunk: Dict[str, Any], rel_sources: List[Dict[str, Any]]) -> bool:
    meta = chunk.get("metadata", {})
    c_doc = meta.get("doc", "").lower()
    c_page = meta.get("page", 0)
    for src in rel_sources:
        s_doc = src.get("doc", "").lower()
        s_page = src.get("page", 0)
        if (s_doc in c_doc or c_doc in s_doc) and (s_page == 0 or abs(c_page - s_page) <= 1):
            return True
    return False


def run_benchmark():
    setup_synthetic_fixture()

    configurations = [
        ("BM25 Keyword Only", "bm25"),
        ("Dense Retrieval Only", "dense"),
        ("Dense + BM25 Linear", "dense_bm25"),
        ("Dense + BM25 + RRF (k=60)", "rrf"),
        ("Hybrid + Exact-Code Match", "hybrid_exact"),
        ("Hybrid + Exact-Code + CrossEncoder", "hybrid"),
    ]

    print("=====================================================================")
    print("      CONTROLLED RETRIEVAL BENCHMARK ON SYNTHETIC RAILWAY DATA       ")
    print("=====================================================================")
    print(f"Corpus size: {len(SYNTHETIC_CORPUS)} chunks | Query set: {len(BENCHMARK_QUERIES)} questions")
    print("---------------------------------------------------------------------")
    print(f"{'Configuration':<40} | {'Hit@5':<8} | {'MRR@5':<8} | {'Recall@5':<8}")
    print("---------------------------------------------------------------------")

    summary_results = []
    for label, mode in configurations:
        reset_search_state()
        hits = 0
        reciprocal_ranks = []
        recalls = []

        for item in BENCHMARK_QUERIES:
            q = item["question"]
            rel = item["relevant_sources"]
            results = search(q, k=5, mode=mode, index_dir=SYNTHETIC_DIR)

            first_rank = 0
            matched_sources: Set[Tuple[str, int]] = set()

            for rank, r in enumerate(results, start=1):
                if is_relevant(r, rel):
                    if first_rank == 0:
                        first_rank = rank
                    meta = r.get("metadata", {})
                    matched_sources.add((meta.get("doc", ""), meta.get("page", 0)))

            if first_rank > 0:
                hits += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)

            total_rel = max(1, len(rel))
            recalls.append(min(1.0, len(matched_sources) / total_rel))

        hit_at_5 = hits / len(BENCHMARK_QUERIES)
        mrr_at_5 = sum(reciprocal_ranks) / len(reciprocal_ranks)
        rec_at_5 = sum(recalls) / len(recalls)

        print(f"{label:<40} | {hit_at_5:<8.4f} | {mrr_at_5:<8.4f} | {rec_at_5:<8.4f}")
        summary_results.append({
            "configuration": label,
            "mode": mode,
            "hit@5": round(hit_at_5, 4),
            "mrr@5": round(mrr_at_5, 4),
            "recall@5": round(rec_at_5, 4),
        })

    print("=====================================================================")
    return summary_results


if __name__ == "__main__":
    run_benchmark()

