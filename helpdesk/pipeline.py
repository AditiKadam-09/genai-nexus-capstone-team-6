"""helpdesk.pipeline: Orchestrates routing and execution across documents and database."""

import re
import logging
from typing import Dict, Any, Optional

from helpdesk.search import search
from helpdesk.answer import answer_from_chunks
from helpdesk.sql import answer_from_sql
from helpdesk.llm import generate

logger = logging.getLogger(__name__)

# Keywords / Patterns for routing heuristics
TIMETABLE_PATTERNS = [
    r"\btrain(?:s)?\b",
    r"\btimetable\b",
    r"\bschedule(?:s)?\b",
    r"\bdeparture\b",
    r"\barrival\b",
    r"\bleave(?:s)?\b",
    r"\breach(?:es)?\b",
    r"\bstop(?:s)?\b",
    r"\bstation(?:s)?\b",
    r"\broute(?:s)?\b",
    r"\bhow many (?:rajdhani|shatabdi|express|mail|garib rath|duronto|jan shatabdi)\b",
    r"\bname of train\b",
    r"\bfrom \w+ to \w+\b",
    r"\bbetween \w+ and \w+\b",
    r"\b\d{5}\b",  # 5-digit Indian Railways train number (e.g. 12002, 12430)
]

DOCUMENT_PATTERNS = [
    r"\bcancellation\b",
    r"\bcancel(?:led|ing|s)?\b",
    r"\brefund(?:s)?\b",
    r"\bclerkage\b",
    r"\btatkal\b",
    r"\bpremium tatkal\b",
    r"\badvance reservation\b",
    r"\barp\b",
    r"\bchart\b",
    r"\bcharting\b",
    r"\bwaitlist(?:ed)?\b",
    r"\brac\b",
    r"\bgazette\b",
    r"\bnotification\b",
    r"\bcircular\b",
    r"\bvande bharat sleeper\b",
    r"\brule(?:s)?\b",
    r"\bpercentage\b",
    r"\bper passenger\b",
    r"\bdeduct(?:ed|ion)?\b",
    r"\bconfirmed ticket\b",
    r"\bhour(?:s)? before\b",
    r"\bday(?:s)? before\b",
    r"\blost ticket\b",
    r"\bticket deposit receipt\b",
    r"\btdr\b",
]


def classify_intent(question: str) -> str:
    """Classify passenger question into: 'documents', 'database', 'both', or 'neither'."""
    q_lower = question.lower()

    # Refusal / injection attempts
    for pat in [r"\b(change|update|delete|drop|insert|alter|modify|remove)\b"]:
        if re.search(pat, q_lower) and any(w in q_lower for w in ["database", "table", "time", "train", "departure", "schedule"]):
            return "neither"

    # Specific unanswerable: fare queries (the pack has no fare tables)
    if re.search(r"\b(fare|ticket price|cost of ticket|how much is a \w+ ticket)\b", q_lower):
        # Unless it specifically asks about cancellation/refund charge
        if not any(w in q_lower for w in ["refund", "cancellation", "cancel", "deduct", "clerkage"]):
            return "neither"

    has_db = any(re.search(pat, q_lower) for pat in TIMETABLE_PATTERNS)
    has_doc = any(re.search(pat, q_lower) for pat in DOCUMENT_PATTERNS)

    if has_db and has_doc:
        return "both"
    elif has_db:
        return "database"
    elif has_doc:
        return "documents"

    # Fallback to LLM classifier if ambiguous
    try:
        classifier_prompt = (
            "You are an intent classifier for an Indian Railways passenger help desk.\n"
            "Classes:\n"
            "- database: questions about train timetable, train names, train numbers, stations, routes, departure/arrival times, stops, counting trains.\n"
            "- documents: questions about railway circulars, rules, cancellation charges, refund percentages, clerkage, Tatkal rules, advance reservation limits (ARP), chart preparation.\n"
            "- both: questions that require BOTH timetable info (e.g. when a train departs) AND document policy (e.g. cancellation charges).\n"
            "- neither: questions asking for ticket fares/pricing (not in database/documents), database modification/deletion, or non-railway questions.\n\n"
            f"Question: {question}\n"
            "Reply with exactly one word: database, documents, both, or neither."
        )
        resp = generate([{"role": "user", "content": classifier_prompt}]).strip().lower()
        for choice in ["both", "database", "documents", "neither"]:
            if choice in resp:
                return choice
    except Exception as exc:
        logger.warning("Intent classification LLM call failed: %s", exc)

    return "documents"


def ask(question: str, top_k: int = 5) -> Dict[str, Any]:
    """Process a passenger question through routing and execution.

    Output format:
    {
        "answer": str,
        "sources": [{"doc": str, "page": Optional[int]}],
        "sql": Optional[str],
        "dont_know": bool
    }
    """
    if not question or not question.strip():
        return {
            "answer": "Please ask a question.",
            "sources": [],
            "sql": None,
            "dont_know": True,
        }

    q_clean = question.strip()
    q_lower = q_clean.lower()

    # Check for direct modification/refusal requests
    if any(cmd in q_lower for cmd in ["change the departure", "update the departure", "delete train", "drop table", "insert into"]):
        return {
            "answer": "I can't do that. I only run read-only queries on the timetable.",
            "sources": [],
            "sql": None,
            "dont_know": False,
        }

    # Check for fare questions (grounded refusal)
    if ("fare" in q_lower or "how much is a" in q_lower or "price" in q_lower) and not any(
        w in q_lower for w in ["refund", "cancel", "cancellation", "clerkage", "deduct"]
    ):
        return {
            "answer": "I don't know. The pack has no fare tables.",
            "sources": [],
            "sql": None,
            "dont_know": True,
        }

    intent = classify_intent(q_clean)
    logger.info("Question: %s | Classified intent: %s", q_clean, intent)

    search_mode = "bm25"

    if intent == "neither":
        return {
            "answer": "I don't know. The pack does not contain information to answer this question.",
            "sources": [],
            "sql": None,
            "dont_know": True,
        }

    if intent == "database":
        res = answer_from_sql(q_clean)
        return {
            "answer": res.get("answer", ""),
            "sources": res.get("sources", []),
            "sql": res.get("sql"),
            "dont_know": res.get("dont_know", False),
        }

    if intent == "documents":
        chunks = search(q_clean, mode=search_mode, k=top_k)
        res = answer_from_chunks(q_clean, chunks)
        return {
            "answer": res.get("answer", ""),
            "sources": res.get("sources", []),
            "sql": None,
            "dont_know": res.get("dont_know", False),
        }

    if intent == "both":
        sql_res = answer_from_sql(q_clean)
        chunks = search(q_clean, mode=search_mode, k=top_k)
        doc_res = answer_from_chunks(q_clean, chunks)

        sql_ans = sql_res.get("answer", "")
        doc_ans = doc_res.get("answer", "")
        sql_query = sql_res.get("sql")
        sources = doc_res.get("sources", [])

        # Synthesize combined response
        combine_prompt = (
            "You are an Indian Railways Passenger Help Desk Assistant. "
            "Combine the following two partial answers into one clear, direct, and coherent response to the passenger's question.\n\n"
            f"Passenger Question: {q_clean}\n\n"
            f"Timetable Data Answer: {sql_ans}\n\n"
            f"Rules/Policy Answer: {doc_ans}\n\n"
            "Combined Answer:"
        )
        try:
            combined_ans = generate([{"role": "user", "content": combine_prompt}]).strip()
        except Exception:
            combined_ans = f"{sql_ans} {doc_ans}".strip()

        dont_know = sql_res.get("dont_know", False) and doc_res.get("dont_know", False)

        return {
            "answer": combined_ans,
            "sources": sources,
            "sql": sql_query,
            "dont_know": dont_know,
        }

    # Default fallback
    chunks = search(q_clean, mode=search_mode, k=top_k)
    return answer_from_chunks(q_clean, chunks)
