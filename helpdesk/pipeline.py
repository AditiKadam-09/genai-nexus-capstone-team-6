"""Question routing and the single ask() entry point for the Railway Help Desk.

ask(question) always returns:
{"answer": str, "sources": [{"doc": str, "page": int or None}],
 "sql": str or None, "dont_know": bool}
"""

from __future__ import annotations

import json
import re
from typing import Any

from helpdesk.answer import generate_answer
from helpdesk.llm import generate
from helpdesk.search import search
from helpdesk.sql import answer_from_sql

TOP_K = 5

OUT_OF_SCOPE_ANSWER = (
    "I don't know. I can only answer questions about the railway rules in "
    "my documents (refunds, cancellations, booking limits, Tatkal and "
    "similar) and the 2016 timetable."
)

# ---------------------------------------------------------------
# Routing
# ---------------------------------------------------------------

# Words that point to the timetable database.
DB_WORDS = [
    r"\b\d{5}\b", r"\btrains?\b", r"\bstations?\b", r"\bdeparts?\b",
    r"\bdeparting\b", r"\bdeparture time\b", r"\barrives?\b",
    r"\barriving\b", r"\bleaves?\b", r"\btimetable\b", r"\bschedules?\b",
    r"\broutes?\b", r"\bstops?\b", r"\bdatabase\b",
]

# Words that point to the rule documents.
DOC_WORDS = [
    r"\brefunds?\b", r"\bcancel\w*", r"\bcharges?\b", r"\bdeduct\w*",
    r"\brules?\b", r"\bcirculars?\b", r"\btatkal\b", r"\baadhaar\b",
    r"\badvance\b", r"\bbooking\b", r"\bgazette\b", r"\bnotification\b",
    r"\bwaitlist\w*", r"\brac\b", r"\bpolicy\b", r"\bpenalty\b",
    r"\bclerkage\b", r"\bpib\b", r"\bpress release\b", r"\breservation\b",
]

_DB_RE = re.compile("|".join(DB_WORDS), re.IGNORECASE)
_DOC_RE = re.compile("|".join(DOC_WORDS), re.IGNORECASE)

ROUTER_PROMPT = """You route questions for an Indian Railways help desk.
Two sources exist:
- documents: refund and cancellation rules, charges, booking limits, Tatkal, ID proof, circulars and press releases.
- database: a 2016 timetable (trains, stations, routes, departure and arrival times, stops, counts of trains).

Reply with JSON only, no markdown:
{"route": "documents", "documents_question": "...", "database_question": "..."}

route is one of:
- "documents": only the rule documents are needed.
- "database": only the timetable is needed. Requests to change or delete data also use "database" (they are refused there).
- "both": the question needs a rule AND a timetable fact. Write each part as a complete, standalone question in documents_question and database_question.
- "none": neither source can answer it: fares or ticket prices, seat availability, live running status, PNR status, predictions about the future (for example whether a waitlisted ticket will get confirmed), or anything unrelated to Indian Railways.

For routes other than "both", copy the original question into the question fields."""


def _plan(route: str, question: str, documents_question=None, database_question=None) -> dict:
    return {
        "route": route,
        "documents_question": documents_question or question,
        "database_question": database_question or question,
    }


def _parse_json(text: str) -> dict | None:
    cleaned = (text or "").replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def route_question(question: str) -> dict:
    """Decide which path answers the question.

    Obvious questions are routed by keywords (no Gemini call, which saves the
    free-tier quota). Mixed or unclear questions are sent to Gemini.
    """
    has_db = bool(_DB_RE.search(question))
    has_doc = bool(_DOC_RE.search(question))

    if has_db and not has_doc:
        return _plan("database", question)
    if has_doc and not has_db:
        return _plan("documents", question)

    try:
        reply = generate([
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": question},
        ])
        data = _parse_json(reply)
        if data and data.get("route") in ("documents", "database", "both", "none"):
            return _plan(
                data["route"],
                question,
                str(data.get("documents_question") or "").strip() or None,
                str(data.get("database_question") or "").strip() or None,
            )
    except Exception:
        pass

    # Fallback when Gemini is unavailable.
    return _plan("both" if (has_db and has_doc) else "documents", question)


# ---------------------------------------------------------------
# The two paths
# ---------------------------------------------------------------

def _dont_know(answer: str, sql: str | None = None) -> dict:
    return {"answer": answer, "sources": [], "sql": sql, "dont_know": True}


def _clean_page(page: Any):
    """Page numbers start at 1. Anything else (web pages) becomes None."""
    if isinstance(page, bool):
        return None
    if isinstance(page, int) and page > 0:
        return page
    return None


_CITATION_RE = re.compile(r"\s*\[Source[^\]]*\]", re.IGNORECASE)


def _from_answer_result(result: dict) -> dict:
    """Convert answer.generate_answer() output to the 4-key format."""
    chunks = result.get("results") or []
    text = str(result.get("answer") or "").strip()

    if result.get("status") == "insufficient_evidence":
        return _dont_know(text or "I don't know.")

    # Sources = the chunks the answer cites; if none is cited, the top two.
    cited = [c for c in chunks if str(c.get("chunk_id") or "") and str(c["chunk_id"]) in text]
    used = cited or chunks[:2]

    sources, seen = [], set()
    for chunk in used:
        meta = chunk.get("metadata") or {}
        doc = meta.get("doc")
        if not doc:
            continue
        page = _clean_page(meta.get("page"))
        if (doc, page) not in seen:
            seen.add((doc, page))
            sources.append({"doc": doc, "page": page})

    clean_text = _CITATION_RE.sub("", text).strip()
    return {"answer": clean_text or text, "sources": sources, "sql": None, "dont_know": False}


def _run_documents(question: str) -> dict:
    try:
        chunks = search(question, k=TOP_K)
        result = generate_answer(question, retrieved_chunks=chunks)
        return _from_answer_result(result)
    except Exception as error:
        return _dont_know(
            "I could not search the railway documents right now. "
            f"({type(error).__name__})"
        )


def _run_database(question: str) -> dict:
    try:
        return answer_from_sql(question)
    except Exception as error:
        return _dont_know(
            "I could not read the timetable right now. "
            f"({type(error).__name__})"
        )


def _merge(docs: dict, db: dict) -> dict:
    """Combine the two answers of a mixed question."""
    if docs["dont_know"] and db["dont_know"]:
        return _dont_know(docs["answer"] or db["answer"], db["sql"])

    answered = [r["answer"] for r in (db, docs) if not r["dont_know"]]
    missing = [r["answer"] for r in (db, docs) if r["dont_know"]]
    return {
        "answer": " ".join(answered + missing),
        "sources": [] if docs["dont_know"] else docs["sources"],
        "sql": None if db["dont_know"] else db["sql"],
        "dont_know": False,
    }


def _finalize(result: dict) -> dict:
    """Make sure the result has exactly the four required keys."""
    sources = []
    for item in result.get("sources") or []:
        if isinstance(item, dict) and item.get("doc"):
            sources.append({"doc": item["doc"], "page": _clean_page(item.get("page"))})
    return {
        "answer": str(result.get("answer") or ""),
        "sources": sources,
        "sql": result.get("sql") or None,
        "dont_know": bool(result.get("dont_know")),
    }


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def ask(question: str) -> dict:
    """Answer one question. Never raises; always returns the 4-key dict."""
    try:
        question = (question or "").strip()
        if not question:
            return _finalize(_dont_know("Please ask a question."))

        plan = route_question(question)
        route = plan["route"]

        if route == "none":
            return _finalize(_dont_know(OUT_OF_SCOPE_ANSWER))
        if route == "database":
            return _finalize(_run_database(plan["database_question"]))
        if route == "documents":
            return _finalize(_run_documents(plan["documents_question"]))

        docs = _run_documents(plan["documents_question"])
        db = _run_database(plan["database_question"])
        return _finalize(_merge(docs, db))
    except Exception as error:
        return _finalize(_dont_know(f"Sorry, something went wrong: {type(error).__name__}"))
