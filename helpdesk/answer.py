"""Grounded answer generation using retrieved document chunks and Gemini."""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def answer_from_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    llm=None,
) -> Dict[str, Any]:
    """Generate a grounded plain-language answer citing document evidence.

    Returns:
        {
            "answer": "Plain-language response",
            "sources": [{"doc": "filename.pdf", "page": 3}],
            "sql": None,
            "dont_know": False
        }
    """
    if not chunks:
        return {
            "answer": "I don't know. The supplied railway documents do not contain evidence to answer this question.",
            "sources": [],
            "sql": None,
            "dont_know": True,
        }

    # Extract distinct document/page sources
    sources = []
    seen = set()
    for c in chunks:
        meta = c.get("metadata", {})
        doc = meta.get("doc")
        page = meta.get("page")
        if doc:
            key = (doc, page)
            if key not in seen:
                seen.add(key)
                sources.append({"doc": doc, "page": page})

    # Prepare evidence context
    evidence_blocks = []
    for idx, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        doc = meta.get("doc", "Unknown")
        page = meta.get("page", "Unknown")
        text = c.get("text", "").strip()
        evidence_blocks.append(f"[Evidence {idx}] (Document: {doc}, Page: {page})\n{text}")

    context = "\n\n".join(evidence_blocks)

    system_prompt = (
        "You are an official Indian Railways Passenger Help Desk Assistant. "
        "Your task is to answer passenger queries strictly and solely based on the provided document evidence.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Answer ONLY using facts, rules, time limits, and amounts present in the evidence.\n"
        "2. Do NOT invent, assume, extrapolate, or bring outside knowledge.\n"
        "3. If the evidence does NOT contain sufficient information to answer the question, state: "
        "'I don't know. The supplied documents do not contain this information.' and set dont_know=true.\n"
        "4. Always mention the applicable circular/rule numbers or document names if mentioned in the evidence.\n"
        "5. Be concise, direct, and factual."
    )

    user_prompt = f"Passenger Question: {question}\n\nApplicable Evidence:\n{context}\n\nPlease provide a clear, factual answer:"

    try:
        if llm is None:
            from helpdesk.llm import generate
            response_text = generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        else:
            response_text = llm(f"{system_prompt}\n\n{user_prompt}")

        if not response_text:
            return {
                "answer": "I don't know. Could not generate an answer.",
                "sources": sources,
                "sql": None,
                "dont_know": True,
            }

        # Check for abstention cues
        lower_ans = response_text.lower()
        dont_know = "don't know" in lower_ans or "do not contain" in lower_ans or "cannot answer" in lower_ans

        return {
            "answer": response_text,
            "sources": sources,
            "sql": None,
            "dont_know": dont_know,
        }

    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        return {
            "answer": "I could not generate an answer due to an internal error.",
            "sources": sources,
            "sql": None,
            "dont_know": True,
        }
