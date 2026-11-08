"""Gemini client wrapper using google-genai SDK."""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
_client = None


def get_genai_client():
    """Lazy initialize google-genai client."""
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set in environment or .env file.")
        return None
    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception as exc:
        logger.error("Failed to initialize genai.Client: %s", exc)
        return None


def generate(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Generate text completion from Gemini."""
    client = get_genai_client()
    if client is None:
        return "I could not reach the language model right now. Please try again in a minute."

    # Construct prompt from messages
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System Instructions:\n{content}\n")
        elif role == "user":
            prompt_parts.append(f"User: {content}\n")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}\n")

    full_prompt = "\n".join(prompt_parts) if len(prompt_parts) > 1 else messages[-1].get("content", "")

    try:
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
        )
        return response.text.strip() if response and response.text else ""
    except Exception as exc:
        logger.error("Gemini generation call failed: %s", exc)
        return "I could not reach the language model right now. Please try again in a minute."
