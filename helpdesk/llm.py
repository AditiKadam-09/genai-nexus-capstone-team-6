"""Shared Gemini client for the Railway Passenger Help Desk.

The public generate(messages) function is intentionally compatible with
helpdesk.sql, which imports it to generate Text-to-SQL queries.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Iterable

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
_client = None


class LLMError(RuntimeError):
    """Raised when the Gemini request cannot be completed."""


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise LLMError(
            "Gemini API key is missing. Set GEMINI_API_KEY in your environment "
            "or in the project-root .env file."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise LLMError(
            "The Gemini SDK is not installed. Run: pip install -r requirements.txt"
        ) from exc

    _client = genai.Client(api_key=api_key)
    return _client


def _convert_messages(messages: Iterable[dict[str, Any]]):
    """Convert OpenAI-style role/content messages into Gemini contents."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = message.get("content", "")
        if isinstance(content, list):
            # Accept simple text-part message payloads.
            content = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        content = str(content)

        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": content}]})

    system_instruction = "\n\n".join(system_parts) or None
    return system_instruction, contents


def generate(
    messages: str | list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> str:
    """Generate text with Gemini.

    `messages` accepts either a plain prompt string or a list of
    {role, content} dictionaries. Returns response text.

    This interface is shared by answer generation and Text-to-SQL.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not messages:
        raise ValueError("messages cannot be empty")

    system_instruction, contents = _convert_messages(messages)
    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Respond to the request."}]}]

    client = _get_client()
    selected_model = model or DEFAULT_MODEL

    try:
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )
    except ImportError as exc:
        raise LLMError("Unable to import Gemini SDK configuration types.") from exc

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=selected_model,
                contents=contents,
                config=config,
            )
            text = getattr(response, "text", None)
            if not text or not str(text).strip():
                raise LLMError("Gemini returned an empty response.")
            return str(text).strip()
        except LLMError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                message = str(exc)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    wait_seconds = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s, 100s
                    print(
                        f"[Gemini rate limit reached - waiting {wait_seconds}s, then retrying]",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                else:
                    time.sleep(min(2 ** attempt, 8))

    raise LLMError(f"Gemini request failed after {_MAX_RETRIES + 1} attempts: {last_error}") from last_error


def reset_client():
    """Reset the cached client; useful for tests."""
    global _client
    _client = None
