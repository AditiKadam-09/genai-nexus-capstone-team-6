"""Text and table cleaning utilities for document ingestion.

Preserves structure, formats tables into markdown, and binds footnotes
to preceding tables according to capstone project requirements.
"""

import re
from typing import List, Dict, Any, Tuple


def clean_text(text: str) -> str:
    """Clean and normalize raw text while preserving formatting.

    - Replaces non-breaking spaces and irregular tabs
    - Normalizes consecutive spaces on a single line
    - Trims excess blank lines (max 2 consecutive newlines)
    - Strips leading and trailing whitespace
    """
    if not text:
        return ""

    # Replace unicode spaces and non-breaking spaces
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\u2003", " ")
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs with spaces if inside sentences
    text = re.sub(r"[ \t]+", " ", text)

    # Clean empty lines
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = []
    consecutive_empty = 0
    for line in lines:
        if not line:
            consecutive_empty += 1
            if consecutive_empty <= 1:
                cleaned_lines.append("")
        else:
            consecutive_empty = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def table_to_text(headers: List[str], rows: List[List[str]]) -> str:
    """Convert headers and rows into clean markdown table representation."""
    cleaned_headers = [clean_text(str(h)).replace("\n", " ").replace("|", "\\|") for h in headers]
    if not cleaned_headers:
        cleaned_headers = [f"Col_{i+1}" for i in range(len(rows[0]) if rows else 0)]

    lines = ["| " + " | ".join(cleaned_headers) + " |"]
    lines.append("|" + "---|" * len(cleaned_headers))

    for row in rows:
        cleaned_row = []
        for i in range(len(cleaned_headers)):
            cell = row[i] if i < len(row) else ""
            cleaned_cell = clean_text(str(cell)).replace("\n", " ").replace("|", "\\|")
            cleaned_row.append(cleaned_cell)
        lines.append("| " + " | ".join(cleaned_row) + " |")

    return "\n".join(lines)


def clean_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean extracted items list and bind footnotes to preceding tables.

    Items format:
    - {'type': 'heading', 'text': ..., 'page': ...}
    - {'type': 'table', 'headers': [...], 'rows': [[...]], 'page': ...}
    - {'type': 'paragraph', 'text': ..., 'page': ...}
    """
    cleaned: List[Dict[str, Any]] = []
    last_table_idx = -1

    for item in items:
        item_type = item.get("type", "paragraph")
        page = item.get("page")

        if item_type == "heading":
            text = clean_text(item.get("text", ""))
            if text:
                cleaned.append({
                    "type": "heading",
                    "text": text,
                    "page": page,
                })
                last_table_idx = -1

        elif item_type == "table":
            headers = [clean_text(str(h)) for h in item.get("headers", [])]
            raw_rows = item.get("rows", [])
            cleaned_rows = [
                [clean_text(str(cell)) for cell in row]
                for row in raw_rows
                if any(clean_text(str(cell)) for cell in row)
            ]
            if cleaned_rows or headers:
                cleaned.append({
                    "type": "table",
                    "headers": headers,
                    "rows": cleaned_rows,
                    "page": page,
                    "table_text": table_to_text(headers, cleaned_rows),
                    "footnotes": [],
                })
                last_table_idx = len(cleaned) - 1

        elif item_type == "paragraph":
            text = clean_text(item.get("text", ""))
            if not text:
                continue

            # Check if this paragraph is a footnote belonging to the preceding table
            is_footnote = (
                text.startswith("*")
                or text.startswith("Note:")
                or text.startswith("NOTE:")
                or text.startswith("Conditions:")
            )

            if last_table_idx >= 0 and is_footnote:
                cleaned[last_table_idx]["footnotes"].append(text)
                # Also append footnote to table_text
                cleaned[last_table_idx]["table_text"] += "\n" + text
            else:
                cleaned.append({
                    "type": "paragraph",
                    "text": text,
                    "page": page,
                })
                last_table_idx = -1

    return cleaned
