"""Chunking module implementing section building, table-aware splitting,
recursive paragraph splitting, and overlap packing.

Follows the curriculum guidelines from Module 2:
- Contextual prefix added to embedding text:
    "Document: <doc_title>\nSection: <section_path>\n\n<chunk_text>"
- Table header preservation across multi-row chunks
- Footnote preservation with preceding tables
- Rich metadata (doc, page, section_title, section_path, split_method)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from ingestion.cleaner import table_to_text, clean_text
from ingestion.config import CHUNK_SIZE, CHUNK_OVERLAP, DOC_METADATA_MAP

logger = logging.getLogger(__name__)


def build_sections(items: List[Dict[str, Any]], doc_name: str = "") -> List[Dict[str, Any]]:
    """Group items into hierarchical sections based on headings."""
    sections: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, Any]] = None
    parent_title: Optional[str] = None
    last_page: Optional[int] = None

    default_title = DOC_METADATA_MAP.get(doc_name, {}).get("title", doc_name or "General")

    for item in items:
        item_type = item.get("type")
        page = item.get("page")

        if item_type == "heading":
            heading_text = clean_text(item.get("text", ""))
            if not heading_text:
                continue

            if page is not None and page != last_page:
                last_page = page
                parent_title = heading_text
                path = [default_title, heading_text]
            else:
                if parent_title and parent_title != heading_text:
                    path = [default_title, parent_title, heading_text]
                else:
                    path = [default_title, heading_text]

            current_section = {
                "title": heading_text,
                "path": path,
                "page": page,
                "items": [],
            }
            sections.append(current_section)
        else:
            # Start a continuation section when the text moves to a new page,
            # so every chunk gets the page its text is really on.
            page_changed = (
                current_section is not None
                and page is not None
                and current_section.get("page") is not None
                and page != current_section["page"]
            )
            if current_section is None or page_changed:
                if current_section is None:
                    title, path = default_title, [default_title]
                else:
                    title, path = current_section["title"], current_section["path"]
                current_section = {
                    "title": title,
                    "path": path,
                    "page": page,
                    "items": [],
                }
                sections.append(current_section)
            current_section["items"].append(item)

    return sections


def split_table(headers: List[str], rows: List[List[str]], chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split a large table into multiple markdown chunks, preserving headers in each chunk."""
    if not rows:
        return [table_to_text(headers, [])]

    groups = []
    current_rows = []

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate_text = table_to_text(headers, candidate_rows)

        if current_rows and len(candidate_text) > chunk_size:
            groups.append(table_to_text(headers, current_rows))
            current_rows = []

        current_rows.append(row)

    if current_rows:
        groups.append(table_to_text(headers, current_rows))

    return groups


def recursive_split(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text recursively using natural boundaries: double newline, sentences, spaces."""
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]

    for separator in ["\n\n", "\n", ". ", " "]:
        parts = text.split(separator)
        if len(parts) > 1:
            pieces = []
            for i, part in enumerate(parts):
                piece = part.strip()
                if not piece:
                    continue
                if separator == ". " and i < len(parts) - 1:
                    piece += "."
                pieces.extend(recursive_split(piece, chunk_size=chunk_size))
            return pieces

    return [text]


def pack_with_overlap(pieces: List[str], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Pack small text pieces into chunks up to chunk_size, carrying over chunk_overlap tail."""
    chunks = []
    current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        if current != "" and len(current) + 1 + len(piece) > chunk_size:
            chunks.append(current)
            # Carry over overlap
            overlap = current[-chunk_overlap:]
            space_idx = overlap.find(" ")
            if space_idx != -1:
                overlap = overlap[space_idx + 1:]

            if len(overlap) + 1 + len(piece) <= chunk_size:
                current = overlap
            else:
                current = ""

        if current == "":
            current = piece
        else:
            current = current + "\n" + piece

    if current != "":
        chunks.append(current)

    return chunks


def section_text(section: Dict[str, Any]) -> str:
    """Concatenate all item texts in a section."""
    parts = []
    for item in section.get("items", []):
        if item.get("type") == "table":
            parts.append(item.get("table_text", table_to_text(item.get("headers", []), item.get("rows", []))))
        else:
            text = item.get("text", "")
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def chunk_section(section: Dict[str, Any], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, str]]:
    """Chunk a single section into text pieces tagged with their split_method."""
    items = section.get("items", [])
    if not items:
        # Check if heading alone holds information
        heading_text = section.get("title", "")
        if heading_text:
            return [(heading_text, "heading_only")]
        return []

    full_text = section_text(section)
    if len(full_text) <= chunk_size:
        return [(full_text, "whole_section")]

    chunks: List[Tuple[str, str]] = []
    text_paras: List[str] = []
    last_paragraph = ""  # the sentence just before a table (its condition)

    for item in items:
        item_type = item.get("type")
        if item_type == "table":
            headers = item.get("headers", [])
            rows = item.get("rows", [])
            table_groups = split_table(headers, rows, chunk_size=chunk_size)
            # Add footnotes if present
            footnotes = item.get("footnotes", [])
            if footnotes and table_groups:
                table_groups[-1] += "\n" + "\n".join(footnotes)
            # Keep the lead-in sentence with the table, so the rule keeps its condition
            lead_in = last_paragraph if 0 < len(last_paragraph) <= 400 else ""
            for g in table_groups:
                group_text = (lead_in + "\n\n" + g) if lead_in else g
                if len(group_text) > chunk_size * 2:
                    # One huge table row cannot be split between rows (some web pages
                    # wrap their whole text in a layout table): cut it like normal text.
                    pieces = pack_with_overlap(recursive_split(group_text, chunk_size), chunk_size, chunk_overlap)
                    for piece in pieces:
                        chunks.append((piece, "table_rows"))
                else:
                    chunks.append((group_text, "table_rows"))
            last_paragraph = ""
        elif item_type == "paragraph":
            p_text = clean_text(item.get("text", ""))
            if p_text:
                text_paras.append(p_text)
                last_paragraph = p_text

    if text_paras:
        joined_text = "\n\n".join(text_paras)
        splits = recursive_split(joined_text, chunk_size=chunk_size)
        packed = pack_with_overlap(splits, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in packed:
            chunks.append((chunk, "recursive"))

    return chunks


class DocumentChunker:
    """Orchestrates chunk generation across all sections of a document."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_name: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk document items into indexed chunks with rich metadata."""
        doc_meta = DOC_METADATA_MAP.get(doc_name, {
            "key": Path(doc_name).stem,
            "title": doc_name,
            "is_html": doc_name.endswith(".html"),
        })

        doc_key = doc_meta["key"]
        doc_title = doc_meta["title"]
        is_html = doc_meta.get("is_html", False)

        sections = build_sections(items, doc_name=doc_name)
        chunks: List[Dict[str, Any]] = []

        for section in sections:
            section_chunks = chunk_section(
                section,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            for chunk_text, split_method in section_chunks:
                chunk_text = clean_text(chunk_text)
                if not chunk_text:
                    continue

                chunk_id = f"{doc_key}_{str(len(chunks) + 1).zfill(4)}"
                section_path_str = " > ".join(section["path"])
                context = f"Document: {doc_title}\nSection: {section_path_str}"
                embedding_text = f"{context}\n\n{chunk_text}"

                # Ensure page is None for HTML documents
                page_val = None if is_html else section.get("page")

                chunk_entry = {
                    "chunk_id": chunk_id,
                    "doc": doc_name,
                    "page": page_val,
                    "date": doc_meta.get("date"),
                    "text": chunk_text,
                    "embedding_text": embedding_text,
                    "section_title": section.get("title", ""),
                    "section_path": section_path_str,
                    "split_method": split_method,
                    "metadata": {
                        "doc": doc_name,
                        "page": page_val if page_val is not None else -1,  # ChromaDB metadata int fallback
                        "is_html": is_html,
                        "date": doc_meta.get("date"),
                        "section_title": section.get("title", ""),
                        "section_path": section_path_str,
                        "split_method": split_method,
                    }
                }
                chunks.append(chunk_entry)

        logger.info(f"Generated {len(chunks)} chunks for {doc_name}")
        return chunks
