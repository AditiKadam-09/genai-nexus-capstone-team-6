"""HTML document extraction module for Indian Railways refund rules and PIB press releases.

Strictly follows project requirement:
Two documents are HTML, so they have no page numbers: cite them with 'page': None (null in JSON).
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup

from ingestion.cleaner import clean_text

logger = logging.getLogger(__name__)


class HTMLExtractor:
    """Extracts structured content from HTML documents with page=None."""

    def extract(self, file_path: Path) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"HTML file does not exist: {file_path}")

        logger.info(f"Extracting HTML: {file_path.name}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")
        items: List[Dict[str, Any]] = []

        # Remove script and style tags
        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()

        if file_path.name == "indianrail_refund_rules_2015.html":
            items = self._extract_indianrail_rules(soup)
        elif file_path.name == "pib_2026-03-24_reform_express.html":
            items = self._extract_pib_release(soup)
        else:
            items = self._extract_generic_html(soup)

        logger.info(f"Extracted {len(items)} items from HTML {file_path.name}")
        return items

    def _extract_indianrail_rules(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract table and introductory text from indianrail_refund_rules_2015.html."""
        items: List[Dict[str, Any]] = []

        items.append({
            "type": "heading",
            "text": "Revised Refund Rules w.e.f. 12.11.2015 (Indian Railways)",
            "page": None,
        })

        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue

            # First row might be header
            first_row_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
            if len(first_row_cells) > 1 and "Pre revised" in " ".join(first_row_cells):
                headers = first_row_cells
                data_rows = []
                for row in rows[1:]:
                    cells = [clean_text(c.get_text(separator="\n", strip=True)) for c in row.find_all(["td", "th"])]
                    if any(cells):
                        data_rows.append(cells)
                items.append({
                    "type": "table",
                    "headers": headers,
                    "rows": data_rows,
                    "page": None,
                })
            else:
                # Text inside single table cells
                for r in rows:
                    for c in r.find_all("td"):
                        text = clean_text(c.get_text(separator="\n", strip=True))
                        if text:
                            items.append({
                                "type": "paragraph",
                                "text": text,
                                "page": None,
                            })

        return items

    def _extract_pib_release(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract structured sections and paragraphs from PIB press release."""
        items: List[Dict[str, Any]] = []

        title_el = soup.find("h2") or soup.find("h1")
        if title_el:
            items.append({
                "type": "heading",
                "text": clean_text(title_el.get_text()),
                "page": None,
            })

        # Process main release content
        content_div = (
            soup.find("div", class_="ReleaseContent")
            or soup.find("div", id="divRelease")
            or soup.find("body")
        )

        if not content_div:
            return items

        for elem in content_div.find_all(["h1", "h2", "h3", "h4", "p", "table", "ul"]):
            if elem.name in ["h1", "h2", "h3", "h4"]:
                text = clean_text(elem.get_text())
                if text:
                    items.append({
                        "type": "heading",
                        "text": text,
                        "page": None,
                    })
            elif elem.name == "table":
                rows = elem.find_all("tr")
                if rows:
                    headers = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
                    data_rows = [
                        [c.get_text(strip=True) for c in r.find_all(["td", "th"])]
                        for r in rows[1:]
                    ]
                    items.append({
                        "type": "table",
                        "headers": headers,
                        "rows": data_rows,
                        "page": None,
                    })
            elif elem.name in ["p", "ul"]:
                text = clean_text(elem.get_text())
                if text and len(text) > 5:
                    # Check if paragraph is actually a sub-heading (bold or short uppercase)
                    b_child = elem.find(["b", "strong"])
                    if b_child and len(clean_text(b_child.get_text())) == len(text) and len(text) < 80:
                        items.append({
                            "type": "heading",
                            "text": text,
                            "page": None,
                        })
                    else:
                        items.append({
                            "type": "paragraph",
                            "text": text,
                            "page": None,
                        })

        return items

    def _extract_generic_html(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Generic fallback extractor for HTML pages."""
        items: List[Dict[str, Any]] = []
        body = soup.find("body") or soup
        for elem in body.find_all(["h1", "h2", "h3", "h4", "p", "table"]):
            if elem.name in ["h1", "h2", "h3", "h4"]:
                text = clean_text(elem.get_text())
                if text:
                    items.append({"type": "heading", "text": text, "page": None})
            elif elem.name == "table":
                rows = elem.find_all("tr")
                if rows:
                    headers = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
                    data_rows = [
                        [c.get_text(strip=True) for c in r.find_all(["td", "th"])]
                        for r in rows[1:]
                    ]
                    items.append({"type": "table", "headers": headers, "rows": data_rows, "page": None})
            elif elem.name == "p":
                text = clean_text(elem.get_text())
                if text:
                    items.append({"type": "paragraph", "text": text, "page": None})
        return items
