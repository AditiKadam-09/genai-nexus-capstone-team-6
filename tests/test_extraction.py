"""Tests for PDF and HTML extraction modules."""

from pathlib import Path
import pytest

from ingestion.config import DOCUMENTS_DIR
from ingestion.pdf_extractor import PDFExtractor
from ingestion.html_extractor import HTMLExtractor


def test_html_extractor_indianrail_rules():
    extractor = HTMLExtractor()
    html_file = DOCUMENTS_DIR / "indianrail_refund_rules_2015.html"
    assert html_file.exists(), f"Missing test file: {html_file}"

    items = extractor.extract(html_file)
    assert len(items) > 0

    # Ensure page is None for HTML documents
    assert all(it["page"] is None for it in items)

    # Check that the main table was extracted
    tables = [it for it in items if it["type"] == "table"]
    assert len(tables) >= 1
    table = tables[0]
    assert any("Pre revised" in h for h in table["headers"])
    assert any("Revised" in h for h in table["headers"])


def test_html_extractor_pib_release():
    extractor = HTMLExtractor()
    html_file = DOCUMENTS_DIR / "pib_2026-03-24_reform_express.html"
    assert html_file.exists(), f"Missing test file: {html_file}"

    items = extractor.extract(html_file)
    assert len(items) > 0
    assert all(it["page"] is None for it in items)

    # Check for Reform Express headings
    headings = [it["text"] for it in items if it["type"] == "heading"]
    assert any("Reform Express" in h for h in headings)


def test_pdf_extractor_scanned_document():
    extractor = PDFExtractor(use_docling=False, use_ocr=True)
    scanned_pdf = DOCUMENTS_DIR / "rb_cc10_2024_advance_reservation.pdf"
    assert scanned_pdf.exists(), f"Missing test file: {scanned_pdf}"

    items = extractor.extract(scanned_pdf)
    assert len(items) > 0

    # Verify that OCR extracted the English content on page 1
    page1_items = [it for it in items if it.get("page") == 1]
    assert len(page1_items) > 0
    full_p1_text = " ".join(it["text"] for it in page1_items)

    assert "60 days" in full_p1_text or "advance reservations" in full_p1_text.lower()
    assert "Taj Express" in full_p1_text or "taj express" in full_p1_text.lower()


def test_pdf_extractor_file_not_found():
    extractor = PDFExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract(Path("non_existent_document.pdf"))

