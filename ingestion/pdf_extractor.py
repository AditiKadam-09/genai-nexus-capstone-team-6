"""PDF extraction module using Docling and RapidOCR.

Extracts headings, tables, and paragraphs from digital and scanned PDFs.
Provides robust fallback to RapidOCR on scanned pages and PyMuPDF text extraction.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pymupdf as fitz
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts structured content from PDF documents."""

    def __init__(self, use_docling: bool = True, use_ocr: bool = True):
        self.use_docling = use_docling
        self.use_ocr = use_ocr
        self._ocr = None
        self._docling_converter = None

    @property
    def ocr(self) -> RapidOCR:
        if self._ocr is None:
            self._ocr = RapidOCR()
        return self._ocr

    @property
    def docling_converter(self) -> DocumentConverter:
        if self._docling_converter is None:
            self._docling_converter = DocumentConverter()
        return self._docling_converter

    def extract(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract structured items from a PDF file.

        Returns list of dicts:
        - {'type': 'heading' | 'table' | 'paragraph', 'text': ..., 'page': int}
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {file_path}")

        logger.info(f"Extracting PDF: {file_path.name}")

        # Check total pages and quick text check with PyMuPDF
        try:
            pdf_doc = fitz.open(str(file_path))
            total_pages = len(pdf_doc)
            pages_with_text = 0
            for page in pdf_doc:
                if page.get_text().strip():
                    pages_with_text += 1
            pdf_doc.close()
        except Exception as e:
            logger.error(f"Error opening PDF {file_path.name} with PyMuPDF: {e}")
            raise ValueError(f"Invalid or corrupted PDF file: {file_path}") from e

        items: List[Dict[str, Any]] = []

        # 1. Try Docling extraction first if enabled
        if self.use_docling and pages_with_text > 0:
            try:
                logger.info(f"Running Docling converter on {file_path.name}...")
                result = self.docling_converter.convert(str(file_path))
                docling_doc = result.document
                items = self._docling_to_simple(docling_doc)
                logger.info(f"Docling extracted {len(items)} items from {file_path.name}")
            except Exception as e:
                logger.warning(f"Docling conversion failed for {file_path.name}: {e}. Falling back to PyMuPDF.")
                items = []

        # 2. Check if all pages are covered or if any pages need OCR
        extracted_pages = {item["page"] for item in items if item.get("page") is not None}
        missing_pages = [p for p in range(1, total_pages + 1) if p not in extracted_pages]

        # If pages are completely missing or document is purely scanned (pages_with_text == 0)
        if missing_pages and self.use_ocr:
            logger.info(f"Pages requiring OCR in {file_path.name}: {missing_pages}")
            ocr_items = self._extract_with_ocr(file_path, target_pages=missing_pages)
            items.extend(ocr_items)

        # 3. If items are still empty, run full PyMuPDF + OCR fallback
        if not items:
            logger.warning(f"No items extracted by Docling for {file_path.name}. Running PyMuPDF + OCR extractor.")
            items = self._extract_with_pymupdf_and_ocr(file_path)

        # Sort items primarily by page number
        items.sort(key=lambda x: (x.get("page") or 0))
        logger.info(f"Total extracted items for {file_path.name}: {len(items)}")
        return items

    def _docling_to_simple(self, doc: DoclingDocument) -> List[Dict[str, Any]]:
        """Convert Docling document items to simple list of structured items."""
        items = []
        for element, level in doc.iterate_items():
            page = None
            if hasattr(element, "prov") and element.prov:
                page = element.prov[0].page_no

            label = element.label.value if hasattr(element.label, "value") else str(element.label)

            if label in ("section_header", "title"):
                text = getattr(element, "text", "")
                if text and text.strip():
                    items.append({
                        "type": "heading",
                        "text": text.strip(),
                        "page": page,
                    })
            elif label == "table":
                try:
                    df = element.export_to_dataframe(doc=doc)
                    headers = [str(col) for col in df.columns]
                    rows = [[str(cell) for cell in row] for row in df.values.tolist()]
                    items.append({
                        "type": "table",
                        "headers": headers,
                        "rows": rows,
                        "page": page,
                    })
                except Exception as e:
                    logger.debug(f"Could not export table to dataframe: {e}. Using text representation.")
                    text = getattr(element, "text", "")
                    if text and text.strip():
                        items.append({
                            "type": "paragraph",
                            "text": text.strip(),
                            "page": page,
                        })
            else:
                text = getattr(element, "text", "")
                if text and text.strip():
                    items.append({
                        "type": "paragraph",
                        "text": text.strip(),
                        "page": page,
                    })

        return items

    def _extract_with_ocr(self, file_path: Path, target_pages: List[int]) -> List[Dict[str, Any]]:
        """Extract text from scanned pages using RapidOCR and PyMuPDF."""
        items: List[Dict[str, Any]] = []
        try:
            doc = fitz.open(str(file_path))
            for page_num in target_pages:
                if page_num < 1 or page_num > len(doc):
                    continue
                page = doc[page_num - 1]
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                result, _ = self.ocr(img_bytes)

                if result:
                    lines = [line_info[1].strip() for line_info in result if line_info and len(line_info) > 1]
                    # Structure OCR text: detect headings vs paragraphs
                    page_items = self._ocr_lines_to_items(lines, page_num)
                    items.extend(page_items)
            doc.close()
        except Exception as e:
            logger.error(f"Error performing OCR on {file_path.name}: {e}")

        return items

    def _ocr_lines_to_items(self, lines: List[str], page_num: int) -> List[Dict[str, Any]]:
        """Structure OCR lines into headings and paragraphs for a page."""
        items: List[Dict[str, Any]] = []
        current_para: List[str] = []

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            # Detect heading-like lines
            is_heading = (
                (len(line_clean) < 80 and (
                    line_clean.isupper()
                    or "GOVERNMENT OF INDIA" in line_clean
                    or "MINISTRY OF RAILWAYS" in line_clean
                    or "COMMERCIAL CIRCULAR" in line_clean.upper()
                    or "NOTIFICATION" in line_clean.upper()
                    or "SUB:" in line_clean.upper()
                    or re_match_heading(line_clean)
                ))
            )

            if is_heading:
                if current_para:
                    items.append({
                        "type": "paragraph",
                        "text": " ".join(current_para),
                        "page": page_num,
                    })
                    current_para = []
                items.append({
                    "type": "heading",
                    "text": line_clean,
                    "page": page_num,
                })
            else:
                current_para.append(line_clean)

        if current_para:
            items.append({
                "type": "paragraph",
                "text": " ".join(current_para),
                "page": page_num,
            })

        return items

    def _extract_with_pymupdf_and_ocr(self, file_path: Path) -> List[Dict[str, Any]]:
        """Fallback extractor using PyMuPDF blocks and RapidOCR."""
        items: List[Dict[str, Any]] = []
        doc = fitz.open(str(file_path))

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text().strip()

            if not text and self.use_ocr:
                # Scanned page - use OCR
                pix = page.get_pixmap(dpi=200)
                result, _ = self.ocr(pix.tobytes("png"))
                if result:
                    lines = [line_info[1].strip() for line_info in result if line_info and len(line_info) > 1]
                    items.extend(self._ocr_lines_to_items(lines, page_num))
            else:
                # Text layer available: extract blocks
                blocks = page.get_text("blocks")
                for b in blocks:
                    block_text = b[4].strip()
                    if not block_text:
                        continue
                    # Check if block looks like heading
                    if len(block_text.splitlines()) == 1 and (len(block_text) < 70 or block_text.isupper()):
                        items.append({
                            "type": "heading",
                            "text": block_text,
                            "page": page_num,
                        })
                    else:
                        items.append({
                            "type": "paragraph",
                            "text": block_text,
                            "page": page_num,
                        })

        doc.close()
        return items


def re_match_heading(text: str) -> bool:
    """Check if text matches standard Indian Railway heading patterns."""
    text_lower = text.lower()
    return any(text_lower.startswith(prefix) for prefix in [
        "sub:", "subject:", "commercial circular", "no.", "ref:", "1.", "2.", "3.", "4.", "5."
    ])
