# genai-nexus-capstone-team-6

Indian Railways Passenger Help Desk AI Capstone Project.

---

## Module: Document Ingestion & Index Building

**Assigned Responsibility**: Document Ingestion & Index Building (Bhargavi Sri Gurram)

### Overview
This module extracts, cleans, chunks, embeds, and indexes official Indian Railways documents, circulars, and gazette notifications into a persistent ChromaDB vector store.

### Features
1. **Multi-Source Extraction**:
   - **Docling Integration (`PDFExtractor`)**: Extracts structured AST elements, headings, paragraphs, and markdown tables from digital PDFs (`irctc_refund_rules_jan2026.pdf`, `rb_cc08_2026_vande_bharat_sleeper_refund.pdf`, `rb_cc08_2025_tatkal.pdf`).
   - **RapidOCR Fallback**: Automatically processes scanned circulars without digital text layers (`rb_cc10_2024_advance_reservation.pdf`, `rb_cc10_2025_reservation_charts.pdf`) and scanned Hindi pages via PyMuPDF rendering.
   - **HTML Extraction (`HTMLExtractor`)**: Parses policy releases (`indianrail_refund_rules_2015.html`, `pib_2026-03-24_reform_express.html`) with section tracking and explicit `"page": null` citations.

2. **Cleaning & Footnote Binding (`cleaner.py`)**:
   - Normalizes unicode whitespace, dashes, and bullet points.
   - Formats tabular data into clean Markdown tables.
   - Detects footnote markers (`*`, `Note:`, `Explanation:`) following tables and binds them directly into the table chunk so refund conditions stay intact with their fee matrices.

3. **Contextual Hierarchical Chunking (`chunker.py`)**:
   - Pre-pends contextual document and section headers (`Document: <doc>\nSection: <title>`) to every chunk.
   - Splits oversized tables row-by-row while preserving header rows on every chunk.
   - Configurable character chunk size (default: 500) and overlap (default: 100).

4. **Normalized Dense Embeddings (`embedder.py`)**:
   - Uses `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors).
   - Generates L2-normalized embeddings for fast cosine similarity.
   - Applies search query prefix: `"Represent this sentence for searching relevant passages: "`.

5. **Persistent ChromaDB Indexing (`indexer.py`)**:
   - Stores vectors in persistent ChromaDB at `index/chroma_db` (or `./chroma_db`) under collection `rail_helpdesk_docs`.
   - Uses `cosine` space distance metric.
   - Deterministic chunk IDs (`{doc}_{chunk_index:04d}`) prevent duplicate indexing.

---

### Usage

#### 1. Building the Index via CLI
```bash
python build_index.py --reindex
```

Options:
- `--docs-dir <path>`: Directory containing PDFs and HTML files (default: `documents` or `data/rail_capstone_pack/documents`).
- `--persist-dir <path>`: Directory where ChromaDB files are saved (default: `index/chroma_db`).
- `--collection <name>`: Collection name (default: `rail_helpdesk_docs`).
- `--reindex`: Wipe existing collection and re-index all documents.
- `--chunk-size <int>`: Max chunk size in characters (default: `500`).
- `--chunk-overlap <int>`: Chunk overlap in characters (default: `100`).

#### 2. Programmatic Ingestion API
```python
from helpdesk.ingest import ingest_documents, IngestionPipeline, DocumentEmbedder

# Run full ingestion
summary = ingest_documents(reindex=True)
print(summary)
```

---

### Running Ingestion Tests
```bash
python -m pytest tests/test_cleaner.py tests/test_chunker.py tests/test_extraction.py tests/test_indexer.py tests/test_ingest.py -v
```
