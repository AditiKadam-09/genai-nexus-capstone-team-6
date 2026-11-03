"""Configuration settings for document ingestion, retrieval, and help desk."""

import os
from pathlib import Path

# Project root detection
_file_path = Path(__file__).resolve()
if "helpdesk" in _file_path.parts:
    # E:\... \helpdesk\ingestion\config.py -> parents[2] is project root
    PROJECT_ROOT = _file_path.parents[2]
elif _file_path.parent.name == "ingestion":
    PROJECT_ROOT = _file_path.parent.parent
else:
    PROJECT_ROOT = _file_path.parent

BASE_DIR = PROJECT_ROOT

# Resolve documents directory across possible layouts
def _find_docs_dir() -> Path:
    env_dir = os.getenv("DOCS_DIR") or os.getenv("DOCUMENTS_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    candidates = [
        PROJECT_ROOT / "documents",
        PROJECT_ROOT / "data" / "rail_capstone_pack" / "documents",
        PROJECT_ROOT / "data" / "documents",
        Path.cwd() / "documents",
        Path.cwd() / "data" / "rail_capstone_pack" / "documents",
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / "documents"

# Resolve database path across possible layouts
def _find_db_path() -> Path:
    env_db = os.getenv("DATABASE_PATH") or os.getenv("TIMETABLE_DB_PATH")
    if env_db and Path(env_db).exists():
        return Path(env_db)
    candidates = [
        PROJECT_ROOT / "database" / "rail_timetable.db",
        PROJECT_ROOT / "data" / "rail_capstone_pack" / "database" / "rail_timetable.db",
        Path.cwd() / "database" / "rail_timetable.db",
        Path.cwd() / "data" / "rail_capstone_pack" / "database" / "rail_timetable.db",
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / "data" / "rail_capstone_pack" / "database" / "rail_timetable.db"

# Resolve persistent ChromaDB directory
def _find_persist_dir() -> Path:
    env_persist = os.getenv("CHROMA_PERSIST_DIRECTORY") or os.getenv("CHROMA_PERSIST_DIR") or os.getenv("INDEX_DIR")
    if env_persist:
        return Path(env_persist)
    # Prefer index/ if index exists (team repo convention), else chroma_db
    if (PROJECT_ROOT / "index").exists():
        return PROJECT_ROOT / "index" / "chroma_db"
    return PROJECT_ROOT / "chroma_db"

DOCUMENTS_DIR = _find_docs_dir()
DOCS_DIR = DOCUMENTS_DIR
DATABASE_PATH = _find_db_path()
TIMETABLE_DB_PATH = DATABASE_PATH
CHROMA_PERSIST_DIRECTORY = _find_persist_dir()
CHROMA_PERSIST_DIR = CHROMA_PERSIST_DIRECTORY

# ChromaDB Settings
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rail_helpdesk_docs")
HNSW_SPACE = "cosine"

# Embedding Model Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384

# Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# LLM / Gemini Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Known Document Titles / Keys
DOC_METADATA_MAP = {
    "indianrail_refund_rules_2015.html": {
        "key": "refund_rules_2015",
        "title": "Revised Refund Rules w.e.f. 12.11.2015 (Indian Railways)",
        "is_html": True,
        "date": "2015-11-12",
    },
    "irctc_refund_rules_jan2026.pdf": {
        "key": "irctc_refund_2026",
        "title": "IRCTC Refund Rules and TDR Filing (updated Jan 2026)",
        "is_html": False,
        "date": "2026-01-19",
    },
    "rb_cc10_2024_advance_reservation.pdf": {
        "key": "rb_cc10_2024",
        "title": "Railway Board Commercial Circular No. 10 of 2024 (Advance Reservation Period 60 Days)",
        "is_html": False,
        "date": "2024-10-16",
    },
    "rb_cc08_2025_tatkal.pdf": {
        "key": "rb_cc08_2025",
        "title": "Railway Board Commercial Circular No. 08 of 2025 (Modification in Tatkal Scheme)",
        "is_html": False,
        "date": "2025-06-10",
    },
    "rb_cc10_2025_reservation_charts.pdf": {
        "key": "rb_cc10_2025",
        "title": "Railway Board Commercial Circular No. 10 of 2025 (Preparation of First Reservation Charts)",
        "is_html": False,
        "date": "2025-07-02",
    },
    "rb_cc08_2026_vande_bharat_sleeper_refund.pdf": {
        "key": "rb_cc08_2026",
        "title": "Railway Board CC No. 08 of 2026 and Gazette G.S.R. 41(E) (Vande Bharat Sleeper Refund Rules)",
        "is_html": False,
        "date": "2026-01-16",
    },
    "pib_2026-03-24_reform_express.html": {
        "key": "pib_reform_2026",
        "title": "PIB Press Release: Indian Railways Passenger-Friendly Initiatives (24-March-2026)",
        "is_html": True,
        "date": "2026-03-24",
    },
}
