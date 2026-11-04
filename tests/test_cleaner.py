"""Tests for text cleaning, markdown table conversion, and footnote binding."""

import pytest
from ingestion.cleaner import clean_text, table_to_text, clean_items


def test_clean_text_normalizes_spaces_and_newlines():
    raw = "  This   is \xa0a  test.\r\n\r\n\r\nNext   line.  "
    cleaned = clean_text(raw)
    assert cleaned == "This is a test.\n\nNext line."


def test_table_to_text():
    headers = ["Class", "Fare Charge"]
    rows = [["1AC", "Rs 240"], ["2AC", "Rs 200"]]
    md = table_to_text(headers, rows)

    assert "| Class | Fare Charge |" in md
    assert "|---|---|" in md
    assert "| 1AC | Rs 240 |" in md
    assert "| 2AC | Rs 200 |" in md


def test_clean_items_binds_footnotes_to_table():
    items = [
        {"type": "heading", "text": "Refund Rates", "page": 1},
        {
            "type": "table",
            "headers": ["Type", "Fee"],
            "rows": [["Tatkal", "Rs 100*"]],
            "page": 1,
        },
        {"type": "paragraph", "text": "*Subject to GST and distance rules.", "page": 1},
        {"type": "paragraph", "text": "Note: Unused tickets will expire.", "page": 1},
        {"type": "paragraph", "text": "Regular paragraph about something else.", "page": 1},
    ]

    cleaned = clean_items(items)
    # The table item should have captured both footnotes
    table_item = next(it for it in cleaned if it["type"] == "table")
    assert len(table_item["footnotes"]) == 2
    assert "*Subject to GST and distance rules." in table_item["table_text"]
    assert "Note: Unused tickets will expire." in table_item["table_text"]

    # The regular paragraph should still be a standalone paragraph
    paras = [it for it in cleaned if it["type"] == "paragraph"]
    assert len(paras) == 1
    assert "Regular paragraph" in paras[0]["text"]

