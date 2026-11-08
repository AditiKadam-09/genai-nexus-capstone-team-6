from helpdesk.answer import format_context, generate_answer

def sample_chunks():
    return [{
        "chunk_id": "rail_001",
        "text": "Passengers may submit the form at the designated counter.",
        "metadata": {
            "doc": "Passenger_Rules.pdf",
            "page": 12,
            "date": "2025-01-01",
            "section_path": "Passenger Services",
        },
        "rank": 1,
        "score": 0.91,
    }]


def test_format_context_includes_source_metadata():
    context = format_context(sample_chunks())
    assert "rail_001" in context
    assert "page: 12" in context
    assert "Passenger Services" in context
    assert "Passengers may submit" in context


def test_format_context_respects_character_limit():
    context = format_context(sample_chunks(), max_chars=10)
    assert context == ""


def test_generate_answer_uses_supplied_chunks_and_returns_sources():
    result = generate_answer(
        "Where can passengers submit the form?",
        retrieved_chunks=sample_chunks(),
        llm_fn=lambda messages: "Passengers can submit it at the designated counter. [Source: rail_001, page 12]",
    )
    assert result["status"] == "answered"
    assert "designated counter" in result["answer"]
    assert result["sources"] == ["rail_001 (page 12)"]


def test_generate_answer_abstains_when_no_chunks():
    result = generate_answer("What is the refund amount?", retrieved_chunks=[])
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_generate_answer_rejects_blank_question():
    try:
        generate_answer("   ", retrieved_chunks=[])
    except ValueError as exc:
        assert "question" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
