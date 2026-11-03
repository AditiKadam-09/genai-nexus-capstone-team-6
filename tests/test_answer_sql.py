from helpdesk.sql import answer_from_sql


def fake_llm(replies):
    iterator = iter(replies)
    return lambda prompt: next(iterator)


JOIN = (
    "SELECT t.number, t.name, a.departure, b.arrival "
    "FROM stops a JOIN stops b ON b.train_number = a.train_number "
    "AND b.stop_sequence > a.stop_sequence "
    "JOIN trains t ON t.number = a.train_number "
    "WHERE a.station_code = 'LTT' AND b.station_code = 'GWL' "
    "ORDER BY a.departure"
)

print("--- TESTS: answer_from_sql ---")

result = answer_from_sql("q", llm=fake_llm([JOIN, "Three trains."]))
assert result["dont_know"] is False and result["sql"] == JOIN
assert result["sources"] == []
print("PASSED: normal answer")

result = answer_from_sql("q", llm=fake_llm(["SELECT nope FROM trains", JOIN, "ok"]))
assert result["dont_know"] is False
print("PASSED: retry after an SQL error")

result = answer_from_sql("q", llm=fake_llm(["REFUSE"]))
assert result["dont_know"] is True and result["sql"] is None
print("PASSED: unsafe request refused")

result = answer_from_sql("q", llm=fake_llm(["NO_DATA"]))
assert result["dont_know"] is True
print("PASSED: data not in the database")

result = answer_from_sql("q", llm=fake_llm(["SELECT x FROM y"] * 3))
assert result["dont_know"] is True
print("PASSED: gives up after 3 bad queries")

result = answer_from_sql("q", llm=fake_llm(["SELECT number FROM trains WHERE number = '99999'"]))
assert result["dont_know"] is True
print("PASSED: zero rows")

result = answer_from_sql("q", llm=fake_llm([JOIN]))   # second call fails
assert result["dont_know"] is False and "2016 timetable" in result["answer"]
print("PASSED: fallback when the LLM cannot phrase the answer")

print("All answer_from_sql tests passed")