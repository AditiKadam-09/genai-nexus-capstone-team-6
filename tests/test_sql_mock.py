from helpdesk.sql import ask_database


# This simulates the SQL generator that we will later connect to Gemini.
# No API key or internet connection is needed.
def mock_sql_generator(question, schema):

    print("\nQuestion received:", question)
    print("Schema received:", list(schema.keys()))

    if "Mumbai" in question:
        return """
        SELECT number, name, departure, arrival
        FROM trains
        WHERE from_station_code = 'LTT'
        AND to_station_code = 'GWL'
        """

    return "SELECT number, name FROM trains LIMIT 5"


# Test 1: Normal railway question
print("\n--- TEST 1: Mock SQL generation ---")

result = ask_database(
    "Find trains from Mumbai to Gwalior",
    mock_sql_generator
)

print("\nGenerated SQL:")
print(result["sql"])

print("\nDatabase results:")
print(result["results"])


# Test 2: Another question
print("\n--- TEST 2: General train search ---")

result = ask_database(
    "Show me some trains",
    mock_sql_generator
)

print(result["results"])


print("\nAll mock integration tests completed!")