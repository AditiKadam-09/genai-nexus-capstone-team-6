from helpdesk.sql import execute_readonly_query, ask_database


print("\n--- TEST 1: Maximum row limit ---")

result = execute_readonly_query(
    "SELECT * FROM schedules"
)

print("Rows returned:", result["row_count"])
print("Truncated:", result["truncated"])

assert result["row_count"] == 50
assert result["truncated"] is True

print("PASSED: Maximum row limit enforced")


print("\n--- TEST 2: Empty question ---")

try:
    ask_database("", lambda q, s: "SELECT 1")
    print("FAILED")
except ValueError as error:
    print("PASSED:", error)


print("\n--- TEST 3: Invalid generator response ---")

try:
    ask_database(
        "Find trains",
        lambda q, s: None
    )
    print("FAILED")
except ValueError as error:
    print("PASSED:", error)


print("\n--- TEST 4: Empty generator response ---")

try:
    ask_database(
        "Find trains",
        lambda q, s: ""
    )
    print("FAILED")
except ValueError as error:
    print("PASSED:", error)


print("\n--- TEST 5: No matching records ---")

result = execute_readonly_query("""
    SELECT number, name
    FROM trains
    WHERE number = '99999'
""")

print("Rows returned:", result["row_count"])

assert result["row_count"] == 0

print("PASSED: Empty results handled correctly")


print("\nAll edge-case tests completed!")