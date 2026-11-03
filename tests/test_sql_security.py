from helpdesk.sql import execute_readonly_query

unsafe_queries = [
    "PRAGMA table_info(trains)",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM unknown_table",
    "SELECT load_extension('x')",
    "ATTACH DATABASE ':memory:' AS extra",
    "CREATE TABLE test_table (id INTEGER)"
]

print("\n--- SQL SECURITY TESTS ---")

for query in unsafe_queries:
    try:
        execute_readonly_query(query)
        print(f"FAILED: {query}")
    except Exception as error:
        print(f"PASSED: Rejected unsafe query: {query}")

print("\nSecurity tests completed!")
print("\n--- TEST: Accessing unauthorized table ---")

try:
    execute_readonly_query(
        "SELECT * FROM sqlite_master"
    )
    print("FAILED: Unauthorized table access allowed")
except ValueError as error:
    print("PASSED: Unauthorized table access blocked")
    print(error)