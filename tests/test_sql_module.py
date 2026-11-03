
from helpdesk.sql import (
    get_database_schema,
    execute_readonly_query,
    validate_sql
)

print("\n--- DATABASE SCHEMA ---")
schema = get_database_schema()

for table, details in schema.items():
    print(f"\n{table} ({details['type']})")

    for column in details["columns"]:
        print(" ", column["name"], column["type"])


print("\n--- TEST 1: Find trains ---")

result = execute_readonly_query("""
    SELECT number, name, departure, arrival
    FROM trains
    WHERE from_station_code = 'LTT'
      AND to_station_code = 'GWL'
""")

print(result)


print("\n--- TEST 2: Find intermediate journeys ---")

result = execute_readonly_query("""
    SELECT
        t.number,
        t.name,
        source.name AS source_station,
        destination.name AS destination_station,
        source_stop.departure,
        destination_stop.arrival
    FROM stops source_stop

    JOIN stops destination_stop
        ON source_stop.train_number = destination_stop.train_number
        AND source_stop.stop_sequence < destination_stop.stop_sequence

    JOIN trains t
        ON t.number = source_stop.train_number

    JOIN stations source
        ON source.code = source_stop.station_code

    JOIN stations destination
        ON destination.code = destination_stop.station_code

    WHERE source_stop.station_code = 'KYN'
      AND destination_stop.station_code = 'BPL'
    LIMIT 10
""")

print(result)


print("\n--- TEST 3: Reject unsafe queries ---")

unsafe_queries = [
    "DELETE FROM trains",
    "DROP TABLE trains",
    "UPDATE trains SET name = 'Test'",
    "SELECT * FROM trains; DELETE FROM trains"
]

for query in unsafe_queries:
    try:
        validate_sql(query)
        print("Unexpectedly accepted:", query)
    except ValueError as error:
        print("Correctly rejected:", query)


print("\nAll initial tests completed!")
