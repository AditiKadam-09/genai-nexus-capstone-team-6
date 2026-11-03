from helpdesk.sql import execute_readonly_query


test_cases = {

    "1. Direct train search": """
        SELECT number, name, departure, arrival
        FROM trains
        WHERE from_station_code = 'LTT'
        AND to_station_code = 'GWL'
    """,

    "2. Intermediate station journey": """
        SELECT
            t.number,
            t.name,
            source_station.name AS source,
            destination_station.name AS destination,
            source_stop.departure,
            destination_stop.arrival
        FROM stops AS source_stop
        JOIN stops AS destination_stop
            ON source_stop.train_number = destination_stop.train_number
            AND source_stop.stop_sequence < destination_stop.stop_sequence
        JOIN trains AS t
            ON t.number = source_stop.train_number
        JOIN stations AS source_station
            ON source_station.code = source_stop.station_code
        JOIN stations AS destination_station
            ON destination_station.code = destination_stop.station_code
        WHERE source_station.code = 'KYN'
        AND destination_station.code = 'BPL'
        LIMIT 10
    """,

    "3. Train route": """
        SELECT
            s.stop_sequence,
            st.name,
            s.day,
            s.arrival,
            s.departure
        FROM schedules AS s
        JOIN stations AS st
            ON st.code = s.station_code
        WHERE s.train_number = '01101'
        ORDER BY s.stop_sequence
        LIMIT 15
    """,

    "4. Trains stopping at Nashik Road": """
        SELECT
            t.number,
            t.name,
            st.name AS station
        FROM stops AS s
        JOIN trains AS t
            ON t.number = s.train_number
        JOIN stations AS st
            ON st.code = s.station_code
        WHERE st.name LIKE '%NASIK ROAD%'
        LIMIT 10
    """,

    "5. Trains with AC 3-tier": """
        SELECT number, name
        FROM trains
        WHERE has_3a = 1
        LIMIT 10
    """,

    "6. Train type information": """
        SELECT
            t.number,
            t.name,
            tt.meaning
        FROM trains AS t
        JOIN train_types AS tt
            ON tt.type_code = t.type_code
        LIMIT 10
    """
}


for test_name, query in test_cases.items():

    print(f"\n--- {test_name} ---")

    try:
        result = execute_readonly_query(query)

        print("Columns:", result["columns"])
        print("Rows returned:", result["row_count"])
        print("Truncated:", result["truncated"])

        for row in result["rows"][:3]:
            print(row)

        if result["row_count"] == 0:
            print("WARNING: No matching records found.")

    except Exception as error:
        print("TEST FAILED:", error)


print("\nAll railway SQL tests completed!")