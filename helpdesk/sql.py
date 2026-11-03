import json
import sqlite3
import re
from pathlib import Path


# Project and database paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "rail_capstone_pack"
    / "database"
    / "rail_timetable.db"
)

MAX_ROWS = 50
MAX_PROGRESS_CALLBACKS = 1000

ALLOWED_TABLES = {
    "stations",
    "train_types",
    "trains",
    "schedules",
    "stops",
}


def get_database_schema(db_path=DB_PATH):
    """Read the actual SQLite database schema."""

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(
        f"{Path(db_path).resolve().as_uri()}?mode=ro",
        uri=True
    )

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
        """)

        schema = {}

        for name, object_type in cursor.fetchall():

            if name not in ALLOWED_TABLES:
                continue

            cursor.execute(
                f'PRAGMA table_info("{name}")'
            )

            columns = [
                {
                    "name": row[1],
                    "type": row[2],
                    "primary_key": bool(row[5])
                }
                for row in cursor.fetchall()
            ]

            schema[name] = {
                "type": object_type,
                "columns": columns
            }

        return schema

    finally:
        conn.close()


def validate_sql(sql):
    """Perform basic SQL validation before execution."""

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    sql = sql.strip()

    # Only SELECT and WITH queries are accepted.
    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    # Reject multiple SQL statements.
    if ";" in sql.rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed.")

    return sql.rstrip(";").strip()


def _sql_authorizer(action, arg1, arg2, database, trigger):
    """Allow only safe database reading operations."""

    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }

    if hasattr(sqlite3, "SQLITE_RECURSIVE"):
        allowed_actions.add(sqlite3.SQLITE_RECURSIVE)

    if action not in allowed_actions:
        return sqlite3.SQLITE_DENY

    # Restrict reads to the official railway dataset.
    if action == sqlite3.SQLITE_READ:
        if arg1 not in ALLOWED_TABLES:
            return sqlite3.SQLITE_DENY

    # Block potentially dangerous SQLite functions.
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (arg2 or "").lower()

        blocked_functions = {
            "load_extension",
            "readfile",
            "writefile",
        }

        if function_name in blocked_functions:
            return sqlite3.SQLITE_DENY

    return sqlite3.SQLITE_OK


def execute_readonly_query(sql, db_path=DB_PATH):
    """Execute a validated query using read-only SQLite access."""

    sql = validate_sql(sql)

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(
        f"{Path(db_path).resolve().as_uri()}?mode=ro",
        uri=True
    )

    try:
        conn.set_authorizer(_sql_authorizer)
        # Interrupt queries that perform too many SQLite operations.
        query_steps = 0

        def limit_query_execution():
            nonlocal query_steps
            query_steps += 1

            if query_steps > MAX_PROGRESS_CALLBACKS:
                return 1

            return 0

        conn.set_progress_handler(limit_query_execution, 1000)

        cursor = conn.cursor()
        cursor.execute(sql)

        if cursor.description is None:
            raise ValueError("Query did not return a result set.")

        columns = [
            description[0]
            for description in cursor.description
        ]

        rows = cursor.fetchmany(MAX_ROWS + 1)

        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated
        }

    except sqlite3.DatabaseError as error:
        raise ValueError(
            f"SQL execution failed: {error}"
        ) from error

    finally:
        conn.close()

def build_sql_prompt(question, schema):
    """
    Build instructions for an LLM to generate railway SQL.
    """

    schema_text = json.dumps(schema, indent=2)

    prompt = f"""
You are an expert SQLite Text-to-SQL assistant
for a Railway Passenger Help Desk.

Your task is to convert the user's question into
one accurate SQLite SELECT query.

DATABASE SCHEMA:
{schema_text}

IMPORTANT DATABASE RULES:

1. Use only the tables and columns provided in the schema.

2. The trains table contains train details.
   - number is TEXT. Preserve leading zeros.
   - from_station_code and to_station_code identify
     source and destination stations.
   - has_1a, has_2a, has_3a, has_sl, has_cc and
     has_fc indicate coach availability (1 or 0).

3. The stations table contains station codes and names.
   - Join using stations.code and the relevant
     station_code column.
   - Do not invent station codes.
   - Resolve station names using the database.

4. The schedules table contains the complete train route.
   - Use stop_sequence to maintain route order.
   - Use schedules when the question asks for all
     scheduled stations, including pass-through stations.

5. The stops view contains only actual halting stations.
   - Use stops when the question asks where a train stops.
   - For journeys between two intermediate stations,
     join stops with itself using train_number.
   - Ensure source_stop.stop_sequence <
     destination_stop.stop_sequence.

6. The day column represents the journey day offset,
   NOT the weekday.

7. Arrival and departure times are stored as text.
   They may be NULL at the starting or ending station.

8. Use train_types to retrieve train type meanings.
   Join train_types.type_code with trains.type_code.

9. Generate only read-only SELECT or WITH queries.
   Never generate INSERT, UPDATE, DELETE, DROP,
   ALTER, CREATE, PRAGMA or ATTACH statements.

10. Do not invent train details or assume information
    that is not available in the database.

11. Use appropriate JOINs and aliases.
    Avoid unnecessary columns.
    Use LIMIT 50 or less.

12. Return only the SQL query.
    Do not include explanations, Markdown fences,
    or any additional text.

13. Resolve station names using the stations table.
    Never invent or assume station codes.

14. If a station name is ambiguous, do not randomly
    select one station. Generate a query to retrieve
    matching station codes and names for clarification.

15. If the requested information is not available
    in the database, do not fabricate an answer.

16. For train numbers, always compare using TEXT
    values and preserve leading zeros.

17. Use case-insensitive station-name matching
    when appropriate.

USER QUESTION:
{question}

SQL QUERY:
"""

    return prompt

def ask_database(question, sql_generator):
    """
    Generate SQL and execute it safely.

    sql_generator must accept:
        question: str
        schema: dict

    It must return a SQL query as a string.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    schema = get_database_schema()

    generated_sql = sql_generator(question, schema)
    if not isinstance(generated_sql, str):
        raise ValueError("SQL generator must return a string.")

    if not generated_sql.strip():
        raise ValueError("SQL generator returned an empty query.")

    # Remove optional Markdown SQL fences.
    generated_sql = re.sub(
        r"^```(?:sql)?\s*|\s*```$",
        "",
        generated_sql.strip(),
        flags=re.IGNORECASE
    )

    results = execute_readonly_query(generated_sql)

    return {
        "question": question,
        "sql": generated_sql,
        "results": results
    }
