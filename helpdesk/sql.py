import json
import sqlite3
import re
import time
from pathlib import Path


# Project and database paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _find_db_path():
    """Find the timetable database wherever the pack was unzipped."""
    for found in sorted((PROJECT_ROOT / "data").rglob("rail_timetable.db")):
        return found

    return PROJECT_ROOT / "database" / "rail_timetable.db"


DB_PATH = _find_db_path()

MAX_ROWS = 50
QUERY_TIMEOUT_SECONDS = 5

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
                # Interrupt queries that run for too long (a time limit, not a step count).
        query_start = time.time()

        def limit_query_execution():
            if time.time() - query_start > QUERY_TIMEOUT_SECONDS:
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
    Avoid unnecessary columns, but when listing trains
    always include the train number, name, departure
    time and arrival time.
    Do not add LIMIT; the system already limits rows.

12. Return only the SQL query.
    Do not include explanations, Markdown fences,
    or any additional text.

13. Resolve station names using the stations table.
    Never invent or assume station codes.

14. If the question contains a station code (for example
    'Jaipur (JP)' or 'LTT'), use that code directly.
    When the user gives only a station NAME, match it
    exact-name-first, and use LIKE only when no station
    has exactly that name:
        station_code IN (SELECT code FROM stations
      WHERE LOWER(name) = 'word'
         OR (LOWER(name) LIKE '%word%'
             AND NOT EXISTS (SELECT 1 FROM stations
                             WHERE LOWER(name) = 'word')))
    Use this pattern for EVERY station name, in every part
    of the query, including subqueries that look up a
    stop_sequence. Never match a station name with a plain
    = alone: station names often have extra words
    (for example 'CITY JN').
    Only list station codes and names when the user
    asks for them.

15. SPECIAL REPLIES. Instead of SQL, reply with exactly
    one word (this is the only exception to rule 12):
    - REFUSE if the user asks to change, add, delete or
      damage any data, or asks for anything unsafe.
    - NO_DATA if the question needs something this
      database does not contain: fares or prices, seat
      availability, waiting lists, PNR status, live
      running status, or refund and cancellation rules.
      Use NO_DATA only when NO part of the question can
      be answered from the timetable.
    If the question has several parts and one part is
    about trains, stations, times or routes, write SQL
    for that part only and ignore the other parts;
    another system answers them.
    Never fabricate an answer.

16. For train numbers, always compare using TEXT
    values and preserve leading zeros.

17. Use case-insensitive station-name matching
    when appropriate.

QUERY PATTERNS (examples with placeholder values, adapt them):

A. Trains that run from station AAA to station BBB.
   The two stations do NOT have to be the first and last
   station of the train, so never use trains.from_station_code
   and trains.to_station_code for this. Use the stops view:

   SELECT t.number, t.name, a.departure, b.arrival
   FROM stops a
   JOIN stops b ON b.train_number = a.train_number
        AND b.stop_sequence > a.stop_sequence
   JOIN trains t ON t.number = a.train_number
   WHERE a.station_code = 'AAA' AND b.station_code = 'BBB'
   ORDER BY a.departure

   When the question gives station names, replace
   a.station_code = 'AAA' with
   a.station_code IN (SELECT code FROM stations
   WHERE LOWER(name) LIKE '%aaa%') and do the same for b.

B. Journey time in minutes between two stops. Use the day
   column so overnight journeys are not negative:

   (b.day - a.day) * 1440
   + (CAST(SUBSTR(b.arrival, 1, 2) AS INTEGER) * 60
      + CAST(SUBSTR(b.arrival, 4, 2) AS INTEGER))
   - (CAST(SUBSTR(a.departure, 1, 2) AS INTEGER) * 60
      + CAST(SUBSTR(a.departure, 4, 2) AS INTEGER))

C. Count trains of one type. Look the code up in train_types:

   SELECT COUNT(*) FROM trains WHERE type_code = 'XXX'

D. Time a train leaves or reaches a station:

   SELECT departure FROM schedules
   WHERE train_number = 'NNNNN' AND station_code = 'AAA'

E. Find a station code from a name:

   SELECT code, name FROM stations
   WHERE LOWER(name) LIKE '%word%'

F. Stops between two stations on one train, and the longest
   halt (halt_min is in minutes; the stops view only holds
   real halts):

   SELECT st.name, s.arrival, s.departure, s.halt_min
   FROM stops a
   JOIN stops s ON s.train_number = a.train_number
        AND s.stop_sequence > a.stop_sequence
   JOIN stops b ON b.train_number = a.train_number
        AND b.stop_sequence > s.stop_sequence
   JOIN stations st ON st.code = s.station_code
   WHERE a.train_number = 'NNNNN'
     AND a.station_code = 'AAA' AND b.station_code = 'BBB'
   ORDER BY s.halt_min DESC

NOTE: the timetable is a 2016 snapshot, so newer trains
are not in it.    

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

# ---------------------------------------------------------------
# Part 2: the function that the pipeline calls
# ---------------------------------------------------------------

SQL_ATTEMPTS = 3  # first try + 2 corrections after an SQL error


def clean_generated_sql(text):
    """Remove optional Markdown fences and extra spaces."""

    return re.sub(
        r"^```(?:sql)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE
    ).strip()


def _result(answer, sql=None, dont_know=False):
    """Build the dictionary that every answer function returns."""

    return {
        "answer": answer,
        "sources": [],
        "sql": sql,
        "dont_know": dont_know
    }


def _default_llm(prompt):
    """Call the team's shared Gemini client (helpdesk/llm.py)."""

    from helpdesk.llm import generate

    return generate([{"role": "user", "content": prompt}])


def _rows_to_text(results):
    """Plain-text version of the rows, used if the LLM is unavailable."""

    lines = [", ".join(results["columns"])]

    for row in results["rows"]:
        lines.append(
            ", ".join("" if value is None else str(value) for value in row)
        )

    return "\n".join(lines)


def build_answer_prompt(question, sql, results):
    """Ask the LLM to explain the rows in plain words."""

    note = ""

    if results["truncated"] or results["row_count"] >= MAX_ROWS:
        note = (
            f"The result was cut at {MAX_ROWS} rows, "
            "so say that more rows exist.\n"
        )

    return (
        "You are a railway help desk assistant. Answer the "
        "passenger's question in plain words, using ONLY the "
        "query result below.\n"
        "Rules:\n"
        "- Do not add any train, station, time or number that "
        "is not in the result.\n"
        "- Keep train numbers exactly as written, including "
        "leading zeros.\n"
        "- Times are 24-hour HH:MM.\n"
        "- If the question also asks about something the result "
        "does not cover (for example fares, charges or rules), "
        "do not answer that part and do not guess; answer only "
        "what the result shows.\n"
        "- End with one short sentence saying the answer is "
        "from the 2016 timetable in the data pack.\n"
        + note
        + f"\nQuestion: {question}\n"
        + f"SQL that was run: {sql}\n"
        + f"Columns: {results['columns']}\n"
        + f"Rows: {results['rows']}\n"
    )


def answer_from_sql(question, llm=None):
    """
    Answer a timetable question and return the standard dictionary:
    {"answer", "sources", "sql", "dont_know"}.

    llm is optional: a function that takes a prompt string and
    returns text. By default the shared helpdesk/llm.py is used.
    """

    llm = llm or _default_llm

    if not question or not question.strip():
        return _result("Please ask a question.", dont_know=True)

    try:
        schema = get_database_schema()
    except Exception as error:
        return _result(
            f"The timetable database could not be opened: {error}",
            dont_know=True
        )

    sql = None
    error_text = None
    extra_note = ""
    zero_rows_retried = False
    results = None

    for attempt in range(SQL_ATTEMPTS):
        prompt = build_sql_prompt(question, schema)

        if error_text:
            prompt += (
                f"\nYour previous query was:\n{sql}\n"
                f"It failed with this error:\n{error_text}\n"
                "Write a corrected SQL query. Return only the SQL.\n"
            )

        prompt += extra_note

        try:
            raw_reply = llm(prompt)
        except Exception as error:
            return _result(
                "I could not reach the language model right now. "
                "Please try again in a minute.",
                sql=sql,
                dont_know=True
            )

        sql = clean_generated_sql(raw_reply or "")
        signal = sql.upper().strip(" .\n")

        if signal.startswith("REFUSE"):
            return _result(
                "I can't do that. I only run read-only queries "
                "on the timetable.",
                dont_know=True
            )

        if signal.startswith("NO_DATA"):
            if attempt == 0:
                # Second look: a mixed question may have a timetable part.
                extra_note = (
                    "\nA reviewer says: your first reply was NO_DATA, "
                    "but the question may have a part that asks for a "
                    "train, a time, a station or a route. If it does, "
                    "write SQL for that part only and ignore the other "
                    "parts. Reply NO_DATA only if no part of the "
                    "question can be answered from the timetable.\n"
                )
                continue

            return _result(
                "I don't know. The timetable in my data does not "
                "contain that information.",
                dont_know=True
            )

        if not sql:
            error_text = "The reply was empty."
            continue

        try:
            results = execute_readonly_query(sql)
        except ValueError as error:
            error_text = str(error)
            continue

        if results["row_count"] == 0 and not zero_rows_retried:
            # An empty result is often a wrong station name or condition:
            # give the model one more try before saying nothing was found.
            zero_rows_retried = True
            error_text = (
                "The query ran but returned 0 rows. Check every station "
                "name (use the exact-first LIKE pattern from rule 14) and "
                "every condition, then write a corrected query. If the "
                "data really does not exist, repeat the same query."
            )
            continue

        break

    if results is None:
        return _result(
            "I could not turn this question into a valid "
            "timetable query.",
            sql=sql,
            dont_know=True
        )

    if results["row_count"] == 0:
        return _result(
            "I could not find any matching records in the "
            "2016 timetable.",
            sql=sql,
            dont_know=True
        )

    try:
        answer = llm(build_answer_prompt(question, sql, results)).strip()

        if not answer:
            raise ValueError("empty answer")
    except Exception:
        answer = (
            "According to the 2016 timetable:\n"
            + _rows_to_text(results)
        )

    return _result(answer, sql=sql)