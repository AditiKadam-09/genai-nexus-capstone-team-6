from helpdesk.sql import get_database_schema, build_sql_prompt

question = "Find trains from Kalyan to Bhopal"

schema = get_database_schema()

prompt = build_sql_prompt(question, schema)

print("\n--- GENERATED TEXT-TO-SQL PROMPT ---")
print(prompt)

print("\n--- PROMPT VALIDATION ---")

checks = [
    "USER QUESTION:",
    "DATABASE SCHEMA:",
    "stop_sequence",
    "journey day offset",
    "read-only SELECT",
    question
]

for check in checks:
    if check.lower() in prompt.lower():
        print(f"PASSED: {check}")
    else:
        print(f"FAILED: {check}")