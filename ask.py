"""Ask the Railway Passenger Help Desk.

  python ask.py "your question"          one question, readable answer
  python ask.py --json "your question"   one question, raw JSON
  python ask.py                          interactive mode: type a question and
                                         press Enter (type exit to quit)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from helpdesk.pipeline import ask


def format_result(result):
    """Readable version of the 4-key result."""
    lines = ["", "ANSWER:", result["answer"].strip()]

    if result["dont_know"]:
        lines += ["", "(dont_know: true - the system could not answer this from its data)"]

    if result["sources"]:
        lines += ["", "SOURCES:"]
        for source in result["sources"]:
            page = "web page" if source["page"] is None else f"page {source['page']}"
            lines.append(f"  - {source['doc']} ({page})")

    if result["sql"]:
        lines += ["", "SQL:", result["sql"].strip()]

    lines.append("")
    return "\n".join(lines)


def warm_up():
    """Load the search models once, so the first question is not slow."""
    try:
        from helpdesk.search import search
        search("warm up", k=1)
    except Exception:
        pass


def interactive():
    print("Railway Passenger Help Desk")
    print("Loading the search models, please wait...")
    warm_up()
    print('Ready. Type a question and press Enter. Type "exit" to quit.\n')

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break

        print(format_result(ask(question)))


def main():
    args = sys.argv[1:]
    as_json = "--json" in args
    words = [a for a in args if a != "--json"]

    if not words:
        interactive()
        return

    result = ask(" ".join(words))
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_result(result))


if __name__ == "__main__":
    main()