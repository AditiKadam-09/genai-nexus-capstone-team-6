"""Run many questions: python run_questions.py <questions.json> <results.json>

The questions file is a JSON list of {"id": ..., "question": ...}.
The results file is a JSON list with one object per question: id, question
and everything ask() returned (answer, sources, sql, dont_know).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from helpdesk.pipeline import ask

PAUSE_SECONDS = 1


def load_questions(path):
    text = Path(path).read_text(encoding="utf-8-sig").strip()

    if Path(path).suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("questions", [])
        questions = []
        for position, item in enumerate(data, start=1):
            if isinstance(item, str):
                questions.append({"id": f"Q{position}", "question": item})
            else:
                questions.append({
                    "id": item.get("id", f"Q{position}"),
                    "question": item["question"],
                })
        return questions

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [{"id": f"Q{i}", "question": line} for i, line in enumerate(lines, start=1)]


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_questions.py <questions> <results>")
        sys.exit(1)

    questions = load_questions(sys.argv[1])
    results = []

    for item in questions:
        print(f"[{item['id']}] {item['question']}", flush=True)
        try:
            answer = ask(item["question"])
        except Exception as error:  # ask() should never raise, but be safe
            answer = {
                "answer": f"Sorry, something went wrong: {type(error).__name__}",
                "sources": [],
                "sql": None,
                "dont_know": True,
            }
        results.append({"id": item["id"], "question": item["question"], **answer})

        # Save after every question so partial results survive a crash.
        Path(sys.argv[2]).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        time.sleep(PAUSE_SECONDS)

    print(f"Done: {len(results)} results written to {sys.argv[2]}")


if __name__ == "__main__":
    main()