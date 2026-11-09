#!/usr/bin/env python3
"""run_questions.py: Batch evaluation runner for Indian Railways Passenger Help Desk."""

import sys
import json
import time
import argparse
from pathlib import Path
from helpdesk.pipeline import ask


def main():
    parser = argparse.ArgumentParser(
        description="Run batch evaluation over a JSON questions dataset."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSON file containing questions.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output JSON file to save generated answers.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: Input JSON must be a list of question objects.")
        sys.exit(1)

    print(f"Loaded {len(data)} questions from {input_path}.")
    results = []

    for idx, item in enumerate(data, start=1):
        q_id = item.get("id", f"Q{idx}")
        question = item.get("question", "")
        print(f"[{idx}/{len(data)}] Answering {q_id}: {question[:60]}...")

        start_t = time.time()
        try:
            res = ask(question)
        except Exception as exc:
            print(f"  Error processing {q_id}: {exc}")
            res = {
                "answer": f"Error: {exc}",
                "sources": [],
                "sql": None,
                "dont_know": True,
            }
        elapsed = time.time() - start_t
        print(f"  Done in {elapsed:.2f}s (dont_know: {res.get('dont_know')})")

        out_entry = {
            "id": q_id,
            "question": question,
            "answer": res.get("answer", ""),
            "sources": res.get("sources", []),
            "sql": res.get("sql"),
            "dont_know": res.get("dont_know", False),
        }
        results.append(out_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {output_path}.")


if __name__ == "__main__":
    main()
