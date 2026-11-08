#!/usr/bin/env python3
"""ask.py: CLI interface for the Indian Railways Passenger Help Desk."""

import sys
import json
import argparse
from helpdesk.pipeline import ask


def main():
    parser = argparse.ArgumentParser(
        description="Query the Indian Railways Passenger Help Desk."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="Passenger question string to be answered.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON matching the required contract.",
    )

    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        sys.exit(1)

    result = ask(args.question)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result.get("answer", ""))
    print("=" * 60)

    sources = result.get("sources", [])
    if sources:
        print("\nSOURCES:")
        for s in sources:
            page_info = f", Page {s['page']}" if s.get("page") is not None else ""
            print(f" - {s.get('doc')}{page_info}")

    sql = result.get("sql")
    if sql:
        print("\nSQL EXECUTED:")
        print(f" {sql}")

    if result.get("dont_know"):
        print("\n[Status: Abstention (dont_know=True)]")
    print()


if __name__ == "__main__":
    main()
