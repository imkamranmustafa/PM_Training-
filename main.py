# main.py
from coordinator import Coordinator
import argparse
import json

def run_cli():
    parser = argparse.ArgumentParser(description="Simple Multi-Agent Chat System (CLI)")
    parser.add_argument("--query", "-q", type=str, help="User query to process")
    args = parser.parse_args()
    coord = Coordinator()
    if args.query:
        out = coord.handle(args.query)
        print("--- FINAL ANSWER ---")
        print(out["answer"])
        print("--- METADATA ---")
        print(json.dumps({"confidence": out.get("confidence"), "task_id": out.get("task_id")}, indent=2))
    else:
        print("Start interactive mode (type 'exit' to quit)")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            out = coord.handle(q)
            print(out["answer"])
            print(f"(confidence: {out.get('confidence')})\n")

if __name__ == "__main__":
    run_cli()
