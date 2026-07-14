import argparse
import json

from dotenv import load_dotenv

load_dotenv()

from graph import build_graph  # noqa: E402  (must load .env before ChatAnthropic is constructed)


def main():
    parser = argparse.ArgumentParser(description="Capture a raw note and route it through the node graph.")
    parser.add_argument("text", help="Raw captured text")
    args = parser.parse_args()

    app = build_graph()
    result = app.invoke({"raw_text": args.text, "matches": [], "stored": []})

    print(f"\nraw:    {args.text}")
    print(f"note_id: {result.get('note_id')}")
    if not result["matches"]:
        print("routed: (no node matched -- inbox only)")
        return

    print("routed:")
    for m in result["matches"]:
        print(f"  - {m['node']} (confidence={m['confidence']:.2f})")

    print("stored:")
    for s in result.get("stored", []):
        print(f"  - {s['node']} row #{s['row_id']}: {json.dumps(s['fields'])}")


if __name__ == "__main__":
    main()
