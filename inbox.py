"""Fallback inbox: surfaces captures that need manual triage --
(a) unrouted notes (router matched no node type), and
(b) routed notes where some match came back below the confidence threshold.
Read-only, no LLM calls.
"""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from db import get_connection

CONFIDENCE_THRESHOLD = 0.6


def main():
    conn = get_connection()
    cur = conn.execute(
        "SELECT id, raw_text, created_at, node_types, router_confidence FROM notes ORDER BY created_at DESC"
    )

    unrouted = []
    low_confidence = []
    for note_id, raw_text, created_at, types_json, conf_json in cur.fetchall():
        node_types = json.loads(types_json)
        confidences = json.loads(conf_json)
        if not node_types:
            unrouted.append((note_id, created_at, raw_text))
        else:
            weak = {n: c for n, c in confidences.items() if c < CONFIDENCE_THRESHOLD}
            if weak:
                low_confidence.append((note_id, created_at, raw_text, weak))
    conn.close()

    print(f"\n===== INBOX (unrouted: {len(unrouted)}, low-confidence: {len(low_confidence)}) =====")

    print("\n-- Unrouted (matched no node) --")
    if not unrouted:
        print("  none")
    for note_id, created_at, raw_text in unrouted:
        print(f"  #{note_id} [{created_at[:16]}] {raw_text}")

    print(f"\n-- Low confidence (< {CONFIDENCE_THRESHOLD}) --")
    if not low_confidence:
        print("  none")
    for note_id, created_at, raw_text, weak in low_confidence:
        weak_str = ", ".join(f"{n}:{c:.2f}" for n, c in weak.items())
        print(f"  #{note_id} [{created_at[:16]}] {raw_text}  ({weak_str})")


if __name__ == "__main__":
    main()
