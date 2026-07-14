import json
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic  # noqa: E402  (must load .env first)

from db import get_connection  # noqa: E402

TABLE_FOR_NODE = {
    "expense": "expenses",
    "task": "tasks",
    "reminder": "reminders",
    "idea": "ideas",
    "research": "research",
    "book": "books",
}


def fetch_todays_captures(conn) -> list[dict]:
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    cur = conn.execute(
        "SELECT id, raw_text, created_at, node_types FROM notes WHERE created_at >= ? ORDER BY created_at",
        (today_start,),
    )
    notes = cur.fetchall()

    captures = []
    for note_id, raw_text, created_at, node_types_json in notes:
        for node_type in json.loads(node_types_json):
            table = TABLE_FOR_NODE.get(node_type)
            if not table:
                continue
            row_cur = conn.execute(f"SELECT * FROM {table} WHERE note_id = ?", (note_id,))
            col_names = [d[0] for d in row_cur.description]
            for row in row_cur.fetchall():
                captures.append(
                    {
                        "node": node_type,
                        "raw_text": raw_text,
                        "created_at": created_at,
                        "fields": dict(zip(col_names, row)),
                    }
                )
    return captures


def build_report(captures: list[dict]) -> str:
    if not captures:
        return "No captures today."

    lines = [f"- [{c['node']}] {c['raw_text']} -> {c['fields']}" for c in captures]
    prompt = (
        "Here is everything captured today in a personal capture app, across its nodes "
        "(expenses, tasks, reminders, ideas, research, books):\n\n"
        + "\n".join(lines)
        + "\n\nWrite a short end-of-day summary report: group by category, call out anything "
        "time-sensitive (unresolved reminders/tasks), and note any interesting connections "
        "between captures."
    )

    model = ChatAnthropic(model="claude-sonnet-5")
    return model.invoke(prompt).content


def main():
    conn = get_connection()
    captures = fetch_todays_captures(conn)
    conn.close()

    report = build_report(captures)
    print("\n===== END OF DAY DIGEST =====\n")
    print(report)


if __name__ == "__main__":
    main()
