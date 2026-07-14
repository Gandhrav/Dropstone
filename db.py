import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "poc.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    node_types TEXT NOT NULL,       -- JSON array, e.g. ["expense","task"]
    router_confidence TEXT NOT NULL -- JSON map, e.g. {"expense":0.92}
);

CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL REFERENCES notes(id),
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    category TEXT,
    merchant TEXT,
    date TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL REFERENCES notes(id),
    task_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    due_date TEXT,
    priority TEXT
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL REFERENCES notes(id),
    reminder_text TEXT NOT NULL,
    due_at TEXT,
    geofence_place TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL REFERENCES notes(id),
    idea_text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'self',
    tags TEXT -- JSON array
);

CREATE TABLE IF NOT EXISTS research (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL REFERENCES notes(id),
    source_url TEXT,
    summary TEXT NOT NULL,
    key_points TEXT, -- JSON array
    topic_tag TEXT,
    monitoring INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id INTEGER NOT NULL REFERENCES notes(id),
    title TEXT NOT NULL,
    url TEXT,
    status TEXT NOT NULL DEFAULT 'want',
    progress TEXT,
    notes TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def insert_note(conn: sqlite3.Connection, raw_text: str, node_types: list[str], confidences: dict[str, float]) -> int:
    cur = conn.execute(
        "INSERT INTO notes (raw_text, created_at, node_types, router_confidence) VALUES (?, ?, ?, ?)",
        (raw_text, datetime.now(timezone.utc).isoformat(), json.dumps(node_types), json.dumps(confidences)),
    )
    conn.commit()
    return cur.lastrowid


def insert_expense(conn: sqlite3.Connection, note_id: int, fields: dict) -> int:
    cur = conn.execute(
        "INSERT INTO expenses (note_id, amount, currency, category, merchant, date) VALUES (?, ?, ?, ?, ?, ?)",
        (note_id, fields["amount"], fields["currency"], fields.get("category"), fields.get("merchant"), fields.get("date")),
    )
    conn.commit()
    return cur.lastrowid


def insert_task(conn: sqlite3.Connection, note_id: int, fields: dict) -> int:
    cur = conn.execute(
        "INSERT INTO tasks (note_id, task_text, due_date, priority) VALUES (?, ?, ?, ?)",
        (note_id, fields["task_text"], fields.get("due_date"), fields.get("priority")),
    )
    conn.commit()
    return cur.lastrowid


def insert_reminder(conn: sqlite3.Connection, note_id: int, fields: dict) -> int:
    cur = conn.execute(
        "INSERT INTO reminders (note_id, reminder_text, due_at, geofence_place) VALUES (?, ?, ?, ?)",
        (note_id, fields["reminder_text"], fields.get("due_at"), fields.get("geofence_place")),
    )
    conn.commit()
    return cur.lastrowid


def insert_idea(conn: sqlite3.Connection, note_id: int, fields: dict) -> int:
    cur = conn.execute(
        "INSERT INTO ideas (note_id, idea_text, source, tags) VALUES (?, ?, ?, ?)",
        (note_id, fields["idea_text"], fields.get("source", "self"), json.dumps(fields.get("tags"))),
    )
    conn.commit()
    return cur.lastrowid


def insert_research(conn: sqlite3.Connection, note_id: int, fields: dict) -> int:
    cur = conn.execute(
        "INSERT INTO research (note_id, source_url, summary, key_points, topic_tag, monitoring) VALUES (?, ?, ?, ?, ?, ?)",
        (
            note_id,
            fields.get("source_url"),
            fields["summary"],
            json.dumps(fields.get("key_points")),
            fields.get("topic_tag"),
            1 if fields.get("monitoring") else 0,
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_book(conn: sqlite3.Connection, note_id: int, fields: dict) -> int:
    cur = conn.execute(
        "INSERT INTO books (note_id, title, url, status, progress, notes) VALUES (?, ?, ?, ?, ?, ?)",
        (
            note_id,
            fields["title"],
            fields.get("url"),
            fields.get("status", "want"),
            fields.get("progress"),
            fields.get("notes"),
        ),
    )
    conn.commit()
    return cur.lastrowid
