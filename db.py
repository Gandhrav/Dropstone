import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

DB_PATH = Path(__file__).parent / "poc.db"

# kNN auto-link threshold: cosine distance (0 = identical, 1 = unrelated).
# bge-small puts same-topic notes around 0.2-0.35; tune after real usage.
AUTO_LINK_MAX_DISTANCE = 0.35
AUTO_LINK_K = 5

# Versioned migrations, tracked via PRAGMA user_version. Index 0 = v1.
# Rules: never edit or reorder an entry once it has run against a real db --
# append a new one instead. Existing dbs (created pre-migrations, user_version=0)
# pass through v1's IF NOT EXISTS harmlessly and get stamped v1.
MIGRATIONS = [
    # v1 -- Phase 1 baseline: generic notes + 6 structured node tables
    """
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
""",
    # v2 -- Phase 2 knowledge-graph substrate: edges + embedding metadata.
    # The vec0 virtual table (notes_vec) is NOT created here: its dimension
    # depends on the runtime embedding provider (local=384, voyage=1024), and
    # migrations are static SQL. ensure_vec_table() creates it lazily and
    # vec_meta records which model/dim the vectors belong to, so a provider
    # switch is detected instead of silently mixing incomparable vectors.
    """
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_note_id INTEGER NOT NULL REFERENCES notes(id),
    target_note_id INTEGER NOT NULL REFERENCES notes(id),
    edge_type TEXT NOT NULL DEFAULT 'related',
    weight REAL,
    provenance TEXT NOT NULL CHECK (provenance IN ('explicit', 'inferred')),
    created_at TEXT NOT NULL,
    UNIQUE (source_note_id, target_note_id, edge_type)
);

CREATE TABLE IF NOT EXISTS vec_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL
);
""",
]


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for target, script in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {target}")
        conn.commit()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _migrate(conn)
    return conn


# ---------- Phase 2: embeddings + edges ----------


def ensure_vec_table(conn: sqlite3.Connection, model_id: str, dim: int) -> None:
    """Create the vector table for the active embedding model, or refuse if
    the db already holds vectors from a different model (dims/values are not
    comparable across models -- see embeddings.py docstring)."""
    row = conn.execute("SELECT model_id, dim FROM vec_meta WHERE id = 1").fetchone()
    if row is None:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0(embedding float[{dim}] distance_metric=cosine)"
        )
        conn.execute("INSERT INTO vec_meta (id, model_id, dim) VALUES (1, ?, ?)", (model_id, dim))
        conn.commit()
    elif row[0] != model_id or row[1] != dim:
        raise RuntimeError(
            f"poc.db vectors were built with {row[0]!r} (dim={row[1]}), but the active "
            f"embedding model is {model_id!r} (dim={dim}). Re-embed all notes into a fresh "
            "vector table before switching providers (drop notes_vec + vec_meta, run backfill.py)."
        )


def insert_embedding(conn: sqlite3.Connection, note_id: int, vector: list[float]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO notes_vec (rowid, embedding) VALUES (?, ?)",
        (note_id, sqlite_vec.serialize_float32(vector)),
    )
    conn.commit()


def knn_similar(conn: sqlite3.Connection, vector: list[float], exclude_note_id: int, k: int = AUTO_LINK_K) -> list[tuple[int, float]]:
    """Nearest existing notes by cosine distance, excluding the note itself.
    Returns [(note_id, distance)] under AUTO_LINK_MAX_DISTANCE."""
    rows = conn.execute(
        "SELECT rowid, distance FROM notes_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (sqlite_vec.serialize_float32(vector), k + 1),
    ).fetchall()
    return [
        (note_id, dist)
        for note_id, dist in rows
        if note_id != exclude_note_id and dist <= AUTO_LINK_MAX_DISTANCE
    ][:k]


def insert_edge(
    conn: sqlite3.Connection,
    source_note_id: int,
    target_note_id: int,
    provenance: str,
    edge_type: str = "related",
    weight: float | None = None,
) -> None:
    """Undirected edge, stored with source < target so A-B and B-A dedupe."""
    a, b = sorted((source_note_id, target_note_id))
    conn.execute(
        "INSERT OR IGNORE INTO edges (source_note_id, target_note_id, edge_type, weight, provenance, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (a, b, edge_type, weight, provenance, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def get_related_notes(conn: sqlite3.Connection, note_id: int, hops: int = 1) -> list[dict]:
    """Notes connected to note_id within N hops (recursive CTE over edges)."""
    rows = conn.execute(
        """
        WITH RECURSIVE walk(note_id, depth) AS (
            SELECT ?, 0
            UNION
            SELECT CASE WHEN e.source_note_id = w.note_id THEN e.target_note_id ELSE e.source_note_id END,
                   w.depth + 1
            FROM edges e
            JOIN walk w ON w.note_id IN (e.source_note_id, e.target_note_id)
            WHERE w.depth < ?
        ),
        nearest(note_id, depth) AS (
            SELECT note_id, MIN(depth) FROM walk GROUP BY note_id
        )
        SELECT w.note_id, w.depth, n.raw_text, n.node_types,
               e.weight, e.provenance, e.edge_type
        FROM nearest w
        JOIN notes n ON n.id = w.note_id
        LEFT JOIN edges e ON (e.source_note_id = MIN(w.note_id, ?) AND e.target_note_id = MAX(w.note_id, ?))
        WHERE w.note_id != ?
        ORDER BY w.depth, e.weight DESC
        """,
        (note_id, hops, note_id, note_id, note_id),
    ).fetchall()
    return [
        {
            "note_id": r[0],
            "depth": r[1],
            "raw_text": r[2],
            "node_types": json.loads(r[3]),
            "weight": r[4],
            "provenance": r[5],
            "edge_type": r[6],
        }
        for r in rows
    ]


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
