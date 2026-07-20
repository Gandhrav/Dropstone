import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from llm import get_chat_model

from db import (
    ensure_vec_table,
    get_connection,
    insert_book,
    insert_edge,
    insert_embedding,
    insert_expense,
    insert_idea,
    insert_note,
    insert_reminder,
    insert_research,
    insert_task,
    knn_similar,
)
from schemas import RouteResult

NODE_CATALOG = """
- expense: money spent or received. Typed amount ("40 groceries") or a described purchase.
- task: a general to-do/checklist item with no fixed time (e.g. "buy milk", "fix bike").
- reminder: something to be nudged about at a specific time or place (e.g. "call mom tomorrow", "pick up dry cleaning near the mall").
- idea: a raw idea or thought the user wants to keep, not from an external source (e.g. "app idea: AI recipe generator from fridge photos").
- research: a pasted article/blog link, or a request to look into / keep watching a topic.
- book: a book mention or a saved article/link to read later, with a want/reading/done status.
"""

SYSTEM_PROMPT = f"""You are the router for a personal capture app. Available node types:
{NODE_CATALOG}
Decide which node(s) this raw capture belongs to. A single capture can match multiple nodes \
(e.g. it's both a task and a time-based reminder). If it matches none of the available nodes, \
return an empty matches list. Only fill the *_fields object matching the node type you assigned \
to that match."""

INSERT_FNS = {
    "expense": insert_expense,
    "task": insert_task,
    "reminder": insert_reminder,
    "idea": insert_idea,
    "research": insert_research,
    "book": insert_book,
}

FIELDS_KEY = {
    "expense": "expense_fields",
    "task": "task_fields",
    "reminder": "reminder_fields",
    "idea": "idea_fields",
    "research": "research_fields",
    "book": "book_fields",
}

# Groups the 6 implemented node types (and future ones, commented) into 4 LangGraph
# dispatcher nodes -- shared confirm-policy/storage-shape families, not just topic.
NODE_CATEGORY = {
    "expense": "tracker",  # + future: health, journal, decision
    "task": "task_action",  # + future: commitment, calendar
    "reminder": "task_action",
    "idea": "knowledge",  # + future: wishlist, anki
    "research": "knowledge",
    "book": "knowledge",
    # "crm": "relationship",  -- future, own dispatcher (person-centric, cross-links everywhere)
}

CATEGORY_NODES = ["task_action", "tracker", "knowledge", "relationship"]

_model = get_chat_model()
_router = _model.with_structured_output(RouteResult)


class CaptureState(TypedDict):
    raw_text: str
    matches: list[dict]
    note_id: int
    match: Optional[dict]
    stored: Annotated[list[dict], operator.add]
    linked: list[dict]


_embedder = None


def _get_embedder():
    """Lazy singleton -- fastembed loads a ~130MB ONNX model on first use;
    eval.py imports route_node without ever needing it."""
    global _embedder
    if _embedder is None:
        from embeddings import get_embedder

        _embedder = get_embedder()
    return _embedder


def route_node(state: CaptureState) -> dict:
    # Models occasionally double-encode the tool args (matches arrives as a
    # JSON string, pydantic rejects it) -- retry once before failing. More
    # relevant with BYOM: smaller models flake at this boundary more often.
    last_err = None
    for _ in range(2):
        try:
            result: RouteResult = _router.invoke(
                [("system", SYSTEM_PROMPT), ("user", state["raw_text"])]
            )
            return {"matches": [m.model_dump() for m in result.matches]}
        except Exception as err:  # pydantic ValidationError wrapped by langchain
            last_err = err
    raise last_err


def save_note_node(state: CaptureState) -> dict:
    conn = get_connection()
    node_types = [m["node"] for m in state["matches"]]
    confidences = {m["node"]: m["confidence"] for m in state["matches"]}
    note_id = insert_note(conn, state["raw_text"], node_types, confidences)
    conn.close()
    return {"note_id": note_id}


def auto_link_node(state: CaptureState) -> dict:
    """Phase 2: embed the note, store the vector, kNN-link to similar past
    notes. Edges carry provenance='inferred' (embedding similarity) as opposed
    to 'explicit' (router/structural knowledge)."""
    embedder = _get_embedder()
    vector = embedder.embed([state["raw_text"]])[0]

    conn = get_connection()
    ensure_vec_table(conn, embedder.model_id, embedder.dim)

    neighbors = knn_similar(conn, vector, exclude_note_id=state["note_id"])
    insert_embedding(conn, state["note_id"], vector)

    linked = []
    for other_id, distance in neighbors:
        similarity = round(1.0 - distance, 3)
        insert_edge(conn, state["note_id"], other_id, provenance="inferred", weight=similarity)
        raw = conn.execute("SELECT raw_text FROM notes WHERE id = ?", (other_id,)).fetchone()
        linked.append({"note_id": other_id, "similarity": similarity, "raw_text": raw[0] if raw else "?"})
    conn.close()

    return {"linked": linked}


def dispatch_to_nodes(state: CaptureState):
    if not state["matches"]:
        return END
    return [
        Send(NODE_CATEGORY[m["node"]], {"match": m, "note_id": state["note_id"]})
        for m in state["matches"]
    ]


def category_store_node(state: CaptureState) -> dict:
    """Shared body for all 4 category dispatchers -- registered under 4 node
    names so Studio shows the category split, without duplicating this logic."""
    match = state["match"]
    note_id = state["note_id"]
    node = match["node"]
    fields = match[FIELDS_KEY[node]]

    conn = get_connection()
    row_id = INSERT_FNS[node](conn, note_id, fields)
    conn.close()

    return {"stored": [{"node": node, "row_id": row_id, "fields": fields}]}


def build_graph():
    builder = StateGraph(CaptureState)
    builder.add_node("route", route_node)
    builder.add_node("save_note", save_note_node)
    builder.add_node("auto_link", auto_link_node)
    for category in CATEGORY_NODES:
        builder.add_node(category, category_store_node)
        builder.add_edge(category, END)

    builder.add_edge(START, "route")
    builder.add_edge("route", "save_note")
    builder.add_edge("save_note", "auto_link")
    builder.add_conditional_edges("auto_link", dispatch_to_nodes)

    return builder.compile()


graph = build_graph()
