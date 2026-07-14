import operator
from typing import Annotated, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from db import (
    get_connection,
    insert_book,
    insert_expense,
    insert_idea,
    insert_note,
    insert_reminder,
    insert_research,
    insert_task,
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

_model = ChatAnthropic(model="claude-sonnet-5")
_router = _model.with_structured_output(RouteResult)


class CaptureState(TypedDict):
    raw_text: str
    matches: list[dict]
    note_id: int
    match: Optional[dict]
    stored: Annotated[list[dict], operator.add]


def route_node(state: CaptureState) -> dict:
    result: RouteResult = _router.invoke(
        [("system", SYSTEM_PROMPT), ("user", state["raw_text"])]
    )
    return {"matches": [m.model_dump() for m in result.matches]}


def save_note_node(state: CaptureState) -> dict:
    conn = get_connection()
    node_types = [m["node"] for m in state["matches"]]
    confidences = {m["node"]: m["confidence"] for m in state["matches"]}
    note_id = insert_note(conn, state["raw_text"], node_types, confidences)
    conn.close()
    return {"note_id": note_id}


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
    for category in CATEGORY_NODES:
        builder.add_node(category, category_store_node)
        builder.add_edge(category, END)

    builder.add_edge(START, "route")
    builder.add_edge("route", "save_note")
    builder.add_conditional_edges("save_note", dispatch_to_nodes)

    return builder.compile()


graph = build_graph()
