from typing import Literal, Optional

from pydantic import BaseModel, Field

NodeType = Literal["expense", "task", "reminder", "idea", "research", "book"]


class ExpenseFields(BaseModel):
    amount: float
    currency: str = Field(description="ISO currency code, default USD if unspecified")
    category: Optional[str] = None
    merchant: Optional[str] = None
    date: Optional[str] = Field(default=None, description="ISO date, infer from context or omit")


class TaskFields(BaseModel):
    task_text: str
    due_date: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high"]] = None


class ReminderFields(BaseModel):
    reminder_text: str
    due_at: Optional[str] = Field(default=None, description="ISO datetime if a time was implied")
    geofence_place: Optional[str] = Field(default=None, description="place name if location-triggered")


class IdeaFields(BaseModel):
    idea_text: str
    source: Literal["self", "article"] = "self"
    tags: Optional[list[str]] = None


class ResearchFields(BaseModel):
    source_url: Optional[str] = Field(default=None, description="omit if this is a plain note, not a link")
    summary: str
    key_points: Optional[list[str]] = None
    topic_tag: Optional[str] = None
    monitoring: bool = Field(default=False, description="true if the user asked to keep watching this topic")


class BookFields(BaseModel):
    title: str
    url: Optional[str] = None
    status: Literal["want", "reading", "done"] = "want"
    progress: Optional[str] = None
    notes: Optional[str] = None


class RouteMatch(BaseModel):
    node: NodeType
    confidence: float = Field(ge=0.0, le=1.0)
    expense_fields: Optional[ExpenseFields] = None
    task_fields: Optional[TaskFields] = None
    reminder_fields: Optional[ReminderFields] = None
    idea_fields: Optional[IdeaFields] = None
    research_fields: Optional[ResearchFields] = None
    book_fields: Optional[BookFields] = None


class RouteResult(BaseModel):
    matches: list[RouteMatch] = Field(
        default_factory=list,
        description="One entry per node type this capture belongs to. Empty if it fits none of the available nodes.",
    )
