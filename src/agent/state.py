from typing import Any, TypedDict


class ConversationState(TypedDict, total=False):
    messages: list[Any]
    transcript: str
    response_text: str
    intent: str
    source_ids: list[str]
    retrieved_context: list[dict]
    order_id: str | None
    order_verified: bool
    verification_attempts: int
    verified_order: dict | None
    product_data: list[dict]
    tool_calls: list[dict]
    decision: str
    ticket_created: bool
    error: str | None
    history: list[str]
    action: str
    response_repaired: bool
    computed_facts: list[str]
