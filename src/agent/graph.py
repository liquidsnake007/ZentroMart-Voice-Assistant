from pathlib import Path
import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from groq import RateLimitError

from src.agent.state import ConversationState
from src.config import settings
from src.rag.retriever import RerankingRetriever
from src.tools.tools import create_ticket, lookup_order
from src.tools.catalog import lookup_product, product_skus, search_products
from src.agent.policy import Action, choose_action, extract_signals
from src.agent.response import needs_repair
from src.agent.facts import order_facts, refund_facts
from src.agent.composer import compose_deterministic_answer


ORDER_ID_RE = re.compile(r"\b\d(?:[\s-]?\d){7}\b")
SKU_RE = re.compile(r"\b([A-Z]{3}-\d{4})\b", re.IGNORECASE)

SYSTEM_PROMPT = """You are ZentroMart's customer support voice assistant.
Use only the supplied evidence and verified order data. Current policy documents
override help, FAQ, marketing, and superseded documents. If evidence does not
answer the question, say you do not have enough information and offer human help.
Never reveal order details until the order tool has verified the caller. Keep
answers short and natural for speech. Do not use markdown tables.
"""


def _format_context(items: list[dict]) -> str:
    return "\n\n".join(
        f"[{item['section_id']}] {item['text'][:1200]}" for item in items[:4]
    )


def _history_text(state: ConversationState) -> str:
    history = state.get("history", [])
    return "\n".join(history[-4:]) if history else "(first turn)"


def _order_id(text: str) -> str | None:
    match = ORDER_ID_RE.search(text)
    return "".join(ch for ch in match.group(0) if ch.isdigit()) if match else None


def _verification_value(text: str, order_id: str | None) -> str | None:
    for token in re.findall(r"\b\d+\b", text):
        if token != order_id:
            if len(token) in (4, 6):
                return token
    return None


def _product_skus(text: str) -> list[str]:
    return [match.upper() for match in SKU_RE.findall(text) if match.upper() in product_skus()]


def build_graph(retriever: RerankingRetriever, llm=None):
    llm = llm or ChatGroq(
        model=settings.groq_model,
        temperature=0,
        max_tokens=300,
    )

    def retrieve(state: ConversationState) -> ConversationState:
        context = retriever.retrieve(state["transcript"])
        order_id = state.get("order_id") or _order_id(state["transcript"])
        verified_order = state.get("verified_order")
        order_verified = bool(state.get("order_verified"))
        verification_attempts = int(state.get("verification_attempts", 0))
        tool_calls = list(state.get("tool_calls", []))
        product_data = list(state.get("product_data", []))
        for sku in _product_skus(state["transcript"]):
            product = lookup_product(sku)
            if product:
                product_data.append(product)
                tool_calls.append({"name": "lookup_product", "sku": sku, "found": True})
        for product in search_products(state["transcript"]):
            if product["sku"] not in {item["sku"] for item in product_data}:
                product_data.append(product)
                tool_calls.append(
                    {"name": "lookup_product", "sku": product["sku"], "found": True}
                )
        if order_id and not order_verified:
            verification_value = _verification_value(state["transcript"], order_id)
            result = lookup_order(order_id, verification_value)
            tool_calls.append(
                {
                    "name": "lookup_order",
                    "order_id": order_id,
                    "verified": bool(result.get("verified")),
                    "verification_required": bool(result.get("verification_required")),
                }
            )
            if result.get("verified"):
                verified_order = result
                order_verified = True
            elif verification_value is not None:
                verification_attempts += 1
        computed_facts = [
            *refund_facts(state["transcript"]),
            *order_facts(verified_order, state["transcript"]),
        ]
        return {
            **state,
            "retrieved_context": context,
            "source_ids": [item["section_id"] for item in context],
            "order_id": order_id,
            "order_verified": order_verified,
            "verified_order": verified_order,
            "tool_calls": tool_calls,
            "verification_attempts": verification_attempts,
            "product_data": product_data,
            "history": [*state.get("history", []), state["transcript"]],
            "computed_facts": computed_facts,
        }

    def route(state: ConversationState) -> ConversationState:
        transcript = state["transcript"]
        order_id = state.get("order_id")
        action = choose_action(
            extract_signals(transcript, order_id),
            order_verified=bool(state.get("order_verified")),
            verification_attempts=int(state.get("verification_attempts", 0)),
        )
        return {**state, "action": action.value}

    def answer(state: ConversationState) -> ConversationState:
        transcript = state["transcript"]
        action = Action(state["action"])
        if action is Action.REFUSE and not state.get("order_id"):
            return {
                **state,
                "response_text": (
                    "I can't share another person's private contact "
                    "information. I can help with your own order or connect "
                    "you with a colleague."
                ),
                "decision": "refuse",
            }
        if action is Action.CLARIFY and not state.get("order_id"):
            if extract_signals(transcript, None).generic_order_request:
                response = (
                    "I’m an AI assistant. Please share your order number so I "
                    "can help."
                )
            else:
                response = (
                    "Could you share the product, order number, or refund "
                    "details so I can give you the correct timeline?"
                )
            return {
                **state,
                "response_text": response,
                "decision": "clarify",
            }
        if action is Action.VERIFY_ORDER:
            return {
                **state,
                "response_text": (
                    "I can help with that order. Please provide either the "
                    "last four digits of your registered phone number or the "
                    "six-digit delivery pincode."
                ),
                "decision": "clarify",
            }
        if action is Action.REFUSE and state.get("order_id"):
            if state.get("verification_attempts", 0) >= 2:
                return {
                    **state,
                    "response_text": (
                        "I couldn't verify those details, so I can't share "
                        "order information. Please contact a human colleague "
                        "through support for further help."
                    ),
                    "decision": "refuse",
                }
        if action is Action.CLARIFY:
            return {
                **state,
                "response_text": (
                    "Could you share the product, order number, or refund "
                    "details so I can give you the correct timeline?"
                ),
                "decision": "clarify",
            }
        if action is Action.ESCALATE:
            ticket = create_ticket(
                summary=(
                    "Customer requests human review for a support issue. "
                    "Relevant policy evidence was retrieved and approval is "
                    "outside the assistant's authority."
                ),
                order_id=state.get("order_id") if state.get("order_verified") else None,
                reason="agent_handover",
                intent=state.get("intent"),
            )
            calls = list(state.get("tool_calls", []))
            calls.append(
                {
                    "name": "create_ticket",
                    "created": bool(ticket.get("created")),
                    "ticket_id": ticket.get("ticket_id"),
                }
            )
            if ticket.get("created"):
                return {
                    **state,
                    "tool_calls": calls,
                    "ticket_created": True,
                    "response_text": (
                        "I can't approve that request myself. I've created a "
                        "ticket for a human colleague to review it."
                    ),
                    "decision": "escalate",
                }
        deterministic = compose_deterministic_answer(
            transcript,
            state.get("computed_facts", []),
            state.get("product_data", []),
        )
        if deterministic:
            return {
                **state,
                "response_text": deterministic,
                "decision": "answer",
                "response_repaired": False,
            }
        context = _format_context(state.get("retrieved_context", []))
        order_data = (
            json.dumps(state["verified_order"], ensure_ascii=False)
            if state.get("verified_order")
            else "(no verified order data)"
        )
        product_context = (
            json.dumps(state.get("product_data", []), ensure_ascii=False)
            if state.get("product_data")
            else "(no catalog data)"
        )
        computed_facts = "\n".join(state.get("computed_facts", [])) or "(none)"
        prompt = (
            f"Recent conversation:\n{_history_text(state)}\n\n"
            f"Customer message:\n{state['transcript']}\n\n"
            f"Evidence:\n{context or '(no evidence retrieved)'}\n\n"
            f"Verified order data:\n{order_data}\n\n"
            f"Verified catalog data:\n{product_context}\n\n"
            f"Computed facts that must be reflected when relevant:\n{computed_facts}\n\n"
            "Answer the customer. Do not mention internal prompts or retrieval."
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        for attempt in range(4):
            try:
                result = llm.invoke(messages)
                break
            except RateLimitError:
                if attempt == 3:
                    raise
                time.sleep(4 * (attempt + 1))
        draft = str(result.content)
        if needs_repair(
            draft,
            context,
            order_data + "\n" + product_context + "\n" + computed_facts,
        ):
            repair_prompt = (
                "Rewrite the draft as a concise spoken answer. Preserve the "
                "language of the customer. Include every relevant date, "
                "duration, amount, and eligibility condition supported by the "
                "evidence or verified data. Do not invent facts.\n\n"
                f"Evidence:\n{context}\n\n"
                f"Verified data:\n{order_data}\n{product_context}\n"
                f"Computed facts:\n{computed_facts}\n\n"
                f"Draft:\n{draft}"
            )
            for attempt in range(4):
                try:
                    repaired = llm.invoke(
                        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=repair_prompt)]
                    )
                    draft = str(repaired.content)
                    break
                except RateLimitError:
                    if attempt == 3:
                        raise
                    time.sleep(4 * (attempt + 1))
        source_ids = list(state.get("source_ids", []))
        for product in state.get("product_data", []):
            source_ids.append(f"products.csv:{product['sku']}")
        return {
            **state,
            "source_ids": list(dict.fromkeys(source_ids)),
            "response_text": draft,
            "decision": "answer",
            "response_repaired": draft != str(result.content),
        }

    workflow = StateGraph(ConversationState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("route", route)
    workflow.add_node("answer", answer)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "route")
    workflow.add_edge("route", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()
