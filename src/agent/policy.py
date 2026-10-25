from dataclasses import dataclass
from enum import StrEnum
import re


class Action(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    VERIFY_ORDER = "verify_order"
    REFUSE = "refuse"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Signals:
    has_order_id: bool
    has_verification_value: bool
    generic_order_request: bool
    privacy_request: bool
    missing_specific_context: bool
    human_request: bool
    fraud_report: bool
    damage_or_wrong_item: bool
    repeat_contact: bool
    authority_request: bool


def extract_signals(text: str, order_id: str | None) -> Signals:
    lowered = text.casefold()
    has_verification_value = any(
        len(token) in (4, 6) and token != order_id
        for token in re.findall(r"\b\d+\b", text)
    )
    return Signals(
        has_order_id=order_id is not None,
        has_verification_value=has_verification_value,
        generic_order_request=(
            order_id is None
            and ("order" in lowered or "ऑर्डर" in text)
            and any(word in lowered for word in ("help", "madad", "मदद", "सहायता"))
        ),
        privacy_request=any(
            phrase in lowered
            for phrase in (
                "phone number of the person",
                "phone number of whoever",
                "number of the person",
                "personal information",
                "फोन नंबर",
            )
        ),
        missing_specific_context=(
            ("how many days" in lowered and "return" in lowered)
            or ("कितने दिन" in text and "वापस" in text)
            or "when will my refund come" in lowered
            or "refund kab" in lowered
            or "रिफ़ंड कब" in text
        ),
        human_request=any(
            phrase in lowered
            for phrase in ("human", "person", "इंसान", "व्यक्ति", "colleague", "escalate")
        ),
        fraud_report=any(
            phrase in lowered
            for phrase in ("fraud", "never placed", "didn't place", "not my order")
        ),
        damage_or_wrong_item=any(
            phrase in lowered
            for phrase in ("crack", "damaged", "wrong item", "broken")
        ),
        repeat_contact=any(
            phrase in lowered
            for phrase in ("third time", "three times", "teesri baar")
        ),
        authority_request=any(
            phrase in lowered
            for phrase in ("approve", "approved", "priority resolution")
        ),
    )


def choose_action(
    signals: Signals,
    *,
    order_verified: bool,
    verification_attempts: int,
) -> Action:
    if signals.privacy_request:
        return Action.REFUSE
    if signals.generic_order_request:
        return Action.CLARIFY
    if signals.has_order_id and not order_verified:
        return Action.REFUSE if verification_attempts >= 2 else Action.VERIFY_ORDER
    if (
        signals.human_request
        or signals.fraud_report
        or signals.repeat_contact
        or signals.authority_request
    ):
        return Action.ESCALATE
    if signals.damage_or_wrong_item and signals.has_order_id and order_verified:
        return Action.ESCALATE
    if signals.missing_specific_context:
        return Action.CLARIFY
    return Action.ANSWER
