def compose_deterministic_answer(text: str, computed_facts: list[str], products: list[dict]) -> str | None:
    lowered = text.casefold()

    if "refund" in lowered and any(
        fact.startswith("expected refund window:") for fact in computed_facts
    ):
        window = next(
            fact.split(":", 1)[1].strip()
            for fact in computed_facts
            if fact.startswith("expected refund window:")
        )
        return (
            f"Your refund is expected between {window.replace(' to ', ' and ')}. "
            "The timeline uses business days, Monday through Saturday, "
            "excluding listed holidays."
        )
    if "refund" in lowered and any(
        "wallet refund is instant within 2 hours" in fact
        for fact in computed_facts
    ):
        return "Your wallet refund is instant and should appear within 2 hours."

    if "return" in lowered and any(
        fact.startswith("return deadline:") for fact in computed_facts
    ):
        deadline = next(
            fact.split(":", 1)[1].strip()
            for fact in computed_facts
            if fact.startswith("return deadline:")
        )
        return f"Yes, you can return it until {deadline}, within the stated return window."

    if any(
        phrase in lowered
        for phrase in ("not home", "not at home", "miss the delivery", "courier comes")
    ):
        return (
            "The courier will try delivery up to 3 times. After the third "
            "failed attempt, the parcel is returned to the warehouse."
        )

    categories = {product.get("return_category") for product in products}
    if categories & {"innerwear", "digital", "grocery"} and (
        "return" in lowered or "send it back" in lowered or "fit" in lowered
    ):
        return (
            "This item cannot be returned for a change of mind or fit. "
            "The policy lists this category as non-returnable."
        )

    if (
        ("lipstick" in lowered or "beauty" in lowered)
        and any(word in lowered for word in ("opened", "open", "खोल", "खुल"))
    ):
        return (
            "Beauty products can be returned only when unopened with the seal "
            "intact. Because this item was opened, it cannot be returned."
        )

    if any(
        phrase in lowered
        for phrase in ("broken", "crack", "damaged", "टूटा", "टूटी", "खराब")
    ):
        return (
            "Please report the damaged item within 48 hours of delivery and "
            "attach photos of the item and package."
        )

    return None
