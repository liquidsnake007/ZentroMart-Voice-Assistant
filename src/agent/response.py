import re


def _facts_in_context(context: str) -> list[str]:
    facts: list[str] = []
    for value in re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d+(?:\.\d+)?\b", context):
        if value not in facts:
            facts.append(value)
    return facts


def needs_repair(response: str, context: str, verified_data: str) -> bool:
    if not response.strip() or len(response.split()) < 5:
        return True
    grounded = context + "\n" + verified_data
    facts = _facts_in_context(grounded)
    if not facts:
        return False
    response_lower = response.casefold()
    mentioned = sum(
        1
        for fact in facts
        if fact.casefold() in response_lower
        or fact.replace("-", " ") in response_lower
    )
    return mentioned == 0 and len(facts) >= 2
