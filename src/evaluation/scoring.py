from dataclasses import dataclass, field
import re
from typing import Iterable


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def fact_matches(reply: str, fact_spec: str) -> bool:
    alternatives = [part.strip() for part in fact_spec.split(" / ") if part.strip()]
    normalised = _normalise(reply)
    return any(_normalise(option) in normalised for option in alternatives)


def all_required_facts(reply: str, required: str) -> bool:
    facts = [fact.strip() for fact in required.split(" ; ") if fact.strip()]
    return all(fact_matches(reply, fact) for fact in facts)


def contains_forbidden_fact(reply: str, forbidden: str) -> bool:
    facts = [fact.strip() for fact in forbidden.split(" ; ") if fact.strip()]
    return any(fact_matches(reply, fact) for fact in facts)


def expected_tool_name(expected: str) -> str | None:
    match = re.search(r"([A-Za-z_]+)\(", expected)
    return match.group(1) if match else None


@dataclass
class EvaluationResult:
    total: int = 0
    decision_correct: int = 0
    required_facts_present: int = 0
    forbidden_facts_absent: int = 0
    tool_expectations_met: int = 0
    failures: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "decision_accuracy": self.decision_correct / self.total if self.total else 0,
            "required_fact_rate": (
                self.required_facts_present / self.total if self.total else 0
            ),
            "forbidden_fact_rate": (
                self.forbidden_facts_absent / self.total if self.total else 0
            ),
            "tool_expectation_rate": (
                self.tool_expectations_met / self.total if self.total else 0
            ),
            "failures": self.failures,
        }


def score_rows(rows: Iterable[dict], outputs: Iterable[dict]) -> EvaluationResult:
    result = EvaluationResult()
    for row, output in zip(rows, outputs):
        result.total += 1
        reply = str(output.get("response_text", ""))
        decision = output.get("decision", "")
        acceptable = set(
            filter(None, (row.get("acceptable_decisions") or row.get("expected_decision", "")).split(" ; "))
        )
        decision_ok = decision in acceptable
        required_ok = all_required_facts(reply, row.get("must_include", ""))
        forbidden_ok = not contains_forbidden_fact(reply, row.get("must_not_include", ""))
        expected_tool = expected_tool_name(row.get("expected_tool_calls", ""))
        actual_tools = {call.get("name") for call in output.get("tool_calls", [])}
        tool_ok = expected_tool is None or expected_tool in actual_tools
        result.decision_correct += int(decision_ok)
        result.required_facts_present += int(required_ok)
        result.forbidden_facts_absent += int(forbidden_ok)
        result.tool_expectations_met += int(tool_ok)
        if not all((decision_ok, required_ok, forbidden_ok, tool_ok)):
            result.failures.append(
                {
                    "id": row.get("id") or f"{row.get('conversation_id')}:{row.get('turn')}",
                    "decision": decision,
                    "expected_decision": row.get("expected_decision"),
                    "required_facts_ok": required_ok,
                    "forbidden_facts_ok": forbidden_ok,
                    "tool_ok": tool_ok,
                }
            )
    return result
