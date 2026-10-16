import re
from dataclasses import dataclass, field

from evals.schema import Assertion


@dataclass
class Observation:
    text: str
    tools_called: list[str] = field(default_factory=list)
    faq_matched: bool = False


def check_assertion(assertion: Assertion, observation: Observation) -> bool:
    """Refusal is a lexical check, not an LLM safety classifier."""
    text = observation.text.casefold()
    value = assertion.value
    match assertion.kind:
        case "contains":
            return str(value).casefold() in text
        case "not_contains":
            return str(value).casefold() not in text
        case "regex":
            return (
                re.search(str(value), observation.text, flags=re.IGNORECASE) is not None
            )
        case "tool_called":
            return value in observation.tools_called
        case "faq_match":
            return observation.faq_matched == value
        case "refusal":
            refused = bool(re.search(r"не могу|не пишу код|только.*поступлен", text))
            return refused == value
    raise ValueError("Unknown assertion kind")
