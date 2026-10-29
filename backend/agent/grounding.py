"""Grounding check: every number in the agent's explanation must come from a tool result.

An LLM can write fluent text with invented figures. This check finds each number in
the text and looks for it among the numbers the tools returned (or in the owner's own
request). Rounding is allowed to the precision written, and a value may be quoted as a
percentage (0.083 as 8.3%). Anything else is reported as ungrounded.

Limits: this finds numbers that were never produced by a tool. It does not check that a
real number is used correctly (for example attached to the wrong ad). It also cannot
see numbers written as words, such as "three".
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*•]\s+)?\d+[.)]\s+")
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?|\.\d+)(%?)")


# Past-tense or "is live" claims that something already happened. Plans are only proposed, so any
# such claim is false. Future and conditional wording ("will be executed after you approve") is allowed.
_ACTION_CLAIM = re.compile(
    r"\b(?:(?:i|we)\s+(?:have\s+|'ve\s+)?(?:executed|launched|ran|started|deployed|published|spent)"
    r"|(?:has|have|was|were)\s+(?:already\s+)?(?:been\s+)?(?:executed|launched|started|deployed|published|run)"
    r"|(?:is|are)\s+(?:now\s+)?(?:live|running)"
    r"|went\s+live)\b",
    re.IGNORECASE,
)


def claims_action_taken(text: str) -> bool:
    """True if the text says a campaign was already run, launched or is live."""
    return bool(_ACTION_CLAIM.search(text))


@dataclass
class GroundingReport:
    ok: bool
    checked: int
    ungrounded: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "numbers_checked": self.checked, "ungrounded": self.ungrounded}


def extract_numbers(text: str) -> list[tuple[str, float, int]]:
    """(text as written, value, decimals written) for each number. List markers like '1.' are skipped."""
    out = []
    for match in _NUMBER.finditer(_LIST_MARKER.sub("", text)):
        raw = match.group(1)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append((match.group(0), float(raw.replace(",", "")), decimals))
    return out


def collect_numbers(obj, out: list[float] | None = None) -> list[float]:
    """Every numeric value inside nested dicts and lists (booleans and text are ignored)."""
    out = [] if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(abs(float(obj)))
    elif isinstance(obj, dict):
        for value in obj.values():
            collect_numbers(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            collect_numbers(value, out)
    return out


def check_grounding(text: str, tool_results: list, extra_texts: tuple[str, ...] = ()) -> GroundingReport:
    candidates = collect_numbers(tool_results)
    for extra in extra_texts:
        candidates.extend(value for _, value, _ in extract_numbers(extra))

    numbers = extract_numbers(text)
    ungrounded = []
    for raw, value, decimals in numbers:
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(value - c) <= tolerance or abs(value - c * 100) <= tolerance for c in candidates):
            ungrounded.append(raw)
    return GroundingReport(ok=not ungrounded, checked=len(numbers), ungrounded=ungrounded)
