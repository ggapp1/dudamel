"""Context assembly: token-budgeted, turn-snapped, tool-pair-safe.

Provider APIs reject histories where an assistant tool_use has no matching
tool result (or vice versa). We therefore only ever cut at TURN boundaries —
a turn is one user message plus everything the assistant did in response.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from dudamel.llm.types import Message


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def message_tokens(m: Message) -> int:
    total = estimate_tokens(m.text)
    for tc in m.tool_calls:
        total += estimate_tokens(tc.name + json.dumps(tc.args, default=str))
    return total


def truncate_tool_result(m: Message, cap_chars: int) -> Message:
    if m.role != "tool" or len(m.text) <= cap_chars:
        return m
    if re.search(r"…\[truncated \d+ chars\]$", m.text):
        return m
    dropped = len(m.text) - cap_chars
    return replace(m, text=m.text[:cap_chars] + f"…[truncated {dropped} chars]")


def _split_turns(messages: list[Message]) -> list[list[Message]]:
    turns: list[list[Message]] = []
    current: list[Message] = []
    for m in messages:
        if m.role == "user":
            if current:
                turns.append(current)
            current = [m]
        elif current:
            current.append(m)
        # non-user message before any user message: orphan — dropped
    if current:
        turns.append(current)
    return turns


def build_window(
    messages: list[Message],
    *,
    token_budget: int,
    tool_result_cap: int = 8192,
) -> list[Message]:
    capped = [truncate_tool_result(m, tool_result_cap) for m in messages]
    turns = _split_turns(capped)
    if not turns:
        return []
    window: list[list[Message]] = [turns[-1]]  # newest turn is non-negotiable
    spent = sum(message_tokens(m) for m in turns[-1])
    for t in reversed(turns[:-1]):
        cost = sum(message_tokens(m) for m in t)
        if spent + cost > token_budget:
            break
        window.insert(0, t)
        spent += cost
    return [m for t in window for m in t]
