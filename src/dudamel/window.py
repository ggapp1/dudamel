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

# Suffix a previous truncation left behind. Recognised ONLY to avoid
# stacking a second marker onto text this function already capped; it is
# never trusted as evidence that the text is short (see
# `truncate_tool_result`).
_MARKER_RE = re.compile(r"…\[truncated \d+ chars\]$")


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
    marker = _MARKER_RE.search(m.text)
    if marker is not None and len(m.text) - len(marker.group(0)) <= cap_chars:
        # Text this function already capped: body within the cap, marker
        # appended. Re-marking it would stack a second suffix every turn.
        # The length test is what makes this safe -- the marker itself is
        # in-band and attacker-controlled (an MCP tool result is untrusted
        # input), so a multi-megabyte blob that merely ENDS with the marker
        # falls through and gets capped like any other oversized result.
        return m
    dropped = len(m.text) - cap_chars
    return replace(m, text=m.text[:cap_chars] + f"…[truncated {dropped} chars]")


def _drop_dangling_tool_calls(messages: list[Message]) -> list[Message]:
    """Crash-window sanitizer: if the process died between appending an
    assistant's tool_calls message and appending its tool result(s), that
    assistant message survives in the DB with an unanswered tool_call.
    Provider APIs reject any tool_use without a matching tool_result, so drop
    such an assistant message outright (a partial match doesn't help either —
    the provider still rejects it), and drop any tool result that referenced
    it too, so no message is left answering a tool_call that no longer
    appears in the window."""
    answered = {m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id is not None}
    kept = [
        m
        for m in messages
        if not (
            m.role == "assistant"
            and m.tool_calls
            and any(tc.id not in answered for tc in m.tool_calls)
        )
    ]
    called = {tc.id for m in kept for tc in m.tool_calls}
    return [m for m in kept if not (m.role == "tool" and m.tool_call_id not in called)]


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
    return _drop_dangling_tool_calls([m for t in window for m in t])
