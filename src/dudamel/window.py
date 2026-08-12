"""Context assembly: token-budgeted, turn-snapped, tool-pair-safe.

Provider APIs reject histories where an assistant tool_use has no matching
tool result (or vice versa). We therefore only ever cut at TURN boundaries —
a turn is one user message plus everything the assistant did in response.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from dudamel.llm.types import Message

# Suffix a previous truncation left behind. Recognised ONLY to avoid
# stacking a second marker onto text this function already capped; it is
# never trusted as evidence that the text is short (see
# `truncate_tool_result`). The digit run is bounded because the marker is
# attacker-supplied text, not a promise: a drop count this function writes
# is a Python int's decimal length, never anywhere near 20 digits.
_MARKER_RE = re.compile(r"…\[truncated \d{1,20} chars\]$")

# Ceiling on a marker this function can produce, given the bound above.
_MAX_MARKER_CHARS = len("…[truncated  chars]") + 20


@dataclass(frozen=True)
class WindowBuild:
    """What `build_window` produced, plus how much of the input it cut.

    `dropped` counts ONLY the leading messages the build removed -- the
    turns that did not fit the budget, plus any orphan messages before the
    first user message. It is deliberately NOT `len(messages_in) -
    len(messages)`: `_drop_dangling_tool_calls` also removes messages, from
    anywhere INSIDE the kept span, and a caller that inferred the count by
    subtraction would read those removals as extra dropped history. The
    compactor summarizes `history[:dropped]` and derives its watermark from
    the same offset, so an inflated count makes it summarize (and mark as
    permanently covered) messages that are still in the window.
    """

    messages: list[Message]
    dropped: int


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
    if len(m.text) <= cap_chars + _MAX_MARKER_CHARS and _MARKER_RE.search(m.text):
        # Text this function already capped: short enough to be a body
        # within the cap plus one marker. Re-marking it would stack a second
        # suffix every turn.
        #
        # The LENGTH ceiling is what makes this safe, and it is checked
        # first: the marker is in-band and attacker-controlled (an MCP tool
        # result is untrusted input), so nothing about the text's shape can
        # buy an exemption. Anything longer than a capped body plus a marker
        # gets capped, whether the excess sits in the body, in the marker's
        # digits, or anywhere else.
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
) -> WindowBuild:
    capped = [truncate_tool_result(m, tool_result_cap) for m in messages]
    turns = _split_turns(capped)
    if not turns:
        return WindowBuild(messages=[], dropped=len(messages))
    window: list[list[Message]] = [turns[-1]]  # newest turn is non-negotiable
    spent = sum(message_tokens(m) for m in turns[-1])
    for t in reversed(turns[:-1]):
        cost = sum(message_tokens(m) for m in t)
        if spent + cost > token_budget:
            break
        window.insert(0, t)
        spent += cost
    kept = [m for t in window for m in t]
    # Everything the cut left behind is a contiguous PREFIX of `messages`:
    # `_split_turns` only ever discards orphans ahead of the first user
    # message, and the budget loop only ever refuses whole older turns. The
    # sanitizer below is the one removal that isn't prefix-shaped, which is
    # exactly why it is applied after the count is taken.
    dropped = len(messages) - len(kept)
    return WindowBuild(messages=_drop_dangling_tool_calls(kept), dropped=dropped)
