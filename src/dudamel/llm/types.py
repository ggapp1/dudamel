"""Provider-neutral message types.

The conversation store persists these (as dicts in the messages.content JSON
column); each provider renders them to its wire format. Tiers from different
providers can therefore share one history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Message:
    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # role == "tool": which call this answers
    is_error: bool = False  # role == "tool"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(
            role=d["role"],
            text=d.get("text", ""),
            tool_calls=[ToolCall(**tc) for tc in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id"),
            is_error=d.get("is_error", False),
        )


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class Completion:
    message: Message
    usage: Usage
    stop_reason: str  # "end" | "tool_calls" | "max_tokens"
