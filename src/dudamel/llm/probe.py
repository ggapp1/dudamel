"""Does a configured backend actually support native tool calling?

`doctor`'s reachability check proves a backend answers. It does not prove the
router can use it: a backend that returns prose where a tool call belongs is
reachable and useless. This probe asks the narrower question by advertising
one trivial tool and looking at what comes back.

It costs real tokens, so it is opt-in.
"""

from __future__ import annotations

from dudamel.llm.provider import Provider, ToolSpec
from dudamel.llm.types import Message

_PROBE_TOOL = ToolSpec(
    name="dudamel_probe",
    description="Report that tool calling works. Call this tool with ok=true.",
    json_schema={
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    },
)

_PROMPT = "Call the dudamel_probe tool with ok set to true. Reply only with the tool call."


async def probe_tool_calling(provider: Provider, *, model: str) -> tuple[bool, str]:
    """Whether `model` returns a well-formed native tool call."""
    try:
        completion = await provider.complete(
            model=model,
            messages=[Message(role="user", text=_PROMPT)],
            tools=[_PROBE_TOOL],
            max_tokens=256,
        )
    except Exception as e:
        return False, f"probe failed ({type(e).__name__}: {e})"

    if completion.message.tool_calls:
        return True, "native tool calling works"
    return (
        False,
        'no usable native tool calling — set tool_calling = "prompted"',
    )
