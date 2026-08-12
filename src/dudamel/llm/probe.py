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

# `PromptedToolsProvider.name` is always f"prompted+{inner.name}" -- this is
# the one signal probe_tool_calling has (short of importing the class) that
# `provider` is the JSON-fallback wrapper rather than a raw backend.
_PROMPTED_PREFIX = "prompted+"


async def probe_tool_calling(provider: Provider, *, model: str) -> tuple[bool, str]:
    """Whether `model` can produce a usable tool call through `provider`.

    `provider` may be a raw backend (native wire tool calling) or a
    `PromptedToolsProvider` (the prompted-JSON fallback for a tier
    configured `tool_calling = "prompted"`) -- `_probe_tier_tool_calling`
    in cli.py builds it via the same `build_provider` the router uses, so
    whichever one the tier is actually configured for is what gets probed.
    The wording below distinguishes the two: reporting a prompted tier's
    success as "native tool calling works" would read as an invitation to
    revert the very workaround the operator set deliberately, and reporting
    its failure with 'set tool_calling = "prompted"' would tell them to do
    something they already did.
    """
    prompted = provider.name.startswith(_PROMPTED_PREFIX)
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
        if prompted:
            return True, "prompted tool-calling fallback works"
        return True, "native tool calling works"
    if prompted:
        return (
            False,
            "prompted tool-calling fallback did not produce a usable call for "
            "this model — it may not follow the JSON-envelope instructions "
            "reliably enough for tool use here",
        )
    return (
        False,
        'no usable native tool calling — set tool_calling = "prompted"',
    )
