import pytest

from dudamel.contract.schema import ToolSchema
from dudamel.contract.types import Tool
from dudamel.exceptions import LLMError
from dudamel.llm.provider import ToolSpec
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call
from dudamel.llm.types import Message


async def sample(query: str) -> str:
    """Search."""
    return query


def make_tool() -> Tool:
    return Tool(
        name="sample",
        app_name="a",
        description="Search.",
        fn=sample,
        schema=ToolSchema(sample),
        read_only=True,
        confirm=False,
        timeout=30.0,
    )


def test_toolspec_from_tool() -> None:
    spec = ToolSpec.from_tool(make_tool())
    assert spec.name == "sample" and spec.description == "Search."
    assert spec.json_schema["properties"]["query"] == {"type": "string"}


async def test_fake_provider_replays_script_and_records_calls() -> None:
    fp = FakeProvider([fake_tool_call("sample", {"query": "x"}), fake_text("done")])
    c1 = await fp.complete(model="m", messages=[Message(role="user", text="go")])
    assert c1.stop_reason == "tool_calls"
    assert c1.message.tool_calls[0].args == {"query": "x"}
    c2 = await fp.complete(model="m", messages=[])
    assert c2.message.text == "done" and c2.stop_reason == "end"
    assert len(fp.calls) == 2 and fp.calls[0]["messages"][0].text == "go"


async def test_fake_provider_raises_scripted_exception() -> None:
    fp = FakeProvider([LLMError("boom", retryable=True)])
    with pytest.raises(LLMError, match="boom"):
        await fp.complete(model="m", messages=[])


async def test_fake_provider_exhausted_script() -> None:
    fp = FakeProvider([])
    with pytest.raises(LLMError, match="exhausted"):
        await fp.complete(model="m", messages=[])


def test_llm_error_retryable_flag() -> None:
    assert LLMError("x").retryable is False
    assert LLMError("x", retryable=True).retryable is True


async def test_empty_tools_list_recorded_faithfully() -> None:
    fp = FakeProvider([fake_text("response")])
    await fp.complete(model="m", messages=[], tools=[])
    assert fp.calls[0]["tools"] == []
