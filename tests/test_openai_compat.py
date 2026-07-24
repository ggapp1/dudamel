import json

import httpx
import pytest

from dudamel.exceptions import LLMError
from dudamel.llm.openai_compat import OpenAICompatProvider
from dudamel.llm.provider import ToolSpec
from dudamel.llm.types import Message, ToolCall


def make_provider(handler) -> OpenAICompatProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return OpenAICompatProvider(base_url="http://x/v1", client=client)


def ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


async def test_text_completion_and_request_shape() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return ok(
            {
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            }
        )

    p = make_provider(handler)
    c = await p.complete(
        model="qwen",
        messages=[Message(role="system", text="s"), Message(role="user", text="u")],
        tools=[ToolSpec(name="t", description="d", json_schema={"type": "object"})],
        max_tokens=99,
    )
    assert c.message.text == "hi" and c.stop_reason == "end"
    assert c.usage.tokens_in == 7 and c.usage.tokens_out == 2
    assert seen["url"].endswith("/v1/chat/completions")
    body = seen["body"]
    assert body["model"] == "qwen" and body["max_tokens"] == 99
    assert body["messages"][0] == {"role": "system", "content": "s"}
    assert body["tools"][0]["function"]["name"] == "t"


async def test_tool_call_parsing() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return ok(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "abc",
                                    "type": "function",
                                    "function": {
                                        "name": "log_workout",
                                        "arguments": '{"sets": "3"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    c = await make_provider(handler).complete(model="m", messages=[])
    assert c.stop_reason == "tool_calls"
    assert c.message.tool_calls == [ToolCall(id="abc", name="log_workout", args={"sets": "3"})]


async def test_unparseable_arguments_become_sentinel() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return ok(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "z",
                                    "type": "function",
                                    "function": {"name": "t", "arguments": "{not json"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )

    c = await make_provider(handler).complete(model="m", messages=[])
    assert c.message.tool_calls[0].args == {"__unparseable__": "{not json"}


async def test_assistant_and_tool_message_rendering() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return ok({"choices": [{"message": {"content": "k"}, "finish_reason": "stop"}]})

    msgs = [
        Message(role="assistant", tool_calls=[ToolCall(id="i1", name="t", args={"a": 1})]),
        Message(role="tool", text="result!", tool_call_id="i1"),
    ]
    await make_provider(handler).complete(model="m", messages=msgs)
    wire = seen["body"]["messages"]
    assert wire[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert wire[1] == {"role": "tool", "tool_call_id": "i1", "content": "result!"}


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (400, False)])
async def test_http_errors_map_to_llm_error(status: int, retryable: bool) -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(LLMError) as exc:
        await make_provider(handler).complete(model="m", messages=[])
    assert exc.value.retryable is retryable


async def test_malformed_tool_call_entry_raises_llm_error() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return ok(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"id": "z", "type": "function"}],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )

    with pytest.raises(LLMError, match="malformed"):
        await make_provider(handler).complete(model="m", messages=[])


async def test_non_json_body_raises_llm_error() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content="<html>oops</html>", headers={"content-type": "text/html"}
        )

    with pytest.raises(LLMError, match="non-JSON"):
        await make_provider(handler).complete(model="m", messages=[])


async def test_api_key_redacted_in_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            content=f"Unauthorized: {request.headers['Authorization']}",
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = OpenAICompatProvider(base_url="http://x/v1", api_key="sk-secret123", client=client)

    with pytest.raises(LLMError) as exc:
        await provider.complete(model="m", messages=[])
    assert "sk-secret123" not in str(exc.value)
    assert "***" in str(exc.value)
