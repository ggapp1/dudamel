from dudamel.llm.types import Completion, Message, ToolCall, Usage


def test_message_dict_roundtrip_text() -> None:
    m = Message(role="user", text="hi")
    assert Message.from_dict(m.to_dict()) == m


def test_message_dict_roundtrip_assistant_tool_calls() -> None:
    m = Message(
        role="assistant",
        text="",
        tool_calls=[ToolCall(id="tc1", name="log_workout", args={"sets": 3})],
    )
    d = m.to_dict()
    assert d["tool_calls"][0]["name"] == "log_workout"
    assert Message.from_dict(d) == m


def test_message_dict_roundtrip_tool_result() -> None:
    m = Message(role="tool", text="done", tool_call_id="tc1", is_error=True)
    assert Message.from_dict(m.to_dict()) == m


def test_defaults_are_not_shared() -> None:
    a, b = Message(role="user"), Message(role="user")
    a.tool_calls.append(ToolCall(id="x", name="y", args={}))
    assert b.tool_calls == []


def test_completion_shape() -> None:
    c = Completion(
        message=Message(role="assistant", text="ok"),
        usage=Usage(tokens_in=10, tokens_out=2),
        stop_reason="end",
    )
    assert c.usage.tokens_in == 10 and c.stop_reason == "end"
