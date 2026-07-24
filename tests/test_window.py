import random

from dudamel.llm.types import Message, ToolCall
from dudamel.window import build_window, estimate_tokens, truncate_tool_result


def turn(i: int, with_tools: bool = False) -> list[Message]:
    msgs = [Message(role="user", text=f"question {i} " + "x" * 100)]
    if with_tools:
        msgs.append(
            Message(role="assistant", tool_calls=[ToolCall(id=f"tc{i}", name="t", args={"i": i})])
        )
        msgs.append(Message(role="tool", text=f"result {i}", tool_call_id=f"tc{i}"))
    msgs.append(Message(role="assistant", text=f"answer {i}"))
    return msgs


def test_estimate_tokens_floor() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 400) == 100


def test_truncate_tool_result() -> None:
    m = Message(role="tool", text="A" * 100, tool_call_id="t")
    out = truncate_tool_result(m, cap_chars=40)
    assert len(out.text) < 100 and out.text.endswith("chars]")
    assert m.text == "A" * 100  # original untouched


def test_newest_turn_always_included() -> None:
    msgs = turn(1)
    assert build_window(msgs, token_budget=1) == msgs  # over budget but newest turn stays


def test_budget_drops_old_turns_at_turn_boundaries() -> None:
    msgs = turn(1) + turn(2) + turn(3)
    # Each turn here costs ~29 tokens (question ~27 + answer ~2); budget=60
    # covers the newest turn (non-negotiable) plus exactly one more, forcing
    # the oldest turn out. budget=100 as originally drafted covers all three
    # turns (~87 tokens total) and does not exercise the drop path.
    win = build_window(msgs, token_budget=60)
    # must start at a user message and contain the newest turn intact
    assert win[0].role == "user"
    assert win[-1].text == "answer 3"
    assert "question 1" not in win[0].text  # oldest turn dropped first


def test_never_splits_tool_pairs() -> None:
    random.seed(7)
    msgs: list[Message] = []
    for i in range(30):
        msgs += turn(i, with_tools=random.random() < 0.5)
    for budget in (50, 150, 400, 1000, 5000):
        win = build_window(msgs, token_budget=budget)
        ids_called = {tc.id for m in win for tc in m.tool_calls}
        ids_answered = {m.tool_call_id for m in win if m.role == "tool"}
        assert ids_called == ids_answered, f"split pair at budget={budget}"
        assert win[0].role == "user"


def test_leading_orphan_tool_messages_dropped() -> None:
    msgs = [
        Message(role="tool", text="orphan", tool_call_id="ghost"),
        *turn(1),
    ]
    win = build_window(msgs, token_budget=10_000)
    assert win[0].role == "user"
    assert all(m.tool_call_id != "ghost" for m in win)


def test_tool_results_truncated_in_window() -> None:
    msgs = [
        Message(role="user", text="q"),
        Message(role="assistant", tool_calls=[ToolCall(id="a", name="t", args={})]),
        Message(role="tool", text="B" * 10_000, tool_call_id="a"),
        Message(role="assistant", text="done"),
    ]
    win = build_window(msgs, token_budget=100_000, tool_result_cap=100)
    tool_msg = next(m for m in win if m.role == "tool")
    assert len(tool_msg.text) < 200 and "truncated" in tool_msg.text
