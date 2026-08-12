"""Tool-calling capability probe.

The probe answers a different question than a reachability check: not "does
this backend respond" but "can the router actually use it". It runs against
the Provider protocol so it can be driven by a scripted fake -- a probe that
could only be tested by mocking the HTTP layer would be testing the mock.
"""

from __future__ import annotations

import json

from dudamel.llm.probe import probe_tool_calling
from dudamel.llm.prompted_tools import PromptedToolsProvider
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call


async def test_probe_succeeds_when_the_backend_returns_a_tool_call() -> None:
    provider = FakeProvider([fake_tool_call("dudamel_probe", {"ok": True})])
    ok, detail = await probe_tool_calling(provider, model="m")
    assert ok is True
    assert "tool calling" in detail.lower()


async def test_probe_fails_when_the_backend_answers_with_prose() -> None:
    """A backend that ignores the tool and replies in text is reachable but
    unusable by the router -- exactly the case this probe exists to catch."""
    provider = FakeProvider([fake_text("Sure! I would call the tool now.")])
    ok, detail = await probe_tool_calling(provider, model="m")
    assert ok is False
    assert 'tool_calling = "prompted"' in detail


async def test_probe_reports_the_error_when_the_backend_raises() -> None:
    provider = FakeProvider([RuntimeError("connection refused")])
    ok, detail = await probe_tool_calling(provider, model="m")
    assert ok is False
    assert "connection refused" in detail


async def test_probe_success_through_a_prompted_wrapper_is_not_reported_as_native() -> None:
    """A tier the operator deliberately set tool_calling = "prompted" on
    (because native calling failed) must not have its probe success worded
    as "native tool calling works" -- that reads as an invitation to revert
    the workaround."""
    envelope = json.dumps({"tool_calls": [{"name": "dudamel_probe", "arguments": {"ok": True}}]})
    provider = PromptedToolsProvider(FakeProvider([fake_text(envelope)]))
    ok, detail = await probe_tool_calling(provider, model="m")
    assert ok is True
    assert "prompted" in detail.lower()
    assert "native" not in detail.lower()


async def test_probe_failure_through_a_prompted_wrapper_does_not_repeat_the_native_remedy() -> None:
    """The native-failure remedy tells the operator to set tool_calling =
    "prompted"; a prompted tier's own probe failure must not repeat advice
    they already followed."""
    provider = PromptedToolsProvider(FakeProvider([fake_text("sure, sounds good")]))
    ok, detail = await probe_tool_calling(provider, model="m")
    assert ok is False
    assert "prompted" in detail.lower()
    assert 'tool_calling = "prompted"' not in detail
