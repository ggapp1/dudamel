"""Tool-calling capability probe.

The probe answers a different question than a reachability check: not "does
this backend respond" but "can the router actually use it". It runs against
the Provider protocol so it can be driven by a scripted fake -- a probe that
could only be tested by mocking the HTTP layer would be testing the mock.
"""

from __future__ import annotations

from dudamel.llm.probe import probe_tool_calling
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
