import asyncio
import json
import time
from pathlib import Path

import pytest

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TierConfig
from dudamel.exceptions import LLMError, RegistryError
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call


def make_orc() -> Orchestrator:
    app = App("gym", description="d")

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record."""
        return f"ok {exercise}"

    return Orchestrator(apps=[app])


def make_settings(tmp_path: Path, **tiers: TierConfig) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/rt.db",
        data_dir=tmp_path,
        llm_tiers=tiers or {"standard": TierConfig(provider="fake", model="f")},
    )


async def test_chat_end_to_end_with_fake_provider(tmp_path) -> None:
    rt = Runtime(
        make_orc(),
        make_settings(tmp_path),
        providers={"standard": FakeProvider([fake_text("hello!")])},
    )
    await rt.start()
    reply = await rt.chat("web:1", "hi", user_id="u1")
    assert reply.text == "hello!"
    await rt.stop()


async def test_app_llm_binding(tmp_path) -> None:
    orc = make_orc()
    rt = Runtime(
        orc,
        make_settings(tmp_path),
        providers={"standard": FakeProvider([fake_text("answer")])},
    )
    await rt.start()
    app = orc.registry.apps["gym"]
    assert await app.llm("question") == "answer"
    await rt.stop()


async def test_app_llm_schema_returns_dict(tmp_path) -> None:
    orc = make_orc()
    rt = Runtime(
        orc,
        make_settings(tmp_path),
        providers={
            "standard": FakeProvider([fake_text(json.dumps({"a": 1})), fake_text("not json")])
        },
    )
    await rt.start()
    app = orc.registry.apps["gym"]
    out = await app.llm("q", schema={"type": "object"})
    assert out == {"a": 1}
    with pytest.raises(LLMError, match="JSON"):
        await app.llm("q", schema={"type": "object"})
    await rt.stop()


async def test_app_notify_fallback_warns(tmp_path, caplog) -> None:
    orc = make_orc()
    rt = Runtime(
        orc,
        make_settings(tmp_path),
        providers={"standard": FakeProvider([])},
    )
    await rt.start()
    await orc.registry.apps["gym"].notify("digest ready")
    assert any("notify (no channel configured)" in r.message for r in caplog.records)
    await rt.stop()


async def test_start_applies_core_migrations(tmp_path) -> None:
    from sqlalchemy import create_engine, inspect

    from dudamel.migrate import sync_url

    settings = make_settings(tmp_path)
    rt = Runtime(make_orc(), settings, providers={"standard": FakeProvider([])})
    await rt.start()
    insp = inspect(create_engine(sync_url(settings.database_url)))
    assert "llm_calls" in insp.get_table_names()
    await rt.stop()


def test_openai_tier_requires_base_url(tmp_path) -> None:
    with pytest.raises(RegistryError, match="base_url"):
        Runtime(
            make_orc(),
            make_settings(
                tmp_path,
                standard=TierConfig(provider="openai-compatible", model="m"),
            ),
        )


def test_anthropic_tier_requires_key_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RegistryError, match="ANTHROPIC_API_KEY"):
        Runtime(
            make_orc(),
            make_settings(tmp_path, standard=TierConfig(provider="anthropic", model="m")),
        )


def test_fake_tier_requires_override(tmp_path) -> None:
    with pytest.raises(RegistryError, match="override"):
        Runtime(make_orc(), make_settings(tmp_path))  # fake tier, no providers=


# --- Plan 3 Task 1: Runtime extensions --------------------------------------


async def test_list_pending_confirmations(tmp_path) -> None:
    app = App("gym", description="d")

    @app.tool(confirm=True)
    async def wipe(reason: str) -> str:
        """Delete stuff."""
        return "wiped"

    orc = Orchestrator(apps=[app])
    rt = Runtime(
        orc,
        make_settings(tmp_path),
        providers={
            "standard": FakeProvider(
                [fake_tool_call("wipe", {"reason": "x"}), fake_text("okay, cancelled")]
            )
        },
    )
    await rt.start()
    reply = await rt.chat("chan:1", "wipe it", user_id="u1")
    assert reply.pending_confirmation_id

    pending = await rt.list_pending_confirmations()
    assert len(pending) == 1
    p = pending[0]
    assert p["id"] == reply.pending_confirmation_id
    assert p["tool"] == "wipe"
    assert p["args"] == {"reason": "x"}
    assert "created_at" in p and "expires_at" in p

    assert len(await rt.list_pending_confirmations(channel="chan:1")) == 1
    assert await rt.list_pending_confirmations(channel="chan:2") == []

    await rt.resolve_confirmation(reply.pending_confirmation_id, approved=False, user_id="u1")
    assert await rt.list_pending_confirmations() == []  # resolved, no longer pending
    await rt.stop()


async def test_bind_notify_rebinds_every_app(tmp_path) -> None:
    orc = make_orc()
    rt = Runtime(orc, make_settings(tmp_path), providers={"standard": FakeProvider([])})
    await rt.start()
    sent: list[str] = []

    async def fake_notify(text: str) -> None:
        sent.append(text)

    rt.bind_notify(fake_notify)
    await orc.registry.apps["gym"].notify("hi")
    assert sent == ["hi"]
    await rt.stop()


async def test_render_widgets_runs_concurrently(tmp_path) -> None:
    app = App("stats", description="d")

    @app.widget(title="A", renderer="markdown")
    async def a() -> str:
        await asyncio.sleep(0.05)
        return "a"

    @app.widget(title="B", renderer="markdown")
    async def b() -> str:
        await asyncio.sleep(0.05)
        return "b"

    orc = Orchestrator(apps=[app])
    rt = Runtime(orc, make_settings(tmp_path), providers={"standard": FakeProvider([])})
    await rt.start()
    started = time.monotonic()
    out = await rt.render_widgets()
    elapsed = time.monotonic() - started
    assert elapsed < 0.09  # concurrent (~0.05s), not serial (~0.1s)
    assert {w["id"] for w in out} == {"a", "b"}
    await rt.stop()


async def test_render_widgets_reports_widget_errors(tmp_path) -> None:
    app = App("stats", description="d")

    @app.widget(title="Good", renderer="markdown")
    async def good() -> str:
        return "ok"

    @app.widget(title="Bad", renderer="stat")
    async def bad() -> dict:
        raise RuntimeError("boom")

    orc = Orchestrator(apps=[app])
    rt = Runtime(orc, make_settings(tmp_path), providers={"standard": FakeProvider([])})
    await rt.start()
    out = await rt.render_widgets()
    by_id = {w["id"]: w for w in out}
    assert by_id["good"]["data"] == "ok"
    assert by_id["bad"]["error"] == "boom"
    await rt.stop()
