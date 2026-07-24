import json
from pathlib import Path

import pytest

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TierConfig
from dudamel.exceptions import LLMError, RegistryError
from dudamel.llm.testing import FakeProvider, fake_text


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
