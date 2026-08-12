import asyncio
import json
import logging
import time
from pathlib import Path

import pytest

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TierConfig
from dudamel.exceptions import DudamelError, LLMError, RegistryError
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call
from dudamel.migrate import (
    ensure_app_migrations,
    generate_app_migration,
    pending_migrations,
    upgrade_core,
)


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
        # These tests place their migrations/ under tmp_path, so that is the
        # project directory app migrations resolve from (Settings.load sets
        # this to the CWD for a real CLI invocation).
        project_dir=tmp_path,
        llm_tiers=tiers or {"standard": TierConfig(provider="fake", model="f")},
    )


async def test_warns_when_migrations_live_under_data_dir_not_project_dir(tmp_path, caplog) -> None:
    """A programmatic embedder who builds Settings(data_dir=X) directly
    (not via Settings.load) and puts app migrations under X gets them
    silently ignored -- start() resolves migrations from project_dir. Warn
    so the operator sees why their migrations never ran."""
    data_dir = tmp_path / "data"
    project_dir = tmp_path / "project"
    (data_dir / "migrations").mkdir(parents=True)
    project_dir.mkdir()
    database_url = f"sqlite+aiosqlite:///{tmp_path}/rt.db"
    upgrade_core(database_url)  # so the auto_migrate=False gate passes cleanly
    settings = Settings(
        database_url=database_url,
        data_dir=data_dir,
        project_dir=project_dir,
        auto_migrate=False,
        llm_tiers={"standard": TierConfig(provider="fake", model="f")},
    )
    rt = Runtime(make_orc(), settings, providers={"standard": FakeProvider([])})
    with caplog.at_level(logging.WARNING, logger="dudamel.runtime"):
        await rt.start()
    assert any(
        "migrations" in r.message and "data_dir" in r.message and "project_dir" in r.message
        for r in caplog.records
    )
    await rt.stop()


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


async def test_start_auto_migrates_by_default(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.auto_migrate is True
    rt = Runtime(Orchestrator(apps=[]), settings, providers={"standard": FakeProvider([])})
    await rt.start()  # must not raise; schema is created
    await rt.stop()


async def test_start_refuses_to_migrate_when_auto_migrate_is_false(tmp_path: Path) -> None:
    """A production deployment opts out so a restart cannot mutate its schema;
    startup fails loudly naming the command to run instead."""
    settings = make_settings(tmp_path)
    settings.auto_migrate = False
    rt = Runtime(Orchestrator(apps=[]), settings, providers={"standard": FakeProvider([])})
    with pytest.raises(DudamelError) as exc:
        await rt.start()
    assert "dudamel db migrate" in str(exc.value)


async def test_start_succeeds_when_auto_migrate_is_false_and_schema_is_current(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    rt = Runtime(Orchestrator(apps=[]), settings, providers={"standard": FakeProvider([])})
    await rt.start()
    await rt.stop()
    settings.auto_migrate = False
    rt2 = Runtime(Orchestrator(apps=[]), settings, providers={"standard": FakeProvider([])})
    await rt2.start()  # already at head — nothing to do, must not raise
    await rt2.stop()


async def test_start_refuses_when_app_migration_is_pending(tmp_path: Path) -> None:
    """auto_migrate=False must gate the app tier too, not just core -- an
    app migration script generated but never applied is exactly what a
    core-only check would miss."""
    from dudamel.migrate import ensure_app_migrations, generate_app_migration, upgrade_core

    app = App("blog", description="d")

    class Post(app.Model):
        title: str

    settings = make_settings(tmp_path)
    upgrade_core(settings.database_url)
    ensure_app_migrations(tmp_path)
    generate_app_migration(Orchestrator(apps=[app]), settings.database_url, "add posts", tmp_path)

    settings.auto_migrate = False
    rt = Runtime(Orchestrator(apps=[]), settings, providers={"standard": FakeProvider([])})
    with pytest.raises(DudamelError) as exc:
        await rt.start()
    assert "dudamel db migrate" in str(exc.value)
    assert "app" in str(exc.value)


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


def test_compact_dropped_turns_requires_compaction_tier(tmp_path) -> None:
    from dudamel.config import RouterConfig

    settings = make_settings(tmp_path)
    settings.router = RouterConfig(compact_dropped_turns=True)
    with pytest.raises(RegistryError, match="compaction_tier"):
        Runtime(make_orc(), settings, providers={"standard": FakeProvider([])})


def test_compact_dropped_turns_rejects_an_unconfigured_tier(tmp_path) -> None:
    from dudamel.config import RouterConfig

    settings = make_settings(tmp_path)
    settings.router = RouterConfig(compact_dropped_turns=True, compaction_tier="ghost")
    with pytest.raises(RegistryError, match="unknown tier 'ghost'; configured tiers:.*standard"):
        Runtime(make_orc(), settings, providers={"standard": FakeProvider([])})


async def test_compact_dropped_turns_builds_a_compactor_for_a_configured_tier(tmp_path) -> None:
    from dudamel.compaction import Compactor
    from dudamel.config import RouterConfig

    settings = make_settings(tmp_path)
    settings.router = RouterConfig(compact_dropped_turns=True, compaction_tier="standard")
    rt = Runtime(make_orc(), settings, providers={"standard": FakeProvider([])})
    assert isinstance(rt._compactor, Compactor)
    assert rt._router._compactor is rt._compactor


def test_build_provider_wraps_a_prompted_tier() -> None:
    from dudamel.runtime import build_provider

    cfg = TierConfig(
        provider="openai-compatible", model="m", base_url="http://x", tool_calling="prompted"
    )
    provider = build_provider("standard", cfg)
    assert provider.name == "prompted+openai-compatible"


def test_build_provider_leaves_a_native_tier_unwrapped() -> None:
    from dudamel.runtime import build_provider

    cfg = TierConfig(provider="openai-compatible", model="m", base_url="http://x")
    provider = build_provider("standard", cfg)
    assert provider.name == "openai-compatible"


async def test_prompted_tier_wraps_a_providers_override_end_to_end(tmp_path) -> None:
    """A providers= override (how a test scripts a fake backend, and the
    only way `provider="fake"` is usable at all) still gets wrapped when
    the tier's config says tool_calling="prompted": scripted completion
    emitting the prompted JSON envelope -> real tool executes -> scripted
    final text closes the turn, exactly like a native tool-calling turn."""
    orc = make_orc()
    envelope = json.dumps(
        {"tool_calls": [{"name": "log_workout", "arguments": {"exercise": "run"}}]}
    )
    settings = make_settings(
        tmp_path,
        standard=TierConfig(provider="fake", model="f", tool_calling="prompted"),
    )
    rt = Runtime(
        orc,
        settings,
        providers={"standard": FakeProvider([fake_text(envelope), fake_text("done!")])},
    )
    await rt.start()
    reply = await rt.chat("web:1", "log a run", user_id="u1")
    assert reply.text == "done!"
    await rt.stop()


# --- Runtime extensions: pending confirmations, notify fallback, widgets ----


def _blog_orc() -> Orchestrator:
    app = App("blog", description="d")

    class Post(app.Model):
        title: str

    return Orchestrator(apps=[app])


async def test_runtime_gate_binds_on_app_migrations_in_project_dir_not_data_dir(
    tmp_path: Path,
) -> None:
    """CONFIRMED finding: with auto_migrate off, Runtime.start() must refuse to
    start against an app schema that is behind head. It has to resolve app
    migrations from the PROJECT directory (where `dudamel db migrate` writes
    them) -- resolving from `settings.data_dir` when that differs from the
    project root finds no migrations/, silently hollowing the gate."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    data_dir = tmp_path / "state"
    data_dir.mkdir()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'app.db'}"

    # Core at head; an app migration exists in the PROJECT dir but is unapplied.
    upgrade_core(db_url)
    ensure_app_migrations(project_dir)
    orc = _blog_orc()
    generate_app_migration(orc, db_url, "add posts", project_dir)

    settings = Settings(
        database_url=db_url,
        data_dir=data_dir,
        project_dir=project_dir,
        auto_migrate=False,
        llm_tiers={"standard": TierConfig(provider="fake", model="f")},
    )
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([])})
    with pytest.raises(DudamelError, match="app schema is behind head"):
        await rt.start()
    await rt.stop()


async def test_runtime_auto_migrate_applies_project_dir_migrations(tmp_path: Path) -> None:
    """The other entry point: with auto_migrate on and data_dir != project_dir,
    Runtime.start() applies the app migrations found in the project directory
    (the same place the CLI reads), and never touches data_dir as a migrations
    root -- so both paths agree on where migrations live."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    data_dir = tmp_path / "state"
    data_dir.mkdir()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'app.db'}"

    upgrade_core(db_url)
    ensure_app_migrations(project_dir)
    orc = _blog_orc()
    generate_app_migration(orc, db_url, "add posts", project_dir)

    settings = Settings(
        database_url=db_url,
        data_dir=data_dir,
        project_dir=project_dir,
        auto_migrate=True,
        llm_tiers={"standard": TierConfig(provider="fake", model="f")},
    )
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([])})
    await rt.start()
    await rt.stop()

    # Applied from the project dir -> nothing pending there; data_dir never
    # became a migrations root.
    assert pending_migrations(db_url, project_dir) == []
    assert not (data_dir / "migrations").exists()


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


async def test_list_pending_confirmations_can_exclude_expired(tmp_path) -> None:
    """A confirmation past its TTL keeps status="pending" until the router
    lazily expires it. include_expired=True (the default, used by /api/pending)
    still lists it; include_expired=False (used by the dashboard chat page)
    filters it out so no dead approve/deny button is shown."""
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select

    from dudamel.models_core import PendingConfirmation

    app = App("gym", description="d")

    @app.tool(confirm=True)
    async def wipe(reason: str) -> str:
        """Delete stuff."""
        return "wiped"

    orc = Orchestrator(apps=[app])
    rt = Runtime(
        orc,
        make_settings(tmp_path),
        providers={"standard": FakeProvider([fake_tool_call("wipe", {"reason": "x"})])},
    )
    await rt.start()
    reply = await rt.chat("web:default", "wipe it", user_id="web")
    assert reply.pending_confirmation_id

    # Age the row past its TTL without touching the conversation (so the router
    # never gets a chance to flip status to "expired").
    async with rt._db.session() as s:
        row = (await s.execute(select(PendingConfirmation))).scalars().one()
        row.expires_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=1)

    assert len(await rt.list_pending_confirmations()) == 1  # default keeps it
    assert await rt.list_pending_confirmations(include_expired=False) == []  # filtered
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


# --- scheduler wiring: created-but-not-started, jobs registered -------------


async def test_runtime_has_scheduler_created_but_not_started(tmp_path) -> None:
    from dudamel.scheduler import JobScheduler

    orc = make_orc()
    rt = Runtime(orc, make_settings(tmp_path), providers={"standard": FakeProvider([])})
    assert isinstance(rt.scheduler, JobScheduler)
    assert rt.scheduler._scheduler.running is False
    await rt.start()
    assert rt.scheduler._scheduler.running is False  # rt.start() must not start it
    await rt.stop()


async def test_runtime_scheduler_registers_apps_jobs(tmp_path) -> None:
    app = App("stats", description="d")

    @app.job(interval_seconds=60)
    async def poll() -> None:
        pass

    orc = Orchestrator(apps=[app])
    rt = Runtime(orc, make_settings(tmp_path), providers={"standard": FakeProvider([])})
    assert "stats.poll" in rt.scheduler._aps_jobs
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


# --- dashboard read surfaces: chat history, activity, job runs, job list ----


async def test_recent_messages_round_trips_chat_history(tmp_path) -> None:
    orc = make_orc()
    rt = Runtime(
        orc, make_settings(tmp_path), providers={"standard": FakeProvider([fake_text("hi there")])}
    )
    await rt.start()
    await rt.chat("web:default", "hello", user_id="web")
    messages = await rt.recent_messages("web:default")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "hello"
    assert messages[1]["text"] == "hi there"
    await rt.stop()


async def test_recent_messages_empty_for_unseen_channel(tmp_path) -> None:
    rt = Runtime(make_orc(), make_settings(tmp_path), providers={"standard": FakeProvider([])})
    await rt.start()
    assert await rt.recent_messages("web:default") == []
    await rt.stop()


async def test_list_activity_reports_newest_first(tmp_path) -> None:
    orc = make_orc()
    rt = Runtime(
        orc,
        make_settings(tmp_path),
        providers={
            "standard": FakeProvider(
                [
                    fake_tool_call("log_workout", {"exercise": "squat"}),
                    fake_text("ok1"),
                    fake_tool_call("log_workout", {"exercise": "bench"}),
                    fake_text("ok2"),
                ]
            )
        },
    )
    await rt.start()
    await rt.chat("web:default", "log squat", user_id="web")
    await rt.chat("web:default", "log bench", user_id="web")
    rows = await rt.list_activity()
    assert [r["tool"] for r in rows] == ["log_workout", "log_workout"]
    assert rows[0]["args"] == {"exercise": "bench"}  # newest first
    assert rows[0]["status"] == "ok"
    await rt.stop()


async def test_list_job_runs_reports_newest_first(tmp_path) -> None:
    from dudamel.db import Database
    from dudamel.models_core import JobRun

    settings = make_settings(tmp_path)
    rt = Runtime(make_orc(), settings, providers={"standard": FakeProvider([])})
    await rt.start()
    db = Database(settings.database_url)
    async with db.session() as s:
        s.add(JobRun(job_id="gym.a", status="ok"))
    async with db.session() as s:
        s.add(JobRun(job_id="gym.b", status="error", detail="boom"))
    rows = await rt.list_job_runs()
    assert [r["job_id"] for r in rows] == ["gym.b", "gym.a"]  # newest first
    assert rows[0]["detail"] == "boom"
    await db.dispose()
    await rt.stop()


def test_list_jobs_reports_next_fire_before_scheduler_starts(tmp_path) -> None:
    app = App("gym", description="d")

    @app.job(interval_seconds=60)
    async def poll() -> None:
        pass

    orc = Orchestrator(apps=[app])
    rt = Runtime(orc, make_settings(tmp_path), providers={"standard": FakeProvider([])})
    jobs = rt.list_jobs()  # scheduler never started
    assert [j["id"] for j in jobs] == ["gym.poll"]
    assert jobs[0]["next_run_time"] is not None
