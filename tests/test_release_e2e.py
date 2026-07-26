"""Release e2e (Plan 4 Task 4): the full "new user" path -- `dudamel new`,
`dudamel db migrate -m init`, then serve() against the scaffolded project
with a FakeProvider standing in for the model -- reaching a real socket's
/health and /api/widgets, then a clean shutdown. Proves the README's
quickstart actually works end to end, with no model involved.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

import httpx
import pytest

from dudamel import cli
from dudamel.config import Settings
from dudamel.llm.testing import FakeProvider
from dudamel.serve import serve


@pytest.fixture(autouse=True)
def _clean_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors test_cli.py's fixture of the same purpose: `new`/`db migrate`
    write/read a real .env, and `_load_dotenv_into_environ` loads it into the
    REAL process environment -- scrub first so a leftover host-env token
    can't shadow the one this test generates and reads back."""
    for var in ("DUDAMEL_WEB_TOKEN", "DUDAMEL_TELEGRAM_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


async def _wait_for_port(settings: Settings, *, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while settings.web.port == 0:
        if time.monotonic() >= deadline:
            raise AssertionError(f"server did not bind within {timeout}s")
        await asyncio.sleep(0.01)
    return settings.web.port


async def test_new_user_path_scaffold_migrate_serve_health_and_widgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "my-assistant"

    # Exactly the documented quickstart (README.md), minus `dudamel run` --
    # replaced below with an in-process serve() call so a FakeProvider can
    # stand in for the model and the web port can be OS-assigned.
    assert cli.main(["new", str(target)]) == 0
    monkeypatch.chdir(target)
    assert cli.main(["db", "migrate", "-m", "init"]) == 0

    cli._load_dotenv_into_environ(target)
    token = (target / ".env").read_text().strip().split("=", 1)[1]

    settings = Settings.load(target)
    settings.web.port = 0  # OS-assigned -- the scaffold's default 8787 could collide

    orchestrator = cli._load_orchestrator(target, "assistant")
    task = asyncio.create_task(
        serve(
            orchestrator,
            settings,
            providers={"standard": FakeProvider([]), "fast": FakeProvider([])},
        )
    )
    try:
        port = await _wait_for_port(settings)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            health = await c.get("/health")
            assert health.status_code == 200
            body = health.json()
            assert body["status"] == "ok"
            assert body["db"] is True

            unauthed = await c.get("/api/widgets")
            assert unauthed.status_code == 401

            widgets = await c.get("/api/widgets", headers={"Authorization": f"Bearer {token}"})
            assert widgets.status_code == 200
            payload = widgets.json()
            assert {w["id"] for w in payload} == {"week_volume"}
            # `db migrate` above created the workouts table, so the widget's
            # own query succeeds even though nothing has been logged yet.
            assert "error" not in payload[0]
            assert payload[0]["data"] == {
                "label": "Weekly volume",
                "value": 0,
                "unit": "kg",
                "delta": None,
            }
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)

    assert not (target / ".dudamel.lock").exists()
