"""Release e2e: the full "new user" path -- `dudamel new`, `dudamel db
migrate -m init`, then serve() against the scaffolded project with a
FakeProvider standing in for the model -- reaching a real socket's /health
and /api/widgets, then a clean shutdown. Proves the README's quickstart
actually works end to end, with no model involved. A second test below
shells out to a real `uv run` subprocess to prove the same for the parts
that can't be exercised in-process (packaging, `pyproject.toml`).
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time
from pathlib import Path

import httpx
import pytest

import dudamel
from dudamel import cli
from dudamel.config import Settings
from dudamel.llm.testing import FakeProvider
from dudamel.serve import serve

REPO_ROOT = Path(__file__).parent.parent


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
            assert body["version"] == dudamel.__version__

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


@pytest.mark.slow
def test_quickstart_runs_via_real_uv_run_subprocess_in_scaffolded_project(
    tmp_path: Path,
) -> None:
    """LITERAL-SHELL regression: the README's quickstart tells a new user to
    `cd` into a freshly scaffolded project and run `uv run dudamel ...` --
    that only works if the scaffold itself is a `uv`-resolvable project
    (a `pyproject.toml` declaring `dudamel` as a dependency). Everything
    above this test drives the same commands in-process, which would stay
    green even if the scaffold were missing that file entirely; this one
    shells out for real, exactly as a reader following the README would.

    The scaffold declares an unbounded `dudamel` dependency and `--find-links`
    is an ADDITIONAL source, not an override -- so `uv` resolves the highest
    version it can see across the real index and the wheel built here. When
    this checkout is behind the published version, the index wins and this
    test would validate the PUBLISHED package rather than the one being
    released. The final step below asserts the resolved version matches the
    wheel built from this checkout: that check binds -- and would catch the
    substitution -- whenever the working tree's version differs from the
    currently published one, and is inert (true either way, substitution or
    not) when the two happen to match. Slow (a wheel build plus real `uv`
    subprocess invocations, each spinning up a project venv) but must always
    run, not be skipped, since it is the one test that would catch a
    packaging regression the in-process tests structurally cannot see.
    """
    dist_dir = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "-o", str(dist_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    target = tmp_path / "my-assistant"
    assert cli.main(["new", str(target)]) == 0
    assert (target / "pyproject.toml").is_file()

    help_proc = subprocess.run(
        ["uv", "run", "--find-links", str(dist_dir), "dudamel", "--help"],
        cwd=target,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "usage: dudamel" in help_proc.stdout

    migrate_proc = subprocess.run(
        ["uv", "run", "--find-links", str(dist_dir), "dudamel", "db", "migrate", "-m", "init"],
        cwd=target,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert migrate_proc.returncode == 0, migrate_proc.stderr
    revisions = list((target / "migrations" / "versions").glob("*.py"))
    assert len(revisions) == 1, revisions

    # The wheel filename carries the version it was built from:
    # dudamel-<version>-py3-none-any.whl
    wheels = list(dist_dir.glob("dudamel-*.whl"))
    assert len(wheels) == 1, wheels
    built_version = wheels[0].name.split("-")[1]

    resolved_proc = subprocess.run(
        [
            "uv",
            "run",
            "--find-links",
            str(dist_dir),
            "python",
            "-c",
            "import dudamel; print(dudamel.__version__)",
        ],
        cwd=target,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert resolved_proc.returncode == 0, resolved_proc.stderr
    assert resolved_proc.stdout.strip() == built_version, (
        f"this test resolved dudamel {resolved_proc.stdout.strip()!r} but the wheel "
        f"under test is {built_version!r} -- --find-links lost to the package index, "
        "so the release candidate was never actually exercised"
    )
