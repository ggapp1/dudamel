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
import hashlib
import subprocess
import time
import zipfile
from pathlib import Path

import httpx
import pytest

import dudamel
from dudamel import cli
from dudamel.config import Settings
from dudamel.llm.testing import FakeProvider
from dudamel.serve import _InstanceLock, serve

REPO_ROOT = Path(__file__).parent.parent

# Runs INSIDE the scaffolded project's resolved environment and prints the
# installed version plus a digest over every file shipped in the `dudamel`
# package -- modules AND package data (templates, static assets), since a
# release can differ in either. `__pycache__` is skipped: it is generated
# after install, so it is not part of what the wheel shipped. Kept in
# lockstep with `_wheel_payload_digest` below, which computes the same
# digest over the wheel this test built -- equal digests mean the files
# under test are the ones this checkout produced.
_ORIGIN_PROBE = """
import hashlib, importlib.metadata as md, pathlib
d = md.distribution("dudamel")
root = pathlib.Path(str(d.locate_file("dudamel")))
paths = sorted(
    p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.relative_to(root).parts
)
h = hashlib.sha256()
for p in paths:
    h.update(p.relative_to(root).as_posix().encode())
    h.update(hashlib.sha256(p.read_bytes()).digest())
print(d.version, h.hexdigest())
"""


def _wheel_payload_digest(wheel: Path) -> str:
    """The `_ORIGIN_PROBE` digest, computed over a built wheel's contents."""
    h = hashlib.sha256()
    with zipfile.ZipFile(wheel) as z:
        names = sorted(n for n in z.namelist() if n.startswith("dudamel/") and not n.endswith("/"))
        for name in names:
            h.update(name[len("dudamel/") :].encode())
            h.update(hashlib.sha256(z.read(name)).digest())
    return h.hexdigest()


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

    # The documented quickstart (README.md), minus `dudamel run` -- replaced
    # below with an in-process serve() call so a FakeProvider can stand in for
    # the model and the web port can be OS-assigned. `dudamel new` scaffolds
    # an empty app list, so the documented `workouts` example is dropped in as
    # a local app first: without one there is no model to migrate and no
    # widget to serve, and this test would prove nothing about either.
    assert cli.main(["new", str(target)]) == 0
    (target / "apps" / "workouts.py").write_bytes(
        (REPO_ROOT / "examples" / "workouts.py").read_bytes()
    )
    (target / "assistant.py").write_text(
        "from apps.workouts import app as workouts_app\n\n"
        "from dudamel import Orchestrator\n\n"
        "orchestrator = Orchestrator(apps=[workouts_app])\n"
    )
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

    # The lockfile persists across a clean shutdown (flock on a persistent
    # file is the single-instance guarantee); the flock is released, so a
    # fresh acquire against it succeeds.
    lockfile = target / ".dudamel.lock"
    assert lockfile.exists()
    reacquire = _InstanceLock(lockfile)
    reacquire.acquire()
    reacquire.release()


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
    is an ADDITIONAL source, not an override -- so `uv` resolves across the
    real index and the wheel built here, and if the index wins this test
    would validate the PUBLISHED package rather than the one being released.
    The final step below rules that out by CONTENT rather than by version:
    it hashes the shipped payload of the wheel built above and requires the
    installed distribution's files to hash identically. That is what a
    version comparison cannot do -- a working tree at the same version as
    the published release (the ordinary state between releases) resolves to
    that version either way, so a version check is simply true regardless of
    which wheel won, while the payload of an index build with local changes
    on top of it is necessarily different. The only way this assertion can
    hold on a substituted install is if the index's wheel is byte-identical
    to the one built here, in which case there is nothing to catch.

    Slow (a wheel build plus real `uv` subprocess invocations, each spinning
    up a project venv) but must always run, not be skipped, since it is the
    one test that would catch a packaging regression the in-process tests
    structurally cannot see.
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
    # A scaffolded project has no models of its own, so `db migrate` below
    # would have nothing to autogenerate; the documented example app gives it
    # a real revision to write.
    (target / "apps" / "workouts.py").write_bytes(
        (REPO_ROOT / "examples" / "workouts.py").read_bytes()
    )
    (target / "assistant.py").write_text(
        "from apps.workouts import app as workouts_app\n\n"
        "from dudamel import Orchestrator\n\n"
        "orchestrator = Orchestrator(apps=[workouts_app])\n"
    )

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
        ["uv", "run", "--find-links", str(dist_dir), "python", "-c", _ORIGIN_PROBE],
        cwd=target,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert resolved_proc.returncode == 0, resolved_proc.stderr
    # Measured, not assumed: resolving the same project WITHOUT --find-links
    # yields the identical version string and a different payload digest --
    # so the digest is what carries the signal here, and it is not weakened
    # by the version happening to match. (uv leaves no `direct_url.json` for
    # a --find-links install, so there is no installer-provided origin
    # record to read instead.)
    resolved_version, resolved_digest = resolved_proc.stdout.split()
    assert resolved_digest == _wheel_payload_digest(wheels[0]), (
        f"the scaffolded project resolved dudamel {resolved_version!r}, whose installed "
        f"files do NOT match the wheel built from this checkout ({built_version!r}) -- "
        "--find-links lost to the package index, so the release candidate was never "
        "actually exercised"
    )
