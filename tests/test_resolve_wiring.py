"""How resolved apps reach the three consumers that run them: `dudamel run`,
`dudamel db migrate`, and the startup migration gate.

The load-bearing distinction throughout: the runtime gets EVERY resolved app,
while the project's autogenerate lane gets LOCAL apps only. A suite app's
revisions ship in the wheel, so if its tables entered the user's own diff,
every user would generate a private revision for shipped code and the shipped
lane's CREATE TABLE would later collide with the table their own lane made.
"""

# No `from __future__ import annotations` here: the model classes below rely on
# real annotation objects, which the framework's column mapping reads directly.
import importlib
import sys
import textwrap
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from dudamel import App, Orchestrator, cli
from dudamel.apps import SuiteApp
from dudamel.config import Settings, TierConfig
from dudamel.exceptions import DudamelError
from dudamel.llm.testing import FakeProvider
from dudamel.migrate import generate_app_migration, upgrade_core
from dudamel.resolve import resolve_apps
from dudamel.runtime import Runtime

FULL_APP = '''
from dudamel import App

app = App("demo", description="d")


# Table demo_note -- deliberately the same table the shipped revision below
# creates, which is what makes the local/merged distinction observable.
class Note(app.Model):
    title: str


@app.tool
async def ping() -> str:
    """Ping."""
    return "pong"


@app.widget(title="Demo", renderer="stat")
async def card() -> dict:
    return {"label": "demo", "value": 1}


@app.job(cron="0 9 * * *")
async def daily() -> None:
    pass
'''

# A shipped revision for the suite app, of the kind that lives in the wheel.
SUITE_REV = '''"""demo init

Revision ID: d1
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "d1"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "demo_note",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(), nullable=False),
    )


def downgrade() -> None:
    raise NotImplementedError
'''

LOCAL_ASSISTANT = """
from dudamel import App, Orchestrator

mine = App("mine", description="d")


class Thing(mine.Model):
    label: str


orchestrator = Orchestrator(apps=[mine])
"""


@pytest.fixture(autouse=True)
def _drop_fake_suite_modules():
    yield
    for name in [n for n in sys.modules if n.split(".", 1)[0].startswith("fake_suite")]:
        del sys.modules[name]


def install_demo_suite_app(tmp_path: Path, monkeypatch, *, with_revision: bool = False) -> Path:
    """Make a `demo` suite app importable and registered, returning its
    versions directory."""
    pkg = tmp_path / "fake_suite"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").touch()
    (pkg / "demo.py").write_text(textwrap.dedent(FULL_APP))
    monkeypatch.syspath_prepend(str(tmp_path))
    # The import system caches a directory listing per sys.path entry, so a
    # package written into a directory already on the path is otherwise
    # invisible.
    importlib.invalidate_caches()
    versions = tmp_path / "demo_versions"
    versions.mkdir(exist_ok=True)
    if with_revision:
        (versions / "d1.py").write_text(SUITE_REV)
    monkeypatch.setattr(
        "dudamel.apps.SUITE_APPS",
        {
            "demo": SuiteApp(
                name="demo",
                module="fake_suite.demo",
                summary="s",
                versions_dir=versions,
            )
        },
    )
    return versions


def write_project(tmp_path: Path, toml: str) -> Path:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "assistant.py").write_text(textwrap.dedent(LOCAL_ASSISTANT))
    (project / "dudamel.toml").write_text(toml)
    return project


def table_names(db_url: str) -> set[str]:
    engine = create_engine(db_url.replace("+aiosqlite", ""))
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


# --- the autogenerate allowlist ---------------------------------------------


def test_user_lane_ignores_suite_app_tables(tmp_path: Path) -> None:
    suite = App("demo", description="d")

    class Note(suite.Model):
        title: str

    local = App("mine", description="d")

    class Thing(local.Model):
        label: str

    url = f"sqlite+aiosqlite:///{tmp_path / 'x.db'}"
    upgrade_core(url)
    # cmd_db_migrate passes ONLY the local apps here.
    path = generate_app_migration(Orchestrator(apps=[local]), url, "init", tmp_path)
    assert path is not None
    body = path.read_text()
    assert "mine_thing" in body
    assert "demo_note" not in body


def test_user_lane_with_no_local_apps_generates_nothing(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'x.db'}"
    upgrade_core(url)
    assert generate_app_migration(Orchestrator(apps=[]), url, "init", tmp_path) is None


# --- resolution reaches the running registry --------------------------------


def test_config_enabled_app_reaches_the_registry(tmp_path: Path, monkeypatch) -> None:
    """Enabled only in dudamel.toml, with an untouched assistant.py: the app's
    tool, widget and job are all live in the registry the runtime is built
    from, and its migration lane is reported for the migrator."""
    versions = install_demo_suite_app(tmp_path, monkeypatch)
    (tmp_path / "dudamel.toml").write_text("[apps.demo]\nenabled = true\n")
    settings = Settings.load(tmp_path)

    resolution = resolve_apps(Orchestrator(apps=[]), settings, strict=True)
    runtime_orc = Orchestrator(apps=resolution.apps)

    assert "ping" in runtime_orc.registry.tools
    # Widget ids are bare function names (`App.widget`); job ids are
    # app-qualified (`App.job`).
    assert [w.id for w in runtime_orc.registry.widgets] == ["card"]
    assert [j.id for j in runtime_orc.registry.jobs] == ["demo.daily"]
    assert resolution.suite_lanes == [("demo", versions)]


# --- `dudamel db migrate` ----------------------------------------------------


def test_db_migrate_excludes_suite_tables_and_applies_their_lane(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The generated project revision covers the local app only, while the
    suite app's shipped lane is what creates its table. Handing the merged app
    set to the autogenerator instead would write `demo_note` into the user's
    private revision and collide with the shipped lane."""
    install_demo_suite_app(tmp_path, monkeypatch, with_revision=True)
    project = write_project(tmp_path, "[apps.demo]\nenabled = true\n")
    monkeypatch.chdir(project)

    assert cli.main(["db", "migrate", "-m", "init"]) == 0
    generated = Path(capsys.readouterr().out.strip())
    body = generated.read_text()
    assert "mine_thing" in body
    assert "demo_note" not in body

    # Both lanes applied by the one command: the project's own and the suite's.
    assert {"mine_thing", "demo_note"} <= table_names(
        f"sqlite+aiosqlite:///{project / 'dudamel.db'}"
    )


def test_db_migrate_refuses_an_unknown_app_block(tmp_path: Path, monkeypatch) -> None:
    """`db migrate` resolves strictly, so a typo'd [apps.*] block is an error
    rather than a silently ignored section."""
    install_demo_suite_app(tmp_path, monkeypatch)
    project = write_project(tmp_path, "[apps.nope]\nenabled = true\n")
    monkeypatch.chdir(project)
    assert cli.main(["db", "migrate", "-m", "init"]) != 0


# --- `dudamel run` -----------------------------------------------------------


def test_run_hands_serve_the_merged_apps_and_the_suite_lanes(tmp_path: Path, monkeypatch) -> None:
    versions = install_demo_suite_app(tmp_path, monkeypatch)
    project = write_project(tmp_path, "[apps.demo]\nenabled = true\n")
    monkeypatch.chdir(project)
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: None)
    captured: dict[str, object] = {}

    async def fake_serve(orchestrator, settings, **kwargs):
        captured["apps"] = sorted(orchestrator.registry.apps)
        captured["tools"] = sorted(orchestrator.registry.tools)
        captured["suite_lanes"] = kwargs.get("suite_lanes")

    monkeypatch.setattr(cli, "serve", fake_serve)
    assert cli.main(["run"]) == 0
    assert captured["apps"] == ["demo", "mine"]
    assert "ping" in captured["tools"]
    assert list(captured["suite_lanes"]) == [("demo", versions)]


def test_run_refuses_an_unknown_app_block(tmp_path: Path, monkeypatch) -> None:
    install_demo_suite_app(tmp_path, monkeypatch)
    project = write_project(tmp_path, "[apps.nope]\nenabled = true\n")
    monkeypatch.chdir(project)
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: None)

    async def fake_serve(orchestrator, settings, **kwargs):  # pragma: no cover
        raise AssertionError("serve() must not be reached")

    monkeypatch.setattr(cli, "serve", fake_serve)
    assert cli.main(["run"]) != 0


# --- the startup gate --------------------------------------------------------


def gate_settings(tmp_path: Path, *, auto_migrate: bool) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'rt.db'}",
        data_dir=tmp_path,
        project_dir=tmp_path,
        auto_migrate=auto_migrate,
        llm_tiers={"standard": TierConfig(provider="fake", model="f")},
    )


def make_runtime(settings: Settings, lanes) -> Runtime:
    return Runtime(
        Orchestrator(apps=[]),
        settings,
        providers={"standard": FakeProvider([])},
        suite_lanes=lanes,
    )


async def test_startup_applies_suite_lanes_when_auto_migrating(tmp_path: Path, monkeypatch) -> None:
    """A project with no migrations/ of its own still gets its enabled suite
    apps' tables -- the lanes are not gated on the project lane existing."""
    versions = install_demo_suite_app(tmp_path, monkeypatch, with_revision=True)
    settings = gate_settings(tmp_path, auto_migrate=True)
    rt = make_runtime(settings, [("demo", versions)])
    await rt.start()
    await rt.stop()
    assert "demo_note" in table_names(settings.database_url)


async def test_startup_gate_refuses_when_a_suite_lane_is_pending(
    tmp_path: Path, monkeypatch
) -> None:
    """With auto_migrate off, an unapplied suite lane must block startup --
    otherwise the gate and `doctor` would disagree about what "up to date"
    means."""
    versions = install_demo_suite_app(tmp_path, monkeypatch, with_revision=True)
    settings = gate_settings(tmp_path, auto_migrate=False)
    upgrade_core(settings.database_url)  # core is current; only the lane is not
    rt = make_runtime(settings, [("demo", versions)])
    with pytest.raises(DudamelError) as exc:
        await rt.start()
    assert "demo" in str(exc.value)


async def test_startup_gate_passes_once_the_suite_lane_is_applied(
    tmp_path: Path, monkeypatch
) -> None:
    versions = install_demo_suite_app(tmp_path, monkeypatch, with_revision=True)
    settings = gate_settings(tmp_path, auto_migrate=True)
    rt = make_runtime(settings, [("demo", versions)])
    await rt.start()
    await rt.stop()

    settings.auto_migrate = False
    rt2 = make_runtime(settings, [("demo", versions)])
    await rt2.start()  # nothing pending -- must not raise
    await rt2.stop()


async def test_serve_forwards_suite_lanes_to_runtime(tmp_path: Path, monkeypatch) -> None:
    """`serve` is the only path from the CLI to `Runtime`, so the lanes have to
    survive that hop."""
    # Reached through sys.modules, not `import dudamel.serve as ...`: the
    # package exports the `serve` *function* under that name, which would
    # shadow the module.
    serve_mod = sys.modules["dudamel.serve"]

    captured: dict[str, object] = {}

    class StopHere(RuntimeError):
        pass

    class FakeRuntime:
        def __init__(self, orchestrator, settings, *, providers=None, suite_lanes=()) -> None:
            captured["suite_lanes"] = list(suite_lanes)
            raise StopHere

    monkeypatch.setattr(serve_mod, "Runtime", FakeRuntime)
    settings = gate_settings(tmp_path, auto_migrate=True)
    with pytest.raises(StopHere):
        await serve_mod.serve(
            Orchestrator(apps=[]), settings, suite_lanes=[("demo", tmp_path / "v")]
        )
    assert captured["suite_lanes"] == [("demo", tmp_path / "v")]
