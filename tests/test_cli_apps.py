"""`dudamel apps list` and `doctor`'s app-resolution section.

Both commands are diagnostic: they describe a configuration that may be
broken, so every failure is a reported line and never an abort. Registry
entries are described from metadata alone -- a disabled or uninstallable app
is listed in full without importing its module.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from dudamel.apps import SuiteApp
from dudamel.cli import main
from dudamel.migrate import ensure_app_migrations, upgrade_all, upgrade_apps, upgrade_core

SUITE_MODULE = """
from dudamel import App

app = App("papers", description="arXiv digests")


@app.tool
async def read_papers() -> str:
    \"\"\"Read today's digest.\"\"\"
    return "ok"
"""

SUITE_REV = '''"""papers init

Revision ID: p1
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "p1"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("papers_paper", sa.Column("id", sa.Integer(), primary_key=True))


def downgrade() -> None:
    raise NotImplementedError
'''

PROJECT_REV = '''"""blog init

Revision ID: a1
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "a1"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("blog_post", sa.Column("id", sa.Integer(), primary_key=True))


def downgrade() -> None:
    raise NotImplementedError
'''

ASSISTANT = """
from dudamel import Orchestrator

orchestrator = Orchestrator(apps=[])
"""


@pytest.fixture(autouse=True)
def _drop_fake_suite_modules():
    yield
    for name in [n for n in sys.modules if n.split(".", 1)[0].startswith("fake_suite")]:
        del sys.modules[name]


def register(monkeypatch: pytest.MonkeyPatch, *entries: SuiteApp) -> None:
    # Patched on the module, never rebound at import: the resolver reads
    # `dudamel.apps.SUITE_APPS` through the module for exactly this reason.
    monkeypatch.setattr("dudamel.apps.SUITE_APPS", {e.name: e for e in entries})


def write_project(tmp_path: Path, toml: str, *, assistant: str = ASSISTANT) -> Path:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "assistant.py").write_text(textwrap.dedent(assistant))
    (project / "dudamel.toml").write_text(toml)
    return project


def install_papers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, revision: bool) -> Path:
    """Make an importable `papers` suite app, returning its versions dir."""
    pkg = tmp_path / "fake_suite"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").touch()
    (pkg / "papers.py").write_text(textwrap.dedent(SUITE_MODULE))
    monkeypatch.syspath_prepend(str(tmp_path))
    # The import system caches a directory listing per sys.path entry, so a
    # package written into a directory already on the path is otherwise
    # invisible.
    importlib.invalidate_caches()
    versions = tmp_path / "papers_versions"
    versions.mkdir(exist_ok=True)
    if revision:
        (versions / "p1.py").write_text(SUITE_REV)
    register(
        monkeypatch,
        SuiteApp(
            name="papers",
            module="fake_suite.papers",
            summary="arXiv digests",
            extra="papers",
            versions_dir=versions,
        ),
    )
    return versions


def db_url_for(project: Path) -> str:
    return f"sqlite+aiosqlite:///{project / 'dudamel.db'}"


# --- apps list ---------------------------------------------------------------


def test_apps_list_describes_disabled_app_without_importing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A disabled app is described from registry metadata alone -- its module
    is never imported, which is what makes lazy loading worth having."""
    register(
        monkeypatch,
        SuiteApp(
            name="papers",
            module="dudamel_module_that_would_explode",
            summary="arXiv digests",
            extra="papers",
            requires=("dudamel_absent_dep",),
        ),
    )
    project = write_project(tmp_path, "[apps.papers]\nenabled = false\n")
    monkeypatch.chdir(project)
    assert main(["apps", "list"]) == 0
    out = capsys.readouterr().out
    assert "papers" in out
    assert "arXiv digests" in out
    assert "disabled" in out
    assert "dudamel_module_that_would_explode" not in sys.modules


def test_apps_list_reports_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    register(
        monkeypatch,
        SuiteApp(
            name="papers",
            module="m",
            summary="s",
            extra="papers",
            requires=("dudamel_absent_dep",),
        ),
    )
    project = write_project(tmp_path, "[apps.papers]\nenabled = true\n")
    monkeypatch.chdir(project)
    assert main(["apps", "list"]) == 0
    out = capsys.readouterr().out
    assert "dudamel[papers]" in out
    assert "error" in out


def test_apps_list_shows_no_lane_status_for_a_disabled_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A disabled app's lane is never consulted, so its lane column is a dash
    rather than a claim about a database it was never compared against."""
    versions = install_papers(tmp_path, monkeypatch, revision=True)
    project = write_project(tmp_path, "[apps.papers]\nenabled = false\n")
    upgrade_core(db_url_for(project))
    upgrade_all(db_url_for(project), project, [("papers", versions)])
    monkeypatch.chdir(project)
    assert main(["apps", "list"]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if "papers" in line)
    assert "disabled" in row
    assert "—" in row
    assert "at head" not in row


def test_apps_list_reports_a_pending_lane_then_at_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    versions = install_papers(tmp_path, monkeypatch, revision=True)
    project = write_project(tmp_path, "[apps.papers]\nenabled = true\n")
    url = db_url_for(project)
    upgrade_core(url)  # core is current; only the app lane is not
    monkeypatch.chdir(project)

    assert main(["apps", "list"]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if "papers" in line)
    assert "enabled" in row
    assert "pending" in row

    upgrade_all(url, project, [("papers", versions)])
    assert main(["apps", "list"]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if "papers" in line)
    assert "at head" in row


def test_apps_list_includes_local_apps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    register(monkeypatch)
    project = write_project(
        tmp_path,
        "",
        assistant="""
        from dudamel import App, Orchestrator

        mine = App("mine", description="my own app")

        orchestrator = Orchestrator(apps=[mine])
        """,
    )
    monkeypatch.chdir(project)
    assert main(["apps", "list"]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if "mine" in line)
    assert "local" in row
    assert "my own app" in row
    assert "enabled" in row
    # The lane column is a status for every row, local ones included; with no
    # database on disk yet that status is "no db", never a bare dash (which
    # means "not consulted").
    assert "no db" in row
    assert "—" not in row


def test_apps_list_reports_the_shared_lane_for_local_apps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A local app's tables live in the project's shared migrations/ lane, so
    its lane column is that lane's state -- the same status vocabulary the
    suite rows use, from the same comparison the startup gate makes."""
    register(monkeypatch)
    project = write_project(
        tmp_path,
        "",
        assistant="""
        from dudamel import App, Orchestrator

        mine = App("mine", description="my own app")
        off = App("off", description="switched off")

        orchestrator = Orchestrator(apps=[mine, off])
        """,
    )
    (project / "dudamel.toml").write_text("[apps.off]\nenabled = false\n")
    url = db_url_for(project)
    upgrade_core(url)
    ensure_app_migrations(project)
    (project / "migrations" / "versions" / "a1.py").write_text(PROJECT_REV)
    monkeypatch.chdir(project)

    assert main(["apps", "list"]) == 0
    rows = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[2:]}
    assert "pending" in rows["mine"]
    # Switched off: nothing was consulted on its behalf, so no status is
    # claimed for it either.
    assert "disabled" in rows["off"]
    assert "—" in rows["off"]

    upgrade_apps(url, project)
    assert main(["apps", "list"]) == 0
    rows = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[2:]}
    assert "at head" in rows["mine"]


def test_apps_list_survives_a_missing_project_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No assistant.py at all is still a describable configuration."""
    register(monkeypatch, SuiteApp(name="papers", module="m", summary="arXiv digests"))
    (tmp_path / "dudamel.toml").write_text("")
    monkeypatch.chdir(tmp_path)
    assert main(["apps", "list"]) == 0
    assert "arXiv digests" in capsys.readouterr().out


# --- doctor ------------------------------------------------------------------


def test_doctor_reports_app_errors_without_dying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A broken [apps.*] block must not stop doctor reaching its other checks."""
    register(monkeypatch)
    project = write_project(tmp_path, "[apps.ghost]\nenabled = true\n")
    monkeypatch.chdir(project)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "ghost" in out
    assert "app resolution" in out
    assert "database connection" in out  # unrelated checks still ran


def test_doctor_on_clean_scaffold_reports_zero_apps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    register(monkeypatch)
    project = write_project(tmp_path, "")
    monkeypatch.chdir(project)
    assert main(["doctor"]) == 0
    assert "app resolution: 0 enabled, 0 error(s)" in capsys.readouterr().out


def test_doctor_does_not_advise_migrate_for_a_suite_only_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An enabled suite app has models, but `db migrate` autogenerates against
    LOCAL apps only -- a suite app's revisions ship in the wheel. So a project
    whose only app comes from the suite has nothing to generate, and doctor
    must not send it to a command that would print `no changes`."""
    install_papers(tmp_path, monkeypatch, revision=True)
    project = write_project(tmp_path, "[apps.papers]\nenabled = true\n")
    ensure_app_migrations(project)  # migrations/ present, versions/ empty
    monkeypatch.chdir(project)

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "app resolution: 1 enabled, 0 error(s)" in out
    assert "app migrations dir: present, no revisions yet — normal until an app defines" in out
    assert "run `dudamel db migrate -m init`" not in out


def test_doctor_sees_a_pending_suite_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The startup gate refuses to start on a pending suite lane, so doctor
    has to report the same lane rather than a clean bill of health."""
    versions = install_papers(tmp_path, monkeypatch, revision=True)
    project = write_project(tmp_path, "[apps.papers]\nenabled = true\n")
    url = db_url_for(project)
    upgrade_core(url)  # core is current; only the app lane is not
    monkeypatch.chdir(project)

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "pending migrations" in out
    assert "app 'papers' schema is behind head" in out

    upgrade_all(url, project, [("papers", versions)])
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "papers' schema is behind head" not in out
    assert "✓ pending migrations" in out


def test_doctor_survives_an_assistant_that_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The project's own code raising at import is reported, not fatal: the
    checks that have nothing to do with assistant.py must still run."""
    register(monkeypatch)
    project = write_project(tmp_path, "", assistant="raise ValueError('boom')\n")
    monkeypatch.chdir(project)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "boom" in out
    assert "database connection" in out
    assert "app resolution: 0 enabled, 0 error(s)" in out


def test_doctor_reads_the_project_lane_from_settings_project_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit `project_dir` in dudamel.toml wins, and the runtime resolves
    the project's own migration lane from exactly that. Reading the cwd instead
    would let doctor report a green schema on a project whose unapplied lane
    the startup gate then refuses to start on."""
    register(monkeypatch)
    project = write_project(tmp_path, 'project_dir = "sub"\n')
    sub = project / "sub"
    sub.mkdir()
    url = db_url_for(project)
    upgrade_core(url)  # core is current; only the project lane is not
    ensure_app_migrations(sub)
    (sub / "migrations" / "versions" / "a1.py").write_text(PROJECT_REV)
    assert not (project / "migrations").exists()
    monkeypatch.chdir(project)

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "✗ pending migrations: app schema is behind head" in out
    assert "✓ app migrations dir: present (1 revision)" in out

    upgrade_apps(url, sub)
    assert main(["doctor"]) == 0
    assert "✓ pending migrations" in capsys.readouterr().out


def test_db_migrate_writes_the_lane_doctor_asks_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The remedy doctor prints has to be one that satisfies it. With an
    explicit `project_dir`, `db migrate` must create the lane where doctor and
    the startup gate read it -- otherwise the operator loops forever on a
    command that cannot close the check it was told to close."""
    register(monkeypatch)
    project = write_project(
        tmp_path,
        'project_dir = "sub"\n',
        assistant="""
        from dudamel import App, Orchestrator

        blog = App("blog", description="d")

        orchestrator = Orchestrator(apps=[blog])
        """,
    )
    (project / "sub").mkdir()
    monkeypatch.chdir(project)

    assert main(["doctor"]) == 0
    assert "✗ app migrations dir: migrations/ not found" in capsys.readouterr().out

    assert main(["db", "migrate", "-m", "init"]) == 0
    capsys.readouterr()
    assert (project / "sub" / "migrations").exists()
    assert not (project / "migrations").exists()

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "✓ app migrations dir: present" in out
    assert "✓ pending migrations" in out


def test_doctor_tool_table_covers_apps_enabled_only_in_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tool table is what an operator reads to decide what may run
    unconfirmed, and doctor announces the app two lines above it: a suite app
    enabled purely in dudamel.toml is not in the project's own registry, so a
    table built from that registry would omit its tools."""
    install_papers(tmp_path, monkeypatch, revision=False)
    project = write_project(tmp_path, "[apps.papers]\nenabled = true\n")
    monkeypatch.chdir(project)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "app resolution: 1 enabled" in out
    assert "read_papers" in out


def test_doctor_reports_a_cross_app_tool_collision_instead_of_dying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two apps declaring the same tool name each resolve cleanly -- the guard
    lives in Registry, which only sees them together. `dudamel run` refuses
    such a project, which is exactly when an operator reaches for doctor, so
    the collision has to be a reported line and not the end of the report."""
    install_papers(tmp_path, monkeypatch, revision=False)
    project = write_project(
        tmp_path,
        "[apps.papers]\nenabled = true\n",
        assistant="""
        from dudamel import App, Orchestrator

        blog = App("blog", description="d")


        @blog.tool
        async def read_papers() -> str:
            \"\"\"Collides with the suite app's tool.\"\"\"
            return "no"


        orchestrator = Orchestrator(apps=[blog])
        """,
    )
    monkeypatch.chdir(project)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "app resolution: 2 enabled" in out
    assert "✗ tool table" in out
    assert "read_papers" in out
    assert "cookie_secure" in out  # the rest of the report still printed


def test_doctor_does_not_create_a_missing_sqlite_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Diagnosing a project must not be what creates its database."""
    register(monkeypatch)
    project = write_project(tmp_path, "")
    monkeypatch.chdir(project)
    assert main(["doctor"]) == 0
    assert "pending migrations" in capsys.readouterr().out
    assert not (project / "dudamel.db").exists()
