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
from dudamel.migrate import upgrade_all, upgrade_core

SUITE_MODULE = """
from dudamel import App

app = App("papers", description="arXiv digests")
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
