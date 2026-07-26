"""Acceptance tests for dudamel/cli.py (Plan 4 Task 3) -- the `dudamel`
command: `new/run/db migrate/doctor/token rotate`. argparse only, actionable
errors (never a bare traceback unless --debug), every command works from a
scaffolded project directory (Global Constraints)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dudamel import cli
from dudamel.orchestrator import Orchestrator

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def _clean_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here may scaffold a project whose .env carries a real
    DUDAMEL_WEB_TOKEN, and `run`/`doctor` load that .env into the REAL
    process environment (see `cli._load_dotenv_into_environ` -- required
    because web/telegram token resolution reads `os.environ` directly, not
    a `Settings` field). `monkeypatch.delenv` here registers each var's
    CURRENT (absent-or-not) state for restoration, so whatever the CLI
    later writes into `os.environ` is undone at teardown regardless -- this
    keeps that mutation from leaking into unrelated tests later in the same
    pytest process."""
    for var in ("DUDAMEL_WEB_TOKEN", "DUDAMEL_TELEGRAM_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def scaffold(tmp_path: Path, name: str = "proj") -> Path:
    target = tmp_path / name
    assert cli.main(["new", str(target)]) == 0
    return target


# --- new -----------------------------------------------------------------


def test_scaffold_workouts_byte_identical_to_examples() -> None:
    """Global Constraints: the scaffold's bundled app must be byte-identical
    to the documented hero example."""
    scaffold_copy = REPO_ROOT / "src" / "dudamel" / "scaffold_template" / "apps" / "workouts.py"
    hero = REPO_ROOT / "examples" / "workouts.py"
    assert scaffold_copy.read_bytes() == hero.read_bytes()


def test_new_creates_expected_tree(tmp_path: Path) -> None:
    target = scaffold(tmp_path)
    for rel in (
        "assistant.py",
        "apps/__init__.py",
        "apps/workouts.py",
        "dudamel.toml",
        "pyproject.toml",
        ".env",
        ".env.example",
        ".gitignore",
        "README.md",
        "migrations/env.py",
        "migrations/script.py.mako",
        "deploy/dudamel.plist",
        "deploy/dudamel.service",
    ):
        assert (target / rel).is_file(), rel
    # migrations/versions/ exists and is committed empty (no revisions yet)
    assert (target / "migrations" / "versions").is_dir()
    assert not list((target / "migrations" / "versions").glob("*.py"))
    # a generated, non-placeholder token
    env_text = (target / ".env").read_text()
    assert env_text.startswith("DUDAMEL_WEB_TOKEN=")
    assert len(env_text.strip().split("=", 1)[1]) > 20
    # README got the project name substituted, no leftover placeholder
    readme = (target / "README.md").read_text()
    assert "{{PROJECT_NAME}}" not in readme
    assert "proj" in readme


def test_new_scaffolds_pyproject_toml_so_uv_run_works_in_project(tmp_path: Path) -> None:
    """C1 (BLOCKER): `uv run dudamel ...` inside the scaffolded project
    (the README's quickstart) requires a `pyproject.toml` declaring
    `dudamel` as a dependency -- without one, `uv run` has no project to
    resolve it into."""
    target = scaffold(tmp_path)
    pyproject = (target / "pyproject.toml").read_text()
    assert "{{PROJECT_NAME}}" not in pyproject
    assert 'name = "proj"' in pyproject
    assert 'dependencies = ["dudamel"]' in pyproject
    assert 'requires-python = ">=3.12"' in pyproject


def test_new_sanitizes_project_name_for_pyproject_toml(tmp_path: Path) -> None:
    target = scaffold(tmp_path, name="My Cool App!!")
    pyproject = (target / "pyproject.toml").read_text()
    assert 'name = "my-cool-app"' in pyproject


def test_new_writes_deploy_templates_with_project_path_substituted(tmp_path: Path) -> None:
    target = scaffold(tmp_path)
    plist = (target / "deploy" / "dudamel.plist").read_text()
    service = (target / "deploy" / "dudamel.service").read_text()
    for rendered in (plist, service):
        assert "{{PROJECT_DIR}}" not in rendered
        assert "{{PROJECT_NAME}}" not in rendered
        assert str(target.resolve()) in rendered
    assert "dudamel run" in plist
    assert "KeepAlive" in plist
    assert "Restart=always" in service
    assert "dudamel run" in service


def test_new_refuses_nonempty_target_dir(tmp_path: Path) -> None:
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "keepme.txt").write_text("hi")
    rc = cli.main(["new", str(target)])
    assert rc == 1
    assert (target / "keepme.txt").exists()
    assert not (target / "assistant.py").exists()


def test_new_refuses_existing_file_target(tmp_path: Path) -> None:
    target = tmp_path / "notadir"
    target.write_text("i am a file")
    assert cli.main(["new", str(target)]) == 1


def test_new_into_empty_existing_dir_is_allowed(tmp_path: Path) -> None:
    target = tmp_path / "empty"
    target.mkdir()
    assert cli.main(["new", str(target)]) == 0
    assert (target / "assistant.py").exists()


def test_new_creates_env_with_restricted_permissions(tmp_path: Path) -> None:
    """Regression test: .env file should have 0o600 permissions after creation."""
    target = scaffold(tmp_path)
    env_path = target / ".env"
    mode = env_path.stat().st_mode & 0o777
    assert mode == 0o600, f"Expected .env to have 0o600 permissions, got {oct(mode)}"


# --- project module discovery / import ---------------------------------------


def test_scaffolded_project_imports_and_registers_workouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = scaffold(tmp_path)
    monkeypatch.chdir(target)
    orchestrator = cli._load_orchestrator(Path.cwd(), "assistant")
    assert isinstance(orchestrator, Orchestrator)
    assert set(orchestrator.registry.tools) == {"log_workout"}
    assert [w.id for w in orchestrator.registry.widgets] == ["week_volume"]
    assert [j.id for j in orchestrator.registry.jobs] == ["workouts.evening_summary"]


def test_run_missing_module_gives_actionable_error_not_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # not a dudamel project: no assistant.py here
    rc = cli.main(["run"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "assistant.py" in err
    assert "not found" in err
    assert "Traceback" not in err


def test_run_debug_flag_reraises_instead_of_swallowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(cli.CliError, match="assistant.py"):
        cli.main(["run", "--debug"])


def test_run_wires_orchestrator_and_settings_into_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without actually starting a server: `run` must load the scaffold's
    .env into the environment, build Settings from its dudamel.toml, import
    assistant.py, and hand both to `serve()`."""
    target = scaffold(tmp_path)
    monkeypatch.chdir(target)
    captured = {}

    async def fake_serve(orchestrator, settings, **kwargs):
        captured["orchestrator"] = orchestrator
        captured["settings"] = settings

    monkeypatch.setattr(cli, "serve", fake_serve)
    assert cli.main(["run"]) == 0
    assert set(captured["orchestrator"].registry.tools) == {"log_workout"}
    assert captured["settings"].llm_tiers["standard"].provider == "openai-compatible"
    assert captured["settings"].web.port == 8787


# --- db migrate ----------------------------------------------------------


def test_db_migrate_then_no_changes_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = scaffold(tmp_path)
    monkeypatch.chdir(target)
    capsys.readouterr()  # drain `new`'s "created .../ next steps: ..." output

    assert cli.main(["db", "migrate", "-m", "init"]) == 0
    out = capsys.readouterr().out.strip()
    generated = Path(out)
    assert generated.is_file()
    assert "workouts_sets" in generated.read_text()
    assert generated.parent == target / "migrations" / "versions"

    assert cli.main(["db", "migrate", "-m", "again"]) == 0
    assert capsys.readouterr().out.strip() == "no changes"


def test_db_migrate_applies_core_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression test: after new + db migrate, core migrations are applied.
    Doctor core check should pass (alembic_version_core table exists and is at head)."""
    from sqlalchemy import create_engine, inspect

    from dudamel.migrate import sync_url

    target = scaffold(tmp_path)
    monkeypatch.chdir(target)
    capsys.readouterr()  # drain `new`'s output

    # Before db migrate, the core version table doesn't exist yet
    db_url = "sqlite+aiosqlite:///" + str(target / "dudamel.db")
    engine = create_engine(sync_url(db_url))
    insp = inspect(engine)
    assert "alembic_version_core" not in insp.get_table_names()
    engine.dispose()

    # After db migrate, core migrations should be applied
    assert cli.main(["db", "migrate", "-m", "init"]) == 0
    capsys.readouterr()  # drain

    # Now alembic_version_core should exist
    engine = create_engine(sync_url(db_url))
    insp = inspect(engine)
    assert "alembic_version_core" in insp.get_table_names()
    engine.dispose()

    # And doctor should report core migrations at head
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "✓ core migrations" in out


def test_db_migrate_requires_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = scaffold(tmp_path)
    monkeypatch.chdir(target)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["db", "migrate"])
    assert exc_info.value.code == 2  # argparse usage error


# --- doctor ------------------------------------------------------------------


def test_doctor_runs_green_on_scaffolded_project_even_fully_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`doctor` must never crash regardless of network/service state --
    endpoint checks degrade to a ✗ line, never an exception."""
    target = scaffold(tmp_path)
    monkeypatch.chdir(target)
    # Create the database so doctor can connect to it
    assert cli.main(["db", "migrate", "-m", "init"]) == 0
    capsys.readouterr()  # drain

    rc = cli.main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "database connection" in out
    assert "✓ database connection" in out  # now db exists after migrate
    assert "llm tier 'standard'" in out
    assert "llm tier 'fast'" in out
    assert "web token" in out
    assert "✓ web token" in out  # scaffold's .env sets one
    assert "telegram" in out
    assert "tailscale" in out
    # tool-safety table
    assert "log_workout" in out
    assert "read_only" in out and "confirm" in out and "origin" in out
    # no MCP servers configured in the bundled scaffold -- no note printed
    assert "MCP server(s) configured" not in out


def test_doctor_notes_mcp_servers_configured_without_mounting_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """I1: `doctor` never starts the orchestrator, so MCP tools (only
    discoverable by actually connecting -- see mcp_mount.py) can't appear in
    the tool-safety table; this surfaces that gap instead of silently
    under-reporting it."""
    target = scaffold(tmp_path)
    monkeypatch.chdir(target)
    assistant = target / "assistant.py"
    assistant.write_text(
        assistant.read_text().replace(
            "Orchestrator(apps=[workouts_app])",
            'Orchestrator(apps=[workouts_app], mcp=["true", "false"])',
        )
    )
    capsys.readouterr()  # drain `new`'s output

    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "ℹ 2 MCP server(s) configured — tools mount at run time" in out
    assert "safety flags visible then" in out


def test_doctor_reports_missing_web_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = scaffold(tmp_path)
    (target / ".env").unlink()
    (target / ".env").write_text("")
    monkeypatch.chdir(target)

    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "✗ web token" in out


def test_doctor_on_non_project_dir_reports_app_import_failure_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "✗ app import" in out


def test_doctor_in_non_project_dir_does_not_create_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression test: doctor in non-project dir should not create .db file."""
    monkeypatch.chdir(tmp_path)

    # Run doctor in empty temp dir
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out

    # Verify output mentions database not created
    assert "✗ database" in out
    assert "not created yet" in out

    # Verify no .db file was created
    db_files = list(tmp_path.glob("*.db"))
    assert len(db_files) == 0, f"doctor should not create .db file, but found: {db_files}"


# --- token rotate --------------------------------------------------------


def test_token_rotate_changes_only_that_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = scaffold(tmp_path)
    (target / ".env").write_text(
        "# a comment\nDUDAMEL_WEB_TOKEN=old-token-value\nFOO=bar\nBAZ=qux\n"
    )
    monkeypatch.chdir(target)

    assert cli.main(["token", "rotate"]) == 0
    capsys.readouterr()  # drain

    new_text = (target / ".env").read_text()
    lines = new_text.splitlines()
    assert lines[0] == "# a comment"
    assert lines[2] == "FOO=bar"
    assert lines[3] == "BAZ=qux"
    assert lines[1].startswith("DUDAMEL_WEB_TOKEN=")
    assert lines[1] != "DUDAMEL_WEB_TOKEN=old-token-value"
    new_token = lines[1].split("=", 1)[1]
    assert len(new_token) > 20


def test_token_rotate_appends_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = scaffold(tmp_path)
    (target / ".env").write_text("FOO=bar\n")
    monkeypatch.chdir(target)

    assert cli.main(["token", "rotate"]) == 0
    lines = (target / ".env").read_text().splitlines()
    assert lines[0] == "FOO=bar"
    assert lines[1].startswith("DUDAMEL_WEB_TOKEN=")


def test_token_rotate_preserves_env_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: .env file should have 0o600 permissions after rotation."""
    target = scaffold(tmp_path)
    monkeypatch.chdir(target)

    # Rotate the token
    assert cli.main(["token", "rotate"]) == 0

    # Check that permissions are still 0o600
    env_path = target / ".env"
    mode = env_path.stat().st_mode & 0o777
    assert mode == 0o600, f"Expected .env to have 0o600 permissions after rotate, got {oct(mode)}"


def test_token_rotate_without_env_file_is_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here at all
    rc = cli.main(["token", "rotate"])
    assert rc == 1
    err = capsys.readouterr().err
    assert ".env" in err
    assert "Traceback" not in err


# --- entry point / --debug ---------------------------------------------------


def test_cli_entry_point_resolves() -> None:
    result = subprocess.run(
        ["uv", "run", "dudamel", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "usage: dudamel" in result.stdout


def test_debug_flag_accepted_after_every_leaf_subcommand() -> None:
    parser = cli._build_parser()
    assert parser.parse_args(["doctor", "--debug"]).debug is True
    assert parser.parse_args(["run", "--debug"]).debug is True
    assert parser.parse_args(["new", "x", "--debug"]).debug is True
    assert parser.parse_args(["db", "migrate", "-m", "x", "--debug"]).debug is True
    assert parser.parse_args(["token", "rotate", "--debug"]).debug is True


def test_no_debug_flag_defaults_false() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["doctor"])
    assert args.debug is False
