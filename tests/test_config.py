from pathlib import Path

from dudamel.config import Settings


def test_defaults(tmp_path: Path) -> None:
    s = Settings.load(tmp_path)  # no dudamel.toml present
    assert s.database_url == "sqlite+aiosqlite:///dudamel.db"
    assert s.data_dir == Path(".")


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    (tmp_path / "dudamel.toml").write_text('database_url = "sqlite+aiosqlite:///custom.db"\n')
    s = Settings.load(tmp_path)
    assert s.database_url == "sqlite+aiosqlite:///custom.db"


def test_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "dudamel.toml").write_text('database_url = "sqlite+aiosqlite:///toml.db"\n')
    monkeypatch.setenv("DUDAMEL_DATABASE_URL", "sqlite+aiosqlite:///env.db")
    s = Settings.load(tmp_path)
    assert s.database_url == "sqlite+aiosqlite:///env.db"


def test_dotenv_beats_toml_and_real_env_beats_dotenv(tmp_path: Path, monkeypatch) -> None:
    """Regression test: .env file loading must be scoped to project_dir, not CWD."""
    # Create .env in tmp_path (not in CWD)
    (tmp_path / ".env").write_text("DUDAMEL_DATABASE_URL=sqlite+aiosqlite:///dotenv.db\n")
    # Create dudamel.toml in tmp_path
    (tmp_path / "dudamel.toml").write_text('database_url = "sqlite+aiosqlite:///toml.db"\n')

    # Call load() with CWD NOT being tmp_path
    # (Verify by ensuring current dir is different)
    import os

    original_cwd = os.getcwd()
    try:
        # Load from tmp_path while CWD is original (not tmp_path)
        s = Settings.load(tmp_path)
        # .env (dotenv) should beat toml
        assert s.database_url == "sqlite+aiosqlite:///dotenv.db"
    finally:
        os.chdir(original_cwd)

    # Test that a real env var beats the dotenv file
    monkeypatch.setenv("DUDAMEL_DATABASE_URL", "sqlite+aiosqlite:///env.db")
    s = Settings.load(tmp_path)
    assert s.database_url == "sqlite+aiosqlite:///env.db"
