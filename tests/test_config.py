from pathlib import Path

from dudamel.config import Settings


def test_defaults(tmp_path: Path):
    s = Settings.load(tmp_path)  # no dudamel.toml present
    assert s.database_url == "sqlite+aiosqlite:///dudamel.db"
    assert s.data_dir == Path(".")


def test_toml_overrides_defaults(tmp_path: Path):
    (tmp_path / "dudamel.toml").write_text('database_url = "sqlite+aiosqlite:///custom.db"\n')
    s = Settings.load(tmp_path)
    assert s.database_url == "sqlite+aiosqlite:///custom.db"


def test_env_overrides_toml(tmp_path: Path, monkeypatch):
    (tmp_path / "dudamel.toml").write_text('database_url = "sqlite+aiosqlite:///toml.db"\n')
    monkeypatch.setenv("DUDAMEL_DATABASE_URL", "sqlite+aiosqlite:///env.db")
    s = Settings.load(tmp_path)
    assert s.database_url == "sqlite+aiosqlite:///env.db"
