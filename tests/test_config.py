from pathlib import Path

import pytest

from dudamel.config import McpConfig, Settings
from dudamel.mcp_mount import (
    CALL_TIMEOUT,
    MAX_RECONNECT_ATTEMPTS,
    MOUNT_TIMEOUT,
    RECONNECT_BACKOFF_SECONDS,
    RECONNECT_COOLDOWN_SECONDS,
)


def test_defaults(tmp_path: Path) -> None:
    s = Settings.load(tmp_path)  # no dudamel.toml present
    assert s.database_url == "sqlite+aiosqlite:///dudamel.db"
    assert s.data_dir == Path(".")


def test_mcp_timeouts_are_configurable_and_default_to_the_module_constants() -> None:
    assert McpConfig().call_timeout == CALL_TIMEOUT
    assert McpConfig().mount_timeout == MOUNT_TIMEOUT
    assert McpConfig(call_timeout=2.5, mount_timeout=5.0).call_timeout == 2.5
    assert McpConfig(call_timeout=2.5, mount_timeout=5.0).mount_timeout == 5.0


def test_mcp_reconnect_settings_default_to_the_module_constants() -> None:
    """Same rule as the two timeouts above: the module constants stay the one
    source of each default, so a default-constructed McpConfig cannot drift
    away from the behavior mcp_mount ships with."""
    assert McpConfig().reconnect_attempts == MAX_RECONNECT_ATTEMPTS
    assert McpConfig().reconnect_backoff_seconds == RECONNECT_BACKOFF_SECONDS
    assert McpConfig().reconnect_cooldown_seconds == RECONNECT_COOLDOWN_SECONDS
    tuned = McpConfig(
        reconnect_attempts=1, reconnect_backoff_seconds=0.25, reconnect_cooldown_seconds=5.0
    )
    assert tuned.reconnect_attempts == 1
    assert tuned.reconnect_backoff_seconds == 0.25
    assert tuned.reconnect_cooldown_seconds == 5.0


@pytest.mark.parametrize(
    "field",
    ["reconnect_attempts", "reconnect_backoff_seconds", "reconnect_cooldown_seconds"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_mcp_reconnect_settings_reject_non_positive_values(field: str, value: float) -> None:
    """Zero or negative is never a coherent setting here -- zero attempts
    disables reconnecting silently, and a negative cooldown/backoff is not a
    duration -- so it is rejected at config load rather than acted on."""
    with pytest.raises(ValueError, match="must be positive"):
        McpConfig(**{field: value})


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
