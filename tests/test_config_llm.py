from pathlib import Path

import pytest
from pydantic import ValidationError

from dudamel.config import BudgetConfig, Settings

TOML = """
[llm.tiers.standard]
provider = "openai-compatible"
base_url = "http://localhost:11434/v1"
model = "qwen3.5:9b"

[llm.tiers.deep]
provider = "anthropic"
model = "claude-sonnet-5"
api_key_env = "ANTHROPIC_API_KEY"
max_tokens = 2048

[llm.budget]
daily_tokens = 2_000_000

[router]
iteration_cap = 4
"""


def test_llm_sections_parsed(tmp_path: Path) -> None:
    (tmp_path / "dudamel.toml").write_text(TOML)
    s = Settings.load(tmp_path)
    assert s.llm_tiers["standard"].provider == "openai-compatible"
    assert s.llm_tiers["standard"].base_url == "http://localhost:11434/v1"
    assert s.llm_tiers["deep"].max_tokens == 2048
    assert s.llm_budget.daily_tokens == 2_000_000
    assert s.router.iteration_cap == 4
    assert s.router.window_tokens == 8000  # default survives partial section


def test_defaults_without_sections(tmp_path: Path) -> None:
    s = Settings.load(tmp_path)
    assert s.llm_tiers == {}
    assert s.llm_budget.daily_tokens is None
    assert s.router.iteration_cap == 8 and s.router.max_tools == 16
    assert s.router.taint_mode == "turn"


def test_tool_calling_defaults_to_native_and_is_unaffected(tmp_path: Path) -> None:
    """Every existing config keeps native tool calling with no changes."""
    (tmp_path / "dudamel.toml").write_text(TOML)
    s = Settings.load(tmp_path)
    assert s.llm_tiers["standard"].tool_calling == "native"
    assert s.llm_tiers["deep"].tool_calling == "native"


def test_tool_calling_prompted_is_parsed(tmp_path: Path) -> None:
    toml = """
[llm.tiers.local]
provider = "openai-compatible"
base_url = "http://localhost:11434/v1"
model = "qwen3.5:9b"
tool_calling = "prompted"
"""
    (tmp_path / "dudamel.toml").write_text(toml)
    s = Settings.load(tmp_path)
    assert s.llm_tiers["local"].tool_calling == "prompted"


def test_tool_calling_rejects_unknown_value(tmp_path: Path) -> None:
    toml = """
[llm.tiers.local]
provider = "openai-compatible"
base_url = "http://localhost:11434/v1"
model = "qwen3.5:9b"
tool_calling = "auto"
"""
    (tmp_path / "dudamel.toml").write_text(toml)
    with pytest.raises(ValidationError):
        Settings.load(tmp_path)


def test_existing_precedence_untouched(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "dudamel.toml").write_text('database_url = "sqlite+aiosqlite:///toml.db"\n' + TOML)
    monkeypatch.setenv("DUDAMEL_DATABASE_URL", "sqlite+aiosqlite:///env.db")
    s = Settings.load(tmp_path)
    assert s.database_url == "sqlite+aiosqlite:///env.db"
    assert s.llm_tiers["standard"].model == "qwen3.5:9b"


def test_daily_usd_raises_an_actionable_error_at_config_load() -> None:
    """A key that parses but does nothing is a silent hole in the operator's
    intended spend ceiling; rejecting it names the enforced alternative."""
    with pytest.raises(ValidationError, match="daily_usd is not enforced; use daily_tokens"):
        BudgetConfig(daily_usd=5.0)


def test_daily_usd_in_toml_surfaces_the_validation_error(tmp_path: Path) -> None:
    """Settings.load() must surface the BudgetConfig error for TOML [llm.budget]."""
    toml_with_daily_usd = """
[llm.tiers.standard]
provider = "openai-compatible"
base_url = "http://localhost:11434/v1"
model = "qwen3.5:9b"

[llm.budget]
daily_usd = 5.0
"""
    (tmp_path / "dudamel.toml").write_text(toml_with_daily_usd)
    with pytest.raises(ValidationError, match="daily_usd is not enforced; use daily_tokens"):
        Settings.load(tmp_path)
