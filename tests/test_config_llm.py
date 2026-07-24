from pathlib import Path

from dudamel.config import Settings

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
    assert s.llm_budget.daily_usd is None
    assert s.router.iteration_cap == 4
    assert s.router.window_tokens == 8000  # default survives partial section


def test_defaults_without_sections(tmp_path: Path) -> None:
    s = Settings.load(tmp_path)
    assert s.llm_tiers == {}
    assert s.llm_budget.daily_tokens is None
    assert s.router.iteration_cap == 8 and s.router.max_tools == 16
    assert s.router.taint_mode == "turn"


def test_existing_precedence_untouched(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "dudamel.toml").write_text('database_url = "sqlite+aiosqlite:///toml.db"\n' + TOML)
    monkeypatch.setenv("DUDAMEL_DATABASE_URL", "sqlite+aiosqlite:///env.db")
    s = Settings.load(tmp_path)
    assert s.database_url == "sqlite+aiosqlite:///env.db"
    assert s.llm_tiers["standard"].model == "qwen3.5:9b"
