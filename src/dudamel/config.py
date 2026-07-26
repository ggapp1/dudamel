from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class TierConfig(BaseModel):
    provider: Literal["openai-compatible", "anthropic", "fake"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 1024


class BudgetConfig(BaseModel):
    daily_tokens: int | None = None
    daily_usd: float | None = None  # parsed; v1 enforcement is tokens-only (WARN)


class RouterConfig(BaseModel):
    iteration_cap: int = 8
    window_tokens: int = 8000
    tool_result_cap: int = 8192
    max_tools: int = 16
    confirm_ttl_seconds: int = 900
    taint_mode: Literal["turn", "window", "off"] = "turn"


class WebConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    token_env: str = "DUDAMEL_WEB_TOKEN"
    allowed_hosts: list[str] = ["localhost", "127.0.0.1"]


class TelegramConfig(BaseModel):
    token_env: str = "DUDAMEL_TELEGRAM_TOKEN"
    allowed_user_ids: list[int] = []
    allow_groups: bool = False


class McpConfig(BaseModel):
    # Env passthrough to MCP subprocesses is explicit config, never ambient:
    # only variables named here ever reach a mounted stdio server's
    # environment beyond the SDK's own safe default set (PATH etc.) --
    # see `MCPMount`/`_MountedServer` in mcp_mount.py.
    env_passthrough: list[str] = []


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DUDAMEL_", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///dudamel.db"
    data_dir: Path = Path(".")
    llm_tiers: dict[str, TierConfig] = {}
    llm_budget: BudgetConfig = BudgetConfig()
    router: RouterConfig = RouterConfig()
    web: WebConfig = WebConfig()
    telegram: TelegramConfig = TelegramConfig()
    mcp: McpConfig = McpConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # env > .env file > toml-as-init-kwargs > field defaults
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    @classmethod
    def load(cls, project_dir: Path) -> Settings:
        toml_path = project_dir / "dudamel.toml"
        data: dict[str, Any] = {}
        if toml_path.exists():
            data = tomllib.loads(toml_path.read_text())
        llm = data.pop("llm", {})
        if "tiers" in llm:
            data["llm_tiers"] = llm["tiers"]
        if "budget" in llm:
            data["llm_budget"] = llm["budget"]
        if "router" in data:
            data["router"] = data.pop("router")
        return cls(_env_file=project_dir / ".env", **data)
