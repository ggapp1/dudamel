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

from dudamel.mcp_mount import CALL_TIMEOUT, MOUNT_TIMEOUT


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
    # Peers whose X-Forwarded-For dudamel will believe. Empty means forwarded
    # headers are ignored entirely and the client address is always the real
    # peer. Only list a proxy you actually run: anything here can name any
    # client it likes.
    trusted_proxies: list[str] = []
    # None means auto: secure whenever the bind host is not loopback. An
    # explicit value always wins. A static False would make the insecure
    # setting the silent default in exactly the deployment that needs it,
    # and browsers treat http://localhost as a secure context, so deriving
    # this from the host is safe for local development.
    cookie_secure: bool | None = None


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
    # How long a single tool call may run, and how long connect() +
    # list_tools() may run at mount time. Defaulted from mcp_mount's own
    # module constants so there is exactly one source of each default value
    # -- see `MCPMount.__init__` for how each is actually enforced.
    call_timeout: float = CALL_TIMEOUT
    mount_timeout: float = MOUNT_TIMEOUT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DUDAMEL_", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///dudamel.db"
    data_dir: Path = Path(".")
    # When false, `Runtime.start()` refuses to start against a schema that is
    # behind its migration scripts, instead of upgrading it in place. Set this
    # in production: a process restart should never silently mutate a schema.
    auto_migrate: bool = True
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
