from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from tzlocal import get_localzone

from dudamel.mcp_mount import (
    CALL_TIMEOUT,
    MAX_RECONNECT_ATTEMPTS,
    MOUNT_TIMEOUT,
    RECONNECT_BACKOFF_SECONDS,
    RECONNECT_COOLDOWN_SECONDS,
)


class TierConfig(BaseModel):
    provider: Literal["openai-compatible", "anthropic", "fake"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 1024
    # "native": send `tools` on the wire and expect the backend's own
    # function-calling machinery (the default; every existing config keeps
    # this and is unaffected). "prompted": the backend has no native tool
    # calling -- Runtime wraps its Provider in PromptedToolsProvider, which
    # flattens tool traffic into prompt text and parses calls back out of
    # plain completions. Named `tool_calling`, not reusing `taint_mode`'s
    # vocabulary, because the two are unrelated axes of a tier's config.
    tool_calling: Literal["native", "prompted"] = "native"


class BudgetConfig(BaseModel):
    daily_tokens: int | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_daily_usd(cls, data: Any) -> Any:
        if isinstance(data, dict) and "daily_usd" in data:
            raise ValueError("daily_usd is not enforced; use daily_tokens")
        return data


class RouterConfig(BaseModel):
    iteration_cap: int = 8
    window_tokens: int = 8000
    tool_result_cap: int = 8192
    max_tools: int = 16
    confirm_ttl_seconds: int = 900
    taint_mode: Literal["turn", "window", "off"] = "turn"
    persona: str | None = None
    # Opt-in: summarize turns a window build drops so a long conversation
    # degrades to "the gist" instead of silently forgetting them (see
    # compaction.py). `compaction_tier` names one of `[llm.tiers]` and is
    # required (and validated against those tiers at startup, in
    # Runtime.__init__) only when this is true.
    compact_dropped_turns: bool = False
    compaction_tier: str | None = None


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
    # The reconnect budget, defaulted from the same module constants for the
    # same reason: how many connection attempts one burst may spend, the base
    # delay between them (it doubles each attempt), and how long a server whose
    # burst failed outright fails fast before the next call gets a fresh burst.
    reconnect_attempts: int = MAX_RECONNECT_ATTEMPTS
    reconnect_backoff_seconds: float = RECONNECT_BACKOFF_SECONDS
    reconnect_cooldown_seconds: float = RECONNECT_COOLDOWN_SECONDS

    @model_validator(mode="after")
    def reject_non_positive_settings(self) -> McpConfig:
        # Zero is not a smaller budget, it is a silent disabling of reconnect
        # entirely, and a negative backoff/cooldown is not a duration at all --
        # both are config bugs worth failing at load rather than at the first
        # dead server. The two timeouts are the same kind of bug with the same
        # remedy: a non-positive call_timeout makes every tool call fail on the
        # transport-native timeout, and a non-positive mount_timeout makes every
        # mount attempt fail instantly with an opaque TimeoutError.
        for field, value in (
            ("call_timeout", self.call_timeout),
            ("mount_timeout", self.mount_timeout),
            ("reconnect_attempts", self.reconnect_attempts),
            ("reconnect_backoff_seconds", self.reconnect_backoff_seconds),
            ("reconnect_cooldown_seconds", self.reconnect_cooldown_seconds),
        ):
            if value <= 0:
                raise ValueError(
                    f"[mcp] {field} must be positive (got {value}); "
                    "remove the setting to use the default"
                )
        return self


class HomeSection(BaseModel):
    # The one layout mistake nothing else could see. Every other quiet case
    # has a voice -- `dudamel doctor` reports an id matching no widget and a
    # widget listed twice -- but `widget = [...]` for `widgets` left a section
    # holding no widgets, which `compose_home` drops as empty: the section
    # never appeared and nothing said why. Refused at load instead, where the
    # message names the key and the file it came from.
    model_config = ConfigDict(extra="forbid")

    title: str
    widgets: list[str] = []


class HomeConfig(BaseModel):
    """Homescreen layout. `[[home.section]]` in dudamel.toml maps to `section`.

    Absent entirely, the dashboard renders exactly as it did before this
    existed: every widget, in registration order, in one grid.
    """

    model_config = ConfigDict(extra="forbid")

    section: list[HomeSection] = []


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DUDAMEL_",
        # NOTE: a misspelled top-level key is dropped rather than rejected, so
        # `timezome = "UTC"` does nothing and says nothing. Left as-is
        # deliberately -- "forbid" would reject any config carrying a key from a
        # different dudamel version. `doctor`'s timezone line is the mitigation:
        # it prints the zone that was actually resolved, so a typo shows up as
        # "from the host" next to a config that says otherwise.
        extra="ignore",
        # Lets a single per-app setting be overridden from the environment
        # (DUDAMEL_APPS__WEATHER__LATITUDE) so secrets never have to live in
        # dudamel.toml. Applies to every nested section, hence the regression
        # test covering [web] and [llm.budget].
        env_nested_delimiter="__",
    )

    database_url: str = "sqlite+aiosqlite:///dudamel.db"
    # Where dudamel keeps runtime state (currently just the single-instance
    # lockfile). Distinct from `project_dir`: state need not live in the
    # source tree.
    data_dir: Path = Path(".")
    # The project's source directory -- where `dudamel new`/`dudamel db
    # migrate` create and read `migrations/`. Set by `Settings.load` to the
    # directory it loaded from (the CWD for every CLI command), so `Runtime`
    # and the CLI resolve app migrations from ONE place even when `data_dir`
    # points elsewhere. Resolving them anywhere else would let the
    # auto_migrate startup gate miss a pending app migration.
    project_dir: Path = Path(".")
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
    home: HomeConfig = HomeConfig()
    # Raw per-app config blocks straight from [apps.*]. Values stay untyped
    # here: an app's settings model lives inside that app's module, which is
    # imported only when the app is enabled, so validation happens later
    # during resolution.
    apps: dict[str, dict[str, Any]] = {}
    # One zone for the whole framework: the scheduler's cron expressions and
    # every app's idea of "today". A top-level key, like `database_url` -- there
    # is no section wrapping these.
    #
    # `None` means the host's zone, which is what the scheduler has always
    # actually done: apscheduler builds a trigger with `get_localzone()` when
    # none is passed, so defaulting to UTC here would move every existing
    # operator's jobs without them asking. The cost is that behaviour depends on
    # /etc/localtime, so `doctor` prints the zone it resolved and where it came
    # from.
    timezone: str | None = None

    @field_validator("timezone")
    @classmethod
    def _check_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            # ValueError, not just ZoneInfoNotFoundError: zoneinfo rejects an
            # absolute or non-normalised key before it touches the filesystem,
            # and raises the other type doing it.
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value

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
        # Record where we loaded from so migrations resolve consistently with
        # the CLI (an explicit [project_dir] in the toml still wins).
        data.setdefault("project_dir", project_dir)
        return cls(_env_file=project_dir / ".env", **data)


def resolve_timezone(settings: Settings) -> ZoneInfo:
    """The framework's zone, concrete. The one place `None` becomes a zone.

    Returns `get_localzone()`'s object as-is. It is already a `ZoneInfo`, and
    round-tripping it through its own name is not merely redundant -- when
    /etc/localtime is a regular file with no name file, tzlocal returns a zone
    whose key is "local", which no `ZoneInfo(...)` lookup can reconstruct.
    """
    if settings.timezone is not None:
        return ZoneInfo(settings.timezone)
    return get_localzone()


def timezone_source(settings: Settings) -> str:
    return "config" if settings.timezone is not None else "host"
