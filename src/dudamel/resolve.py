"""Turns `Settings.apps` plus the project's Orchestrator into the set of apps
that will actually run.

Three stages, because a settings model lives inside its app's module and an app
module is imported only when enabled:

  1. no imports  -- unknown names, reserved-name collisions, `enabled`,
                    and the declared-dependency preflight
  2. import      -- enabled suite apps only
  3. settings    -- validated against each imported app's own model

`strict=True` (run, db migrate) raises on the first failure. `strict=False`
(doctor, apps list) collects every failure and skips the offending app, so a
diagnostic command can describe a broken configuration instead of dying on it.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from dudamel import apps as suite
from dudamel.app import App
from dudamel.apps import SuiteApp, missing_requirements, suite_versions_dir
from dudamel.config import Settings
from dudamel.exceptions import AppResolutionError, AppSettingsError
from dudamel.orchestrator import Orchestrator


@dataclass(frozen=True)
class AppError:
    app: str | None
    stage: int
    message: str


@dataclass
class Resolution:
    # Everything that will run: suite apps plus surviving local apps.
    apps: list[App] = field(default_factory=list)
    # Local apps only -- the user's autogenerate lane must diff against these
    # and never against shipped tables, whose revisions ship in the wheel.
    local_apps: list[App] = field(default_factory=list)
    # (app name, versions dir) for each enabled suite app, in apply order.
    suite_lanes: list[tuple[str, Path]] = field(default_factory=list)
    errors: list[AppError] = field(default_factory=list)


def _is_enabled(blocks: dict[str, dict[str, Any]], name: str) -> bool:
    """Presence means enabled: an app with no [apps.<name>] block at all is off.
    `enabled` defaults to true only WITHIN a block that exists."""
    if name not in blocks:
        return False
    return bool(blocks[name].get("enabled", True))


def _settings_values(block: dict[str, Any]) -> dict[str, Any]:
    """The block minus `enabled`, which is the resolver's switch and not one of
    the app's own settings. `bind_settings` rejects unknown keys, so leaving it
    in would make every configured app fail validation."""
    return {k: v for k, v in block.items() if k != "enabled"}


def _import_fresh(entry: SuiteApp) -> ModuleType:
    """Import a suite module, discarding any previously imported copy.

    Suite modules expose a module-global `App`, and settings are bound onto it
    during resolution. Reusing a cached module would let a second resolution
    silently reconfigure an app the first one is still running, so each
    resolution gets its own instance.

    Only the app module and its own submodules are purged. Its parent packages
    stay cached deliberately: for a real entry those are `dudamel` and
    `dudamel.apps`, and re-executing them would mint a second `App` class that
    nothing else in the process would recognise.
    """
    prefix = entry.module + "."
    for name in [n for n in sys.modules if n == entry.module or n.startswith(prefix)]:
        del sys.modules[name]
    return importlib.import_module(entry.module)


def resolve_apps(orchestrator: Orchestrator, settings: Settings, *, strict: bool) -> Resolution:
    resolution = Resolution()
    blocks = dict(settings.apps)
    local_by_name = dict(orchestrator.registry.apps)
    # Read through the module rather than binding the dict at import time: the
    # registry is module state, and tests (and any future in-process override)
    # replace the attribute itself.
    suite_apps = suite.SUITE_APPS

    def fail(app: str | None, stage: int, message: str) -> None:
        if strict:
            raise AppResolutionError(message)
        resolution.errors.append(AppError(app=app, stage=stage, message=message))

    # --- stage 1: no imports ------------------------------------------------
    for name in sorted(local_by_name):
        if name in suite_apps:
            fail(
                name,
                1,
                f"local app {name!r} uses a name reserved by the first-party suite; "
                "rename your app (its tables are prefixed with its name, so this "
                "would also collide in the database)",
            )
    for name in sorted(blocks):
        if name not in suite_apps and name not in local_by_name:
            known = ", ".join(sorted(suite_apps)) or "(none)"
            fail(name, 1, f"[apps.{name}] names no known app; suite apps: {known}")

    # Suite apps only. Checked for every block, enabled or not: a disabled app's
    # import is skipped, so nothing downstream would ever look at its settings --
    # and this key used to define the day boundary that existing rows are already
    # keyed to. Accepting it and ignoring it would re-key those rows' meaning in
    # silence. A local app may still declare a `timezone` setting of its own;
    # the name is not reserved, only this key on these apps.
    for name in sorted(set(blocks) & set(suite_apps)):
        if "timezone" in blocks[name]:
            fail(
                name,
                1,
                f"[apps.{name}] timezone was removed; set a top-level "
                'timezone = "..." in dudamel.toml instead. That key is also the '
                "scheduler's zone, so if it differs from this app's old value "
                "your cron jobs will move by the difference — and any rows this "
                "app already keyed to a local date keep their old boundary",
            )

    enabled_suite: list[SuiteApp] = []
    for name, entry in sorted(suite_apps.items()):
        if not _is_enabled(blocks, name):
            continue
        if name in local_by_name:
            continue  # already reported above; do not compound it
        missing = missing_requirements(entry)
        if missing:
            fail(
                name,
                1,
                f"app {name!r} requires {', '.join(missing)}: "
                f"pip install dudamel[{entry.extra or name}]",
            )
            continue
        enabled_suite.append(entry)

    # --- stage 2: import enabled suite apps ---------------------------------
    imported: list[tuple[SuiteApp, App]] = []
    for entry in enabled_suite:
        try:
            module = _import_fresh(entry)
        # The app's own import problem, reported as-is. `SystemExit` is named
        # alongside `Exception` because it is not one: a module that calls
        # `sys.exit()` at import would otherwise unwind straight through
        # diagnostic mode and take `doctor` down with it, which is the one
        # thing diagnostic mode promises not to do. `KeyboardInterrupt` is
        # deliberately still allowed through -- that one is the operator
        # talking, not the app.
        except (Exception, SystemExit) as e:
            fail(entry.name, 2, f"app {entry.name!r} failed to import: {e!r}")
            continue
        app = getattr(module, "app", None)
        if not isinstance(app, App):
            fail(
                entry.name,
                2,
                f"suite app {entry.name!r}: {entry.module} does not define a module-level `app`",
            )
            continue
        if app.name != entry.name:
            # The registry name and the app's own name address different
            # things -- the lane's bookkeeping table is `entry.name`'s, while
            # the data tables are prefixed with `app.name` -- so a divergence
            # would silently split one app's schema across two identities.
            # It also drives the diagnostic commands: `apps list` keys its
            # resolved set on `app.name` and its rows on `entry.name`, and
            # would print `state=error` beside a lane that is in fact running.
            fail(
                entry.name,
                2,
                f"suite app {entry.name!r}: {entry.module} defines an app named "
                f"{app.name!r}; the registry entry and the app must agree "
                "(the name prefixes the app's tables and names its migration lane)",
            )
            continue
        imported.append((entry, app))

    # --- stage 3: settings --------------------------------------------------
    for entry, app in imported:
        try:
            app.bind_settings(_settings_values(blocks.get(entry.name, {})))
        except AppSettingsError as e:
            fail(entry.name, 3, str(e))
            continue
        resolution.apps.append(app)
        resolution.suite_lanes.append((entry.name, suite_versions_dir(entry)))

    for name in sorted(local_by_name):
        if name in suite_apps:
            continue
        block = blocks.get(name, {})
        # Opposite default from suite apps, deliberately: a local app is
        # registered in Python, so it runs unless config switches it off.
        if not block.get("enabled", True):
            continue
        app = local_by_name[name]
        try:
            app.bind_settings(_settings_values(block))
        except AppSettingsError as e:
            fail(name, 3, str(e))
            continue
        resolution.apps.append(app)
        resolution.local_apps.append(app)

    return resolution
