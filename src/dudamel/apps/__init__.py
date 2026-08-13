"""The first-party app suite: a closed registry of apps that ship in the wheel.

`dudamel.toml`'s [apps.*] can activate only the names in `SUITE_APPS`, so
configuration never causes an arbitrary import path to be imported. Entries are
self-describing: `dudamel apps list` and the dependency preflight answer from
this metadata alone, without importing an app's module and dragging in its
optional dependencies.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from dudamel.app import APP_NAME_RE
from dudamel.exceptions import RegistryError


@dataclass(frozen=True)
class SuiteApp:
    name: str
    module: str
    summary: str
    # The pip extra that installs this app's dependencies (`dudamel[papers]`),
    # or None for a self-contained app.
    extra: str | None = None
    # Importable module names that `extra` provides. Checked -- not imported --
    # to decide whether the app can be loaded at all.
    requires: tuple[str, ...] = ()
    # Tests point this at a temp directory; real entries leave it None and get
    # the packaged location below.
    versions_dir: Path | None = None

    def __post_init__(self) -> None:
        # The same rule the `App` constructor enforces, applied here too. This
        # name is not just a label: it is interpolated into the lane's
        # bookkeeping table (`migrate.suite_version_table`) and, via the app's
        # own name, prefixes every table the app owns. Validating it here makes
        # that safety structural rather than a property of whoever last edited
        # the registry.
        if not APP_NAME_RE.match(self.name):
            raise RegistryError(
                f"suite app name {self.name!r} must start with [a-z] and contain only"
                " [a-z0-9]; it names a migration version table and prefixes table names"
            )


# Deliberately empty: the machinery ships before the apps do.
SUITE_APPS: dict[str, SuiteApp] = {}


def suite_versions_dir(entry: SuiteApp) -> Path:
    """Where this app's Alembic revisions live.

    Resolved from the `dudamel.apps` package plus pure path arithmetic, so a
    disabled or uninstallable app's lane can still be located without importing
    the app itself.
    """
    if entry.versions_dir is not None:
        return entry.versions_dir
    return Path(str(files("dudamel.apps"))) / entry.name / "migrations" / "versions"


def _is_importable(module: str) -> bool:
    """Whether `module` could be imported, without importing it.

    `find_spec` returns None for an absent top-level module, but *raises* for a
    dotted name whose parent package is absent, and for a name already in
    `sys.modules` whose `__spec__` is None. Every such failure means the same
    thing here -- the requirement is not usable -- so report it instead of
    propagating: this function exists to turn a missing dependency into a
    friendly "install the extra" message, not into a traceback.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def missing_requirements(entry: SuiteApp) -> tuple[str, ...]:
    """Declared requirements that are not importable. Uses find_spec so a
    missing dependency is detected without executing any app code."""
    return tuple(m for m in entry.requires if not _is_importable(m))
