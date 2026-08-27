import importlib
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from dudamel.apps import SUITE_APPS, SuiteApp, missing_requirements, suite_versions_dir
from dudamel.exceptions import RegistryError

# A stdlib module the test run does not already import, so find_spec has to ask
# the finders rather than short-circuit on a cached sys.modules entry -- this is
# what makes "installed" distinguishable from "already imported".
PRESENT = "wave"


def test_registry_is_open_only_to_apps_that_ship_in_the_wheel() -> None:
    """The registry has been deliberately opened; this is the successor to the
    emptiness assertion it replaces.

    What mattered about "empty" was never the emptiness -- it was that
    `[apps.*]` in a user's config can never name an arbitrary import path. That
    property is what is asserted here, and it keeps holding as later apps add
    entries: every module must live under `dudamel.apps.`, and every key must
    equal its own entry's name, since the key is what config addresses and the
    name is what prefixes the app's tables.
    """
    assert SUITE_APPS, "the suite registry is unexpectedly empty"
    for key, entry in SUITE_APPS.items():
        assert key == entry.name
        assert entry.module.startswith("dudamel.apps.")


def test_present_requirement_fixture_is_not_preimported() -> None:
    """Guards the premise of the tests below: if something starts importing
    `wave` first, they would stop exercising the finder."""
    assert PRESENT not in sys.modules


def test_versions_dir_defaults_under_the_apps_package() -> None:
    entry = SuiteApp(name="notes", module="dudamel.apps.notes", summary="s")
    path = suite_versions_dir(entry)
    assert path.parts[-4:] == ("apps", "notes", "migrations", "versions")


def test_versions_dir_override_wins(tmp_path: Path) -> None:
    entry = SuiteApp(name="notes", module="m", summary="s", versions_dir=tmp_path)
    assert suite_versions_dir(entry) == tmp_path


def test_missing_requirements_reports_absent_modules() -> None:
    entry = SuiteApp(
        name="papers",
        module="m",
        summary="s",
        extra="papers",
        requires=(PRESENT, "dudamel_not_a_real_module"),
    )
    assert missing_requirements(entry) == ("dudamel_not_a_real_module",)


def test_missing_requirements_empty_when_all_present() -> None:
    entry = SuiteApp(name="tasks", module="m", summary="s", requires=(PRESENT,))
    assert missing_requirements(entry) == ()


def test_missing_requirements_reports_dotted_name_with_absent_parent() -> None:
    """find_spec raises rather than returning None when the parent package is
    absent. A declared requirement like `PIL.Image` from an uninstalled extra
    must be reported, not crash the caller."""
    entry = SuiteApp(
        name="images",
        module="m",
        summary="s",
        extra="images",
        requires=("dudamel_not_a_real_module.sub",),
    )
    assert missing_requirements(entry) == ("dudamel_not_a_real_module.sub",)


def test_missing_requirements_reports_module_without_a_spec() -> None:
    """A name in sys.modules whose __spec__ is None makes find_spec raise
    ValueError. That is still just an unusable requirement."""
    name = "dudamel_specless_module"
    module = type(sys)(name)
    module.__spec__ = None
    sys.modules[name] = module
    try:
        entry = SuiteApp(name="odd", module="m", summary="s", requires=(name,))
        assert missing_requirements(entry) == (name,)
    finally:
        del sys.modules[name]


@pytest.mark.parametrize("name", ["", "1notes", "Notes", "note-s", "note_s", "a" * 33, "n;drop"])
def test_invalid_names_are_rejected(name: str) -> None:
    """A registry entry's name is interpolated into its lane's version table
    and prefixes every table the app owns, so it is validated by the same rule
    the `App` constructor applies rather than being trusted to be safe."""
    with pytest.raises(RegistryError):
        SuiteApp(name=name, module="m", summary="s")


def test_registry_entry_matches_its_app() -> None:
    """Each entry's `summary` is a COPY of the app's `description`, carried in
    the registry so `apps list` can describe an app without importing it -- and
    a copy drifts. The names must match too: the entry names the migration
    lane while the app's own name prefixes the tables inside it.

    Vacuous today -- the registry ships empty on purpose -- and that is the
    point: the guard is here before the first entry is, so whoever adds one
    inherits it instead of having to think of it.
    """
    for name, entry in SUITE_APPS.items():
        app = importlib.import_module(entry.module).app
        assert app.name == name, f"{name}: the app in {entry.module} is named {app.name!r}"
        assert entry.summary == app.description, (
            f"{name}: registry summary {entry.summary!r} has drifted from the "
            f"app's description {app.description!r}"
        )


def test_entry_is_frozen() -> None:
    entry = SuiteApp(name="tasks", module="m", summary="s")
    # replace() works; direct mutation does not -- the registry is data, and
    # nothing at runtime may rewrite it.
    assert replace(entry, summary="t").summary == "t"
    with pytest.raises(FrozenInstanceError):
        entry.summary = "t"  # type: ignore[misc]
