import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from dudamel.apps import SUITE_APPS, SuiteApp, missing_requirements, suite_versions_dir

# A stdlib module the test run does not already import, so find_spec has to ask
# the finders rather than short-circuit on a cached sys.modules entry -- this is
# what makes "installed" distinguishable from "already imported".
PRESENT = "wave"


def test_registry_is_empty_in_this_release() -> None:
    """This release ships the machinery, not the apps. This asserts the closed
    registry stays closed until it is deliberately opened."""
    assert SUITE_APPS == {}


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


def test_entry_is_frozen() -> None:
    entry = SuiteApp(name="tasks", module="m", summary="s")
    # replace() works; direct mutation does not -- the registry is data, and
    # nothing at runtime may rewrite it.
    assert replace(entry, summary="t").summary == "t"
    with pytest.raises(FrozenInstanceError):
        entry.summary = "t"  # type: ignore[misc]
