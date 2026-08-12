from dataclasses import replace
from pathlib import Path

from dudamel.apps import SUITE_APPS, SuiteApp, missing_requirements, suite_versions_dir


def test_registry_is_empty_in_this_release() -> None:
    """This release ships the machinery, not the apps. This asserts the closed
    registry stays closed until it is deliberately opened."""
    assert SUITE_APPS == {}


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
        requires=("json", "dudamel_not_a_real_module"),
    )
    assert missing_requirements(entry) == ("dudamel_not_a_real_module",)


def test_missing_requirements_empty_when_all_present() -> None:
    entry = SuiteApp(name="tasks", module="m", summary="s", requires=("json",))
    assert missing_requirements(entry) == ()


def test_entry_is_frozen() -> None:
    entry = SuiteApp(name="tasks", module="m", summary="s")
    # replace() works; direct mutation does not -- the registry is data, and
    # nothing at runtime may rewrite it.
    assert replace(entry, summary="t").summary == "t"
    try:
        entry.summary = "t"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("SuiteApp must be frozen")
