"""The README's app configuration must actually parse and validate.

A config example that no longer works is the most-copied wrong thing in a
project, and `tests/test_readme.py` only pins the workouts Python block.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def _toml_blocks() -> list[str]:
    return re.findall(r"```toml\n(.*?)```", README.read_text(), re.DOTALL)


def _block_containing(*needles: str) -> dict:
    """The first toml block containing ALL of `needles`.

    Several blocks in this README carry `[[home.section]]` -- the older ones
    illustrate layout with example apps. Matching on one substring would find
    those instead of the suite's own block.
    """
    for block in _toml_blocks():
        if all(needle in block for needle in needles):
            return tomllib.loads(block)
    raise AssertionError(f"no README toml block contains all of {needles!r}")


def test_the_apps_block_validates_against_the_real_settings_models():
    from dudamel.apps.notes import NotesSettings
    from dudamel.apps.tasks import TasksSettings
    from dudamel.resolve import _settings_values

    parsed = _block_containing("[apps.tasks]")["apps"]
    models = {"tasks": TasksSettings, "notes": NotesSettings}
    # `habits` has no settings model at all -- the local date comes from the
    # framework -- so its block is activation and nothing else. Asserting that
    # here rather than dropping it from `models` keeps the set comparison below
    # able to notice a README that documents an app the suite does not ship.
    settingless = {"habits"}

    assert set(parsed) == set(models) | settingless, "the README documents a different set of apps"
    for name, model in models.items():
        # `enabled` is consumed by activation, not by the app's own settings.
        model(**_settings_values(parsed[name]))
    for name in settingless:
        assert set(parsed[name]) == {"enabled"}


def test_the_home_layout_block_names_widgets_that_exist():
    import importlib

    from dudamel.apps import SUITE_APPS
    from dudamel.config import HomeConfig

    home = HomeConfig(
        section=_block_containing("[[home.section]]", "tasks.today")["home"]["section"]
    )

    registered = {
        f"{entry.name}.{widget_id}"
        for entry in SUITE_APPS.values()
        for widget_id in importlib.import_module(entry.module).app.widgets
    }
    documented = {widget for section in home.section for widget in section.widgets}

    assert documented <= registered, (
        f"README names widgets that do not exist: {documented - registered}"
    )
    assert documented == registered, f"README omits shipped widgets: {registered - documented}"
