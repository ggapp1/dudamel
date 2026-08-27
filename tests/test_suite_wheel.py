"""Every shipped app's migration lane must be inside the built wheel.

A lane missing from the wheel is not a loud failure: `db migrate` finds no
revisions, reports the app up to date, and applies nothing. The working tree is
a false negative here -- the suite runs against an editable install, so the lane
is always present locally whether or not packaging ships it.

Duplicates the `uv build` in test_wheel_contents.py, which is the natural home
for this assertion; the two should be merged.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from dudamel.apps import SUITE_APPS

REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.slow
def test_wheel_contains_every_suite_lane(tmp_path: Path) -> None:
    assert SUITE_APPS, "SUITE_APPS is empty; this test would assert nothing"
    subprocess.run(
        ["uv", "build", "--wheel", "-o", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()

    for name in SUITE_APPS:
        prefix = f"dudamel/apps/{name}/migrations/versions/"
        assert any(n.startswith(prefix) and n.endswith(".py") for n in names), (
            f"{name}: no revision under {prefix} in the wheel -- "
            "`db migrate` would report it up to date and apply nothing"
        )
        assert f"dudamel/apps/{name}/__init__.py" in names, f"{name}: module missing from wheel"
