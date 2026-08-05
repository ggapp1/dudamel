"""The package version is declared in two places -- `pyproject.toml` (what the
build backend stamps into distribution metadata, and what the release workflow
compares the tag against) and `src/dudamel/_version.py` (what `GET /health`
reports). Nothing in the toolchain connects them, so this module is the
connection: bump one without the other and the suite fails, in CI and again
inside the release workflow before it publishes anything.

No third literal lives here on purpose -- a hardcoded expected version would
have to be bumped too, and would silently pass while the other two disagreed.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import version as metadata_version
from pathlib import Path

import pytest

import dudamel

# __file__-anchored, never cwd-relative: parts of this suite chdir into a
# scaffolded project whose own pyproject.toml also carries a `version`, and a
# cwd-relative lookup would read that one instead and quietly compare the
# package against the scaffold.
REPO_ROOT = Path(__file__).parent.parent


def test_version_is_a_string() -> None:
    assert isinstance(dudamel.__version__, str)
    assert dudamel.__version__


def test_module_version_matches_installed_distribution_metadata() -> None:
    """`_version.py` is what /health reports; distribution metadata is what
    pyproject.toml stamped at build time. A stale editable install shows up
    here first."""
    assert dudamel.__version__ == metadata_version("dudamel"), (
        "installed distribution metadata disagrees with dudamel.__version__ -- "
        "if you just bumped the version, run `uv sync`"
    )


def test_module_version_matches_pyproject_literal() -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not running from a source checkout")
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert dudamel.__version__ == declared, (
        f"pyproject.toml declares {declared!r} but src/dudamel/_version.py "
        f"declares {dudamel.__version__!r} -- bump both"
    )
