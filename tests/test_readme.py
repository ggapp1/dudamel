"""Acceptance test: the `workouts` example embedded in README.md's
quickstart must stay byte-identical to examples/workouts.py -- the README
promises readers "this is the whole file", so any drift between the two
would make the README lie about the hero app."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _code_block_after(text: str, heading: str) -> str:
    idx = text.index(heading)
    match = re.search(r"```python\n(.*?)```", text[idx:], re.DOTALL)
    assert match is not None, f"no ```python code block found after {heading!r}"
    return match.group(1)


def test_readme_workouts_code_block_matches_examples_workouts() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    hero = (REPO_ROOT / "examples" / "workouts.py").read_text()
    embedded = _code_block_after(readme, "The whole example app")
    assert embedded == hero
