"""Bare-annotation models in modules that carry `from __future__ import
annotations`.

That import turns every annotation into a *string*, so the column-type lookup
sees `'str'` instead of `str` and rejects a type it lists as supported. It is
the default habit across this package (every module under `src/dudamel/` has
it), so the first app model written in that style would hard-fail. This module
carries the import itself -- that is the whole point of it being separate.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pytest

from dudamel import App


def test_module_level_model_with_stringized_annotations() -> None:
    app = App("notes", description="d")

    class Note(app.Model):
        title: str
        body: str | None = None
        n: int = 0

    columns = {c.name: (str(c.type), c.nullable) for c in Note.__table__.columns}
    assert columns["title"] == ("VARCHAR", False)
    # `X | None` survives stringization: the annotation is unwrapped after
    # evaluation, so the column is still nullable.
    assert columns["body"] == ("VARCHAR", True)
    assert columns["n"] == ("INTEGER", False)


def test_mixin_annotations_are_evaluated_too() -> None:
    app = App("notes", description="d")

    class Stamped(app.Model):
        __abstract__ = True
        created_at: datetime = app.now()

    class Note(Stamped):
        title: str

    assert str(Note.__table__.columns["created_at"].type) == "DATETIME"
    assert Note.__table__.columns["created_at"].default is not None


MIXIN_MODULE = """
from __future__ import annotations

from datetime import datetime

from dudamel import App

app = App("notes", description="d")


class Stamped(app.Model):
    __abstract__ = True
    created_at: datetime = app.now()
"""

CONCRETE_MODULE = """
from mixinmod import Stamped


class Note(Stamped):
    title: str
"""


def test_mixin_resolves_against_its_own_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each ancestor's annotations are evaluated against the module that
    *declared* them, not against the concrete subclass's module. Here the two
    differ in both their imports and their use of the future import, so a
    single shared namespace taken from the subclass could not resolve
    `datetime` at all.
    """
    (tmp_path / "mixinmod.py").write_text(textwrap.dedent(MIXIN_MODULE))
    (tmp_path / "concretemod.py").write_text(textwrap.dedent(CONCRETE_MODULE))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("concretemod")
        columns = {c.name: str(c.type) for c in module.Note.__table__.columns}
        assert columns["created_at"] == "DATETIME"
        assert columns["title"] == "VARCHAR"
    finally:
        for name in ("concretemod", "mixinmod"):
            sys.modules.pop(name, None)
