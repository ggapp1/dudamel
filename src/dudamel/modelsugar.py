"""Bare-annotation ORM sugar backing `App.Model`.

Lets app authors write plain, type-annotated classes:

    class WorkoutSet(app.Model, table="sets"):
        exercise: str
        reps: int = 5
        logged_at: datetime = app.now()

and get a fully mapped SQLAlchemy 2.0 declarative model back: table name
prefixed with the app name, an auto `id` integer primary key when none is
declared, `X | None` annotations mapped to nullable columns, plain values
used as column defaults, and the `app.now()` sentinel mapped to an
insert-time `datetime.now(UTC)` default factory.

Convention: all datetimes are stored **naive-UTC**. SQLite has no
timezone-aware storage type, so the framework normalizes every stored
timestamp to `datetime.now(UTC).replace(tzinfo=None)` rather than pretending
tz-awareness survives a roundtrip. Treat every `datetime` column value as UTC
by convention; the framework does not attach tzinfo back on read.

The load-bearing trick lives in `Model.__init_subclass__`: bare annotations
must be rewritten into `Mapped[...]` + `mapped_column(...)` *before*
`super().__init_subclass__()` runs, because that call is what triggers
SQLAlchemy's declarative class-mapping machinery. Rewrite too late and
SQLAlchemy raises about annotations that aren't `Mapped[...]`.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from types import UnionType
from typing import Any, ClassVar, Union, get_args, get_origin

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from dudamel.exceptions import RegistryError

NOW = object()  # sentinel returned by app.now(); recognized as a default marker
_MISSING = object()

_COLUMN_TYPES: dict[type, Any] = {
    str: String,
    int: Integer,
    float: Float,
    bool: Boolean,
    datetime: DateTime,
    date: Date,
    dict: JSON,
}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _unwrap_optional(ann: object) -> tuple[object, bool]:
    """Return (inner_type, nullable) — unwraps `X | None` / `Optional[X]`."""
    if get_origin(ann) in (Union, UnionType) and type(None) in get_args(ann):
        inner = [a for a in get_args(ann) if a is not type(None)]
        if len(inner) == 1:
            return inner[0], True
    return ann, False


def make_model_base(app_name: str) -> type:
    """Build a fresh declarative base + abstract `Model` for one app.

    Each app gets its own `MetaData`/registry so table names, autogenerate
    (Task 12), and metadata introspection (Task 10) stay app-scoped.
    """

    class Base(DeclarativeBase):
        pass

    class Model(Base):
        __abstract__ = True

        def __init_subclass__(cls, table: str | None = None, **kwargs: Any) -> None:
            # Rewrite bare annotations into Mapped[...] + mapped_column BEFORE
            # calling super() — that call is what SQLAlchemy's declarative
            # machinery uses to map the class from cls.__annotations__.
            if not cls.__dict__.get("__abstract__", False):
                _rewrite_annotations(cls, app_name, table)
            super().__init_subclass__(**kwargs)

    return Model


def _rewrite_annotations(cls: type, app_name: str, table: str | None) -> None:
    cls.__tablename__ = f"{app_name}_{table or _snake(cls.__name__)}"  # type: ignore[attr-defined]

    # Only the annotations declared directly on this class (not inherited
    # ones already processed on a parent) are candidates for rewriting.
    bare = dict(cls.__dict__.get("__annotations__", {}))
    new_annotations: dict[str, object] = {}
    has_pk = "id" in bare

    if not has_pk:
        new_annotations["id"] = Mapped[int]
        cls.id = mapped_column(Integer, primary_key=True, autoincrement=True)

    for name, ann in bare.items():
        # ClassVar / non-column annotations pass through untouched.
        if get_origin(ann) is ClassVar:
            new_annotations[name] = ann
            continue

        py_type, nullable = _unwrap_optional(ann)
        sa_type = _COLUMN_TYPES.get(py_type)
        if sa_type is None:
            raise RegistryError(
                f"{cls.__name__}.{name}: unsupported column type {py_type!r}; "
                f"supported: {sorted(t.__name__ for t in _COLUMN_TYPES)}"
            )

        default = cls.__dict__.get(name, _MISSING)
        if default is NOW:
            column = mapped_column(sa_type, nullable=nullable, default=_now_naive_utc)
        elif default is not _MISSING:
            column = mapped_column(sa_type, nullable=nullable, default=default)
        else:
            column = mapped_column(sa_type, nullable=nullable)

        setattr(cls, name, column)
        new_annotations[name] = Mapped[py_type]  # type: ignore[valid-type]

    cls.__annotations__ = new_annotations


def _now_naive_utc() -> datetime:
    """Insert-time default for `app.now()` columns — naive UTC (see module docstring)."""
    return datetime.now(UTC).replace(tzinfo=None)
