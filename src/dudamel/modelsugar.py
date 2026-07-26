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

Abstract mixins (`__abstract__ = True`) are supported: their bare
annotations/defaults are never rewritten on the mixin itself (SQLAlchemy
skips mapping abstract classes), so they stay put in `__dict__` until a
concrete subclass is defined. `_rewrite_annotations` then walks the MRO from
the most-basic abstract ancestor down to the concrete class — skipping the
framework's own `Model`/`Base` (marked with `_dudamel_root = True`) — merging
annotations and defaults so subclass declarations override mixin ones.

v1 only supports single-level concrete models: subclassing an already-mapped
(non-abstract) `Model` raises `RegistryError` — use an `__abstract__ = True`
mixin instead.

A bare `id: int` / `id: str` annotation is honored as a user-declared primary
key (autoincrement only for `int`); declaring a default on `id` raises
`RegistryError` since the primary key's value is not a "default" in the
ordinary column sense.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from types import UnionType
from typing import Any, Union, get_args, get_origin

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from dudamel.exceptions import RegistryError

NOW = object()  # sentinel returned by app.now(); recognized as a default marker
_MISSING = object()

_TABLE_OVERRIDE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_TABLENAME_LENGTH = 63  # PostgreSQL identifier limit; SQLite has none but we hold to the min

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

    Each app gets its own `MetaData`/registry so table names, alembic
    autogenerate, and metadata introspection all stay app-scoped.
    """

    class Base(DeclarativeBase):
        _dudamel_root = True

    class Model(Base):
        __abstract__ = True
        _dudamel_root = True

        def __init_subclass__(cls, table: str | None = None, **kwargs: Any) -> None:
            # Reject subclassing an already-mapped model *before* doing
            # anything else — v1 only supports single-level concrete models.
            _reject_concrete_base(cls)
            if table is not None and not _TABLE_OVERRIDE_RE.match(table):
                raise RegistryError(
                    f"{cls.__name__}: table override {table!r} must match "
                    f"{_TABLE_OVERRIDE_RE.pattern}"
                )
            # Rewrite bare annotations into Mapped[...] + mapped_column BEFORE
            # calling super() — that call is what SQLAlchemy's declarative
            # machinery uses to map the class from cls.__annotations__.
            if not cls.__dict__.get("__abstract__", False):
                _rewrite_annotations(cls, app_name, table)
            super().__init_subclass__(**kwargs)

    return Model


def _reject_concrete_base(cls: type) -> None:
    """Raise if `cls` subclasses an already-mapped (non-abstract) Model.

    A concrete model always ends up with `__tablename__` set directly on it
    by `_rewrite_annotations`; the framework's own `Model`/`Base` and any
    `__abstract__ = True` mixin never do. So any ancestor bearing
    `__tablename__` is unambiguously a concrete model being subclassed.
    """
    for base in cls.__mro__[1:]:
        if hasattr(base, "__tablename__"):
            raise RegistryError(
                f"{cls.__name__}: subclassing a concrete model is not supported; "
                "use an __abstract__ = True mixin instead"
            )


def _mro_ancestors(cls: type) -> list[type]:
    """Classes between the framework root and `cls`, most-basic-first.

    Walks `cls.__mro__` (most-derived first) in reverse, dropping everything
    up to and including the framework's own `Model`/`Base` classes (marked
    directly in their own `__dict__` with `_dudamel_root = True`). What's left
    is the chain of `__abstract__ = True` mixins the app author wrote, ending
    with `cls` itself — merging annotations/defaults in this order lets a
    subclass's own declarations override a mixin's.
    """
    ordered = list(reversed(cls.__mro__))
    ancestors: list[type] = []
    past_root = False
    for candidate in ordered:
        if "_dudamel_root" in candidate.__dict__:
            past_root = True
            continue
        if past_root:
            ancestors.append(candidate)
    return ancestors


def _mro_default(cls: type, name: str) -> object:
    """MRO-aware default lookup for `name`.

    Checks every class between the framework root and `cls` (most-basic
    mixin first) for a directly-declared value named `name`; the
    most-specific class that declares one wins, so a subclass's own default
    overrides a mixin's.
    """
    value: object = _MISSING
    for ancestor in _mro_ancestors(cls):
        if name in ancestor.__dict__:
            value = ancestor.__dict__[name]
    return value


def _rewrite_annotations(cls: type, app_name: str, table: str | None) -> None:
    tablename = f"{app_name}_{table or _snake(cls.__name__)}"
    if len(tablename) > _MAX_TABLENAME_LENGTH:
        raise RegistryError(
            f"{cls.__name__}: table name {tablename!r} is {len(tablename)} chars, "
            f"over the {_MAX_TABLENAME_LENGTH}-char SQL identifier limit; "
            "pass a shorter table= override"
        )
    cls.__tablename__ = tablename  # type: ignore[attr-defined]

    # Bare annotations come from every `__abstract__ = True` mixin between the
    # framework root and `cls`, merged most-basic-first so `cls`'s own
    # declarations (and any subclass override of a mixin field) win. Without
    # this walk, a mixin's fields (e.g. `created_at` on a `Timestamped`
    # abstract base) would silently never become columns on the concrete
    # subclass — only `cls.__dict__["__annotations__"]` was ever checked.
    bare: dict[str, object] = {}
    for ancestor in _mro_ancestors(cls):
        bare.update(ancestor.__dict__.get("__annotations__", {}))

    new_annotations: dict[str, object] = {}
    has_pk = "id" in bare

    if not has_pk:
        new_annotations["id"] = Mapped[int]
        cls.id = mapped_column(Integer, primary_key=True, autoincrement=True)

    for name, ann in bare.items():
        py_type, nullable = _unwrap_optional(ann)
        sa_type = _COLUMN_TYPES.get(py_type)
        if sa_type is None:
            raise RegistryError(
                f"{cls.__name__}.{name}: unsupported column type {py_type!r}; "
                f"supported: {sorted(t.__name__ for t in _COLUMN_TYPES)}"
            )

        # `nullable` is passed explicitly in every branch below (not just the
        # no-default case): `Mapped[py_type]` always carries the *unwrapped*
        # inner type, so SQLAlchemy's annotation-based nullability inference
        # can no longer see the original `X | None`. Skipping it on the
        # defaulted branches would silently make `x: int | None = 5` NOT NULL.
        #
        # Defaults are looked up the same MRO-aware way as the annotations
        # themselves, so a mixin's `app.now()`/plain default is inherited
        # unless the concrete class (or a more-specific mixin) overrides it.
        default = _mro_default(cls, name)

        if name == "id":
            if default is not _MISSING:
                raise RegistryError(
                    f"{cls.__name__}.id: id is the primary key; defaults are not supported"
                )
            column = mapped_column(sa_type, primary_key=True, autoincrement=py_type is int)
        elif default is NOW:
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
