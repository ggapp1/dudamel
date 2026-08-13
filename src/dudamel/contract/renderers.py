from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError, field_validator

RENDERERS = {"stat", "table", "list", "markdown"}


class StatPayload(BaseModel):
    label: str
    value: int | float | str
    unit: str | None = None
    delta: float | None = None


class TablePayload(BaseModel):
    columns: list[str]
    rows: list[list[Any]]


# An explicit allowlist of URL prefixes, matched case-insensitively against
# the whole stored string.
#
# Jinja's autoescape stops a payload breaking OUT of the href attribute but
# does nothing about the scheme inside it, so `javascript:` in a list item's
# url is same-origin script execution on click. Matching a prefix rather than
# parsing a scheme is deliberate: it is short enough to audit by eye, and it
# rejects relative URLs too. A widget linking somewhere is linking off-page;
# if a relative target is ever genuinely needed it can be added here with a
# test, which is the right amount of friction for this field.
#
# Control characters are rejected outright rather than stripped before the
# prefix match. Browsers drop them before parsing a scheme, so `java\tscript:`
# has to fail -- but a validator that judges a cleaned copy and then stores
# the original approves one string and ships another, and not every surface
# that renders this url is HTML: a plain-text one escapes nothing, so an
# embedded CR/LF or NUL would reach it verbatim. Rejecting keeps the rule
# "the string we approved is the string we ship", and a URL carrying a
# control character is malformed regardless.
_SAFE_URL_PREFIXES = ("http://", "https://", "mailto:")
_URL_CONTROL_CHARS = re.compile(r"[\x00-\x20\x7f]")


class ItemAction(BaseModel):
    """An action an app attaches to one list item. Written by the app."""

    tool: str
    args: dict[str, Any] = {}
    label: str | None = None  # overrides the tool's own label for this row


class ResolvedAction(BaseModel):
    """One button, as the framework hands it to a surface.

    Produced by `dudamel.widgets`, never by an app. `args` are already
    coerced against the tool's schema and passed through `json_safe`, so a
    surface can serialize them without further thought.
    """

    tool: str
    args: dict[str, Any] = {}
    label: str
    confirm: bool = False


class ListItem(BaseModel):
    title: str
    subtitle: str | None = None
    url: str | None = None
    action: ItemAction | None = None

    @field_validator("url")
    @classmethod
    def _reject_unsafe_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if _URL_CONTROL_CHARS.search(value):
            raise ValueError(
                f"url must not contain ASCII control characters or whitespace; got {value!r}"
            )
        if not value.lower().startswith(_SAFE_URL_PREFIXES):
            raise ValueError(
                f"url must start with one of {', '.join(_SAFE_URL_PREFIXES)}; got {value!r}"
            )
        return value


_LIST_ADAPTER = TypeAdapter(list[ListItem])


def validate_widget_payload(renderer: str, data: Any) -> Any:
    try:
        if renderer == "stat":
            return StatPayload.model_validate(data)
        if renderer == "table":
            return TablePayload.model_validate(data)
        if renderer == "list":
            return _LIST_ADAPTER.validate_python(data)
        if renderer == "markdown":
            if not isinstance(data, str):
                raise ValueError("markdown renderer expects a str")
            return data
    except ValidationError as e:
        raise ValueError(f"widget payload does not match renderer {renderer!r}: {e}") from e
    raise ValueError(f"unknown renderer {renderer!r}")
