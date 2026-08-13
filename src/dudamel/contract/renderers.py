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


ACTION_LABEL_MAX = 32

# Characters removed from every action label, wherever a label is accepted.
#
# C0/C1, DEL, the line separators U+2028/U+2029 and the bidi overrides
# U+202A-U+202E / U+2066-U+2069 -- the same class a plain-text surface already
# strips, held here instead so every surface inherits it. Jinja's autoescape
# stops a label breaking OUT of its attribute and does nothing about the
# direction the remaining characters are drawn in: a label spelled
# `"Nuke"` + U+202E + `" evihcrA"` renders in a browser as "Nuke Archive",
# and the same string reaches
# `data-label`, which feeds both the confirm dialog and the aria-live
# announcement -- so the one mis-tap protection a `confirm=True` action has on
# the web would carry the spoofed reading too. Same reasoning that put the url
# allowlist here: a per-surface fix is one surface's fix.
#
# Stripped, not rejected, unlike `url` above. That rule rejects because it
# cannot clean without approving one string and storing another; this one
# stores exactly what it approved, so that objection does not apply. Labels
# are also the field most likely to be composed from synced-in text
# (`f"Archive {row.title}"`), where rejecting would turn a hostile feed into a
# dead widget rather than a readable button.
#
# Sanitizing happens BEFORE the non-empty and length checks, so a label cannot
# pass the cap and then shrink, and one spelled entirely out of overrides is
# reported as empty rather than stored as "".
_UNSAFE_LABEL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029\u202a-\u202e\u2066-\u2069]")


def clean_action_label(value: str) -> str:
    """An action label with unrenderable characters and outer whitespace gone.

    The one normalization every action label goes through, whether it arrives
    from `@app.tool(action=...)` or from a list item's per-row override. May
    return "" -- each caller raises its own error type for that.
    """
    return _UNSAFE_LABEL_CHARS.sub("", value).strip()


class ItemAction(BaseModel):
    """An action an app attaches to one list item. Written by the app."""

    tool: str
    args: dict[str, Any] = {}
    label: str | None = None  # overrides the tool's own label for this row

    @field_validator("label")
    @classmethod
    def _check_label(cls, value: str | None) -> str | None:
        """Hold a per-row override to the same rule as a registered `action=`
        label (`App._register_tool`): sanitized, stripped, non-empty, at most
        ACTION_LABEL_MAX characters.

        Without this, the override is simply a hole in that rule -- it lands in
        exactly the same place, a button, and every surface sizes that
        affordance around a short string. A plain-text surface additionally
        prints the label next to the row it acts on, where an unbounded one
        pushes the association off the end of the line.
        """
        if value is None:
            return None
        label = clean_action_label(value)
        if not label:
            raise ValueError("action label must not be empty")
        if len(label) > ACTION_LABEL_MAX:
            raise ValueError(
                f"action label must be at most {ACTION_LABEL_MAX} characters "
                f"(got {len(label)}) — it renders inside a button"
            )
        return label


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
