from __future__ import annotations

from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

RENDERERS = {"stat", "table", "list", "markdown"}


class StatPayload(BaseModel):
    label: str
    value: int | float | str
    unit: str | None = None
    delta: float | None = None


class TablePayload(BaseModel):
    columns: list[str]
    rows: list[list[Any]]


class ListItem(BaseModel):
    title: str
    subtitle: str | None = None
    url: str | None = None


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
