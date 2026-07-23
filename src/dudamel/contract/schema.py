from __future__ import annotations

import inspect
from typing import Any, get_type_hints

from pydantic import ConfigDict, ValidationError, create_model

from dudamel.exceptions import ToolValidationError


class ToolSchema:
    """Builds one Pydantic model per tool signature.

    The model is both the JSON schema sent to the LLM and the coercion layer
    that turns the model's string-ish output back into typed Python.
    """

    def __init__(self, fn: Any) -> None:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        fields: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            if name not in hints:
                raise TypeError(f"tool parameter {name!r} needs a type hint")
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (hints[name], default)
        self.arg_model = create_model(
            f"{fn.__name__}_args",
            __config__=ConfigDict(extra="forbid", use_enum_values=True),
            **fields,
        )
        self.description = inspect.getdoc(fn) or ""

    @property
    def json_schema(self) -> dict:
        schema = self.arg_model.model_json_schema()
        schema.pop("title", None)
        schema["additionalProperties"] = False
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        # inline enum $defs so providers see a flat schema
        defs = schema.pop("$defs", {})
        for prop in schema.get("properties", {}).values():
            self._inline_refs(prop, defs)
        return schema

    def _inline_refs(self, node: dict, defs: dict) -> None:
        if "$ref" in node:
            ref = defs[node.pop("$ref").split("/")[-1]]
            ref.pop("title", None)
            node.update(ref)
        for sub in node.get("anyOf", []):
            self._inline_refs(sub, defs)

    def validate(self, args: dict) -> dict:
        try:
            m = self.arg_model.model_validate(args)
        except ValidationError as e:
            issues = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            raise ToolValidationError(f"invalid arguments: {issues}") from e
        return {name: getattr(m, name) for name in self.arg_model.model_fields}
