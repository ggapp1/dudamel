from __future__ import annotations

import copy
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
            __config__=ConfigDict(extra="forbid", use_enum_values=True, validate_default=True),
            **fields,
        )
        self.description = inspect.getdoc(fn) or ""
        # Build (and validate) the flattened schema now, at registration time,
        # so a parameter type that can't be fully inlined fails loudly here
        # instead of silently shipping a broken $ref to a provider later at
        # request time.
        self._json_schema = self._build_json_schema()

    @property
    def json_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self._json_schema)

    def _build_json_schema(self) -> dict[str, Any]:
        schema = self.arg_model.model_json_schema()
        schema.pop("title", None)
        schema["additionalProperties"] = False
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        # Inline $defs (enums, nested models, etc.) so providers see a fully
        # flat schema. Recurse into the whole tree -- not just top-level
        # anyOf branches -- so container/nested types (e.g. list[SomeEnum])
        # don't leave a dangling $ref behind.
        defs = schema.pop("$defs", {})
        for name, prop in schema.get("properties", {}).items():
            self._inline_refs(prop, defs, frozenset(), name)
        self._assert_no_refs(schema, "schema")
        return schema

    def _inline_refs(
        self, node: Any, defs: dict[str, Any], seen: frozenset[str], field: str
    ) -> None:
        """Recursively replace every {"$ref": "#/$defs/X"} with X's definition.

        Walks dict values and list items uniformly so it covers `items`,
        `properties`, `additionalProperties`, `prefixItems`, `anyOf`, and any
        other nesting the JSON Schema draft allows, not just `anyOf`.
        """
        if isinstance(node, list):
            for item in node:
                self._inline_refs(item, defs, seen, field)
            return
        if not isinstance(node, dict):
            return
        if "$ref" in node:
            ref = node.pop("$ref")
            key = ref.rsplit("/", 1)[-1]
            if key in seen:
                raise TypeError(
                    f"tool parameter {field!r} has a circular/self-referential "
                    f"type (via {ref!r}); unsupported parameter type"
                )
            if key not in defs:
                raise TypeError(
                    f"tool parameter {field!r} references unknown schema "
                    f"{ref!r}; unsupported parameter type"
                )
            # Copy so multiple call sites referencing the same $def (e.g. two
            # parameters of the same enum type) don't end up sharing -- and
            # cross-mutating -- the same nested dict/list objects.
            resolved = copy.deepcopy(defs[key])
            resolved.pop("title", None)
            # The def's own body may itself contain $refs (nested types) --
            # keep resolving until none remain, guarded by `seen` for cycles.
            self._inline_refs(resolved, defs, seen | {key}, field)
            for k, v in resolved.items():
                node[k] = v
            return
        for value in node.values():
            self._inline_refs(value, defs, seen, field)

    def _assert_no_refs(self, node: Any, path: str) -> None:
        """Defensive check: the emitted schema must be fully flat.

        If `_inline_refs` ever misses a case, fail loudly here (at
        ToolSchema build/registration time) rather than silently sending a
        broken schema to a provider.
        """
        if isinstance(node, dict):
            if "$ref" in node:
                raise TypeError(
                    f"unresolved $ref left in emitted schema at {path!r} "
                    f"({node['$ref']!r}); unsupported parameter type"
                )
            if "$defs" in node:
                raise TypeError(
                    f"unresolved $defs left in emitted schema at {path!r}; "
                    "unsupported parameter type"
                )
            for key, value in node.items():
                self._assert_no_refs(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                self._assert_no_refs(item, f"{path}[{i}]")

    def validate(self, args: dict) -> dict[str, Any]:
        try:
            m = self.arg_model.model_validate(args)
        except ValidationError as e:
            issues = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: extra argument not accepted"
                if err["type"] == "extra_forbidden"
                else f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            raise ToolValidationError(f"invalid arguments: {issues}") from e
        return {name: getattr(m, name) for name in self.arg_model.model_fields}
