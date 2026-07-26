"""Minimal stdio MCP server fixture for Plan 4 Task 2 acceptance tests.

Exposes two tools:

- ``echo(text)`` -- annotated ``readOnlyHint=True``; the mount must map this
  to a read-only dudamel Tool.
- ``mutate(value)`` -- no annotations at all; per the Global Constraint,
  unannotated MCP tools are treated as MUTATING (taint applies).

Not meant to be run interactively: tests spawn it as
``[sys.executable, <this file>]`` and speak MCP over its stdio.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("fixture")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def echo(text: str) -> str:
    """Echo the given text back."""
    return text


@mcp.tool()
def mutate(value: str) -> str:
    """Pretend to mutate state somewhere (unannotated -- treated as mutating)."""
    return f"mutated:{value}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
