"""Minimal stdio MCP server fixture for Plan 4 Task 2 acceptance tests.

Exposes two tools:

- ``echo(text)`` -- annotated ``readOnlyHint=True``; the mount must map this
  to a read-only dudamel Tool.
- ``mutate(value)`` -- no annotations at all; per the Global Constraint,
  unannotated MCP tools are treated as MUTATING (taint applies).

The server's self-reported ``serverInfo.name`` (the ``{server}`` half of
``{server}__{tool}``) is ``"fixture"`` by default, overridable via either a
CLI arg (``argv[1]``) or the ``MCP_FIXTURE_NAME`` env var (CLI wins) --
tests that need two DISTINCT mounted servers (e.g. the close-order
regression test, which wants to isolate LIFO-close behavior from the
separate MCP-vs-MCP collision/dedupe policy) launch two instances of this
same file with different names instead of maintaining a second fixture.

Not meant to be run interactively: tests spawn it as
``[sys.executable, <this file>]`` (optionally plus a name arg) and speak MCP
over its stdio.
"""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

_NAME = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MCP_FIXTURE_NAME", "fixture")

mcp = FastMCP(_NAME)


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
