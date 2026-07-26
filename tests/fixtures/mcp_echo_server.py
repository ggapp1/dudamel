"""Minimal stdio MCP server fixture for the MCP-mounting acceptance tests.

Exposes three tools:

- ``echo(text)`` -- annotated ``readOnlyHint=True``; the mount must map this
  to a read-only dudamel Tool.
- ``mutate(value)`` -- no annotations at all; dudamel treats an unannotated
  MCP tool as MUTATING by default (taint applies), matching the router's
  "unannotated == mutating" rule for anything it can't prove is safe.
- ``read_env(name)`` -- annotated ``readOnlyHint=True``; returns
  ``os.environ.get(name, "")`` as observed by the SUBPROCESS, so tests can
  prove env-passthrough config actually reaches the spawned server (and that
  omitting a variable from the passthrough list keeps it absent there).

The server's self-reported ``serverInfo.name`` (the ``{server}`` half of
``{server}__{tool}``) is ``"fixture"`` by default, overridable via either a
CLI arg (``argv[1]``) or the ``MCP_FIXTURE_NAME`` env var (CLI wins) --
tests that need two DISTINCT mounted servers (e.g. the close-order
regression test, which wants to isolate LIFO-close behavior from the
separate MCP-vs-MCP collision/dedupe policy) launch two instances of this
same file with different names instead of maintaining a second fixture.
Tests that want two servers to genuinely COLLIDE (e.g. spoofed-identity
dedupe) launch two instances with the SAME (default) name on purpose.

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


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def read_env(name: str) -> str:
    """Return this subprocess's own os.environ.get(name, "") -- proves
    env-passthrough config actually reached the spawned server."""
    return os.environ.get(name, "")


if __name__ == "__main__":
    mcp.run(transport="stdio")
