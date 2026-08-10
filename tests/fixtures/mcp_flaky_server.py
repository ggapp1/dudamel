"""Stdio MCP server fixture that can be killed mid-session and restarted.

The echo fixture (``mcp_echo_server.py``) never dies, so it cannot exercise
reconnect behavior at all. This one exposes four tools built specifically to
probe what happens across a real process death:

- ``die()`` -- exits the process immediately (``os._exit``), killing the
  transport mid-session so a client sees a genuine connection failure
  (broken pipe / EOF) rather than a simulated error.
- ``slow_mutate(value)`` -- records a side effect to disk (a ``started:``
  marker line a test can poll for, immediately followed by the real
  side-effect line, both synced together), then sleeps for a configurable
  delay before returning. Unannotated, so it is treated as mutating.
  Because the side effect lands before the sleep, killing the process
  during that sleep produces exactly the case a naive retry gets wrong: the
  mutation already happened, but the caller never saw a reply.
- ``count()`` -- annotated read-only; reports how many times ``slow_mutate``
  has actually completed, read back from the same on-disk record (marker
  lines don't count). Since that record lives outside the process, a
  freshly restarted server can still answer this correctly for mutations
  performed by a prior incarnation.
- ``echo(text)`` -- annotated read-only, so calling it again after a failure
  is provably safe -- there is no side effect to double up on.

The on-disk side-effect record's path comes from the ``MCP_FLAKY_STATE``
environment variable. It is deliberately never cleaned up by this file: it
has to outlive the process for a restarted server to be able to report what
an earlier incarnation did.

Two more environment variables let a test change what a *restarted* server
advertises, without changing this file:

- ``MCP_FLAKY_ANNOTATIONS=drift`` flips ``echo``'s annotation from read-only
  to mutating.
- ``MCP_FLAKY_DROP`` is a comma-separated list of tool names (currently only
  ``echo`` is droppable) that this incarnation omits from its tool list
  entirely.

Both are read once at import time, so a server restarted with different
values presents a genuinely different tool surface than the one a client
discovered before -- annotations flipped, or a tool that has vanished.

Normally spawned over stdio as ``[sys.executable, <this file>]``, the same
way ``mcp_echo_server.py`` is. Setting ``MCP_FLAKY_HTTP_PORT`` instead makes
it serve streamable HTTP on that port of ``127.0.0.1``, so the same tools can
be reached through the other transport. That mode exists for reconnect tests:
over stdio the client library respawns the server process itself, so only the
HTTP mode can exercise a connection that fails in the background while the
server is genuinely absent, and is then restored out-of-band.

Not meant to be run interactively.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

_NAME = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MCP_FIXTURE_NAME", "flaky")
_STATE = Path(os.environ.get("MCP_FLAKY_STATE", "flaky-state"))
_DRIFT = os.environ.get("MCP_FLAKY_ANNOTATIONS") == "drift"
_DROPPED = set(filter(None, os.environ.get("MCP_FLAKY_DROP", "").split(",")))
_SLOW_SECONDS = float(os.environ.get("MCP_FLAKY_SLOW_SECONDS", "0.5"))

mcp = MCPServer(_NAME)


@mcp.tool()
def die() -> str:
    """Exit immediately, killing the transport mid-session."""
    sys.stdout.flush()
    os._exit(1)


@mcp.tool()
def slow_mutate(value: str) -> str:
    """Record a side effect immediately, then sleep -- long enough for a test
    to kill this process before the reply is delivered. Unannotated, so it
    is treated as mutating.

    Writes two lines in one atomic append: a ``started:<value>`` marker,
    immediately followed by the real ``<value>`` side-effect line, both
    flushed and fsynced together before the sleep even starts. A test that
    wants to kill this process mid-call can poll the state file for the
    marker instead of guessing with a fixed sleep -- since both lines are
    written and synced as one unit, observing the marker on disk is proof
    the side-effect line landed too, not just that the handler started.
    ``count()`` below only counts the second kind of line.
    """
    with _STATE.open("a") as fh:
        fh.write(f"started:{value}\n")
        fh.write(f"{value}\n")
        fh.flush()
        os.fsync(fh.fileno())
    time.sleep(_SLOW_SECONDS)
    return f"mutated:{value}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def count() -> str:
    """How many times slow_mutate has actually completed its side effect,
    across restarts. Ignores ``started:`` marker lines -- those record that
    a call began, not that it finished, so counting them would let a
    mid-call kill look like two completed mutations instead of one."""
    if not _STATE.exists():
        return "0"
    lines = [ln for ln in _STATE.read_text().splitlines() if ln and not ln.startswith("started:")]
    return str(len(lines))


if "echo" not in _DROPPED:

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=not _DRIFT))
    def echo(text: str) -> str:
        """Echo the given text back. Read-only unless MCP_FLAKY_ANNOTATIONS=drift."""
        return text


if __name__ == "__main__":
    _HTTP_PORT = os.environ.get("MCP_FLAKY_HTTP_PORT")
    if _HTTP_PORT:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=int(_HTTP_PORT))
    else:
        mcp.run(transport="stdio")
