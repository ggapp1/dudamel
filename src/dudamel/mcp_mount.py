"""Experimental MCP (Model Context Protocol) stdio server mounting.

MCP support is EXPERIMENTAL. Everything here degrades gracefully: an
unreachable command, a server that fails to speak the protocol, a tool call
that raises or times out -- none of it may ever crash `Runtime.start()` or
take down a chat turn. The one thing that is allowed to be a hard failure is
a *name collision* with a native tool (that is a configuration bug the
operator must fix, not environmental flakiness) -- see `Registry.add_mcp_tools`.

Tool naming: each discovered tool becomes `{server}__{tool}`, both halves
sanitized to `dudamel.contract.types.TOOL_NAME_RE`
(`^[a-zA-Z0-9_-]{1,64}$`). `{server}` is the server's own self-reported
`server_info.name` from MCP `initialize`, not the launch command -- the
protocol already gives us a stable identity, so we don't need to parse argv.

`read_only_hint` is honored; a tool with no annotations at all -- the MCP
spec leaves annotations fully optional -- is treated as MUTATING. That
classification is load-bearing: once a turn has seen any mcp output, the
router confirm-gates every mutating tool it is asked to run, mcp-origin ones
included.

Note what this does NOT defend against: annotations are self-reported by the
server, so a hostile one can declare a destructive tool `read_only_hint:
true` and skip the gate. The real trust boundary is which servers the
operator chooses to mount.

Server-initiated callbacks (sampling / elicitation / roots) are refused
explicitly and immediately -- never left to the SDK default of hanging or
silently no-opping.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
import shlex
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from typing import Any

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.context import ClientRequestContext
from mcp.client.stdio import stdio_client

from dudamel.contract.types import TOOL_NAME_RE, Tool
from dudamel.exceptions import ToolValidationError

logger = logging.getLogger("dudamel.mcp")

CALL_TIMEOUT = 30.0  # per-tool-call timeout (Tool.timeout), enforced by Router
MOUNT_TIMEOUT = 15.0  # connect + list_tools budget per server, at mount time

_REFUSAL = "MCP server-initiated callbacks are not supported in dudamel v1"


# -- refused callbacks --------------------------------------------------------
# Passed explicitly (rather than relying on the SDK's built-in defaults) so
# the refusal text is ours and unambiguous. Trade-off, documented: `_build_
# capabilities` in the SDK's `ClientSession` still gates sampling/elicitation
# /roots advertisement on an identity check -- whether the callback we passed
# `is not` its own internal default sentinel -- so supplying our own refusal
# callback means `initialize()` advertises all three as nominally available,
# and a well-behaved server may still try one. The `sampling_capabilities`
# constructor argument added in this SDK generation does not change that: it
# only shapes the *content* of the `SamplingCapability` object once sampling
# is already being advertised (e.g. which sampling features it claims), it
# cannot suppress advertisement while a non-default callback is present, and
# elicitation/roots have no equivalent parameter to shape at all. There is no
# way to keep our own refusal callbacks and get a minimal handshake. That is
# safe (a server that tries anyway gets an immediate ErrorData refusal, never
# a hang); it is simply not the most minimal handshake. Good enough for an
# experimental feature.


async def _refuse_sampling(
    context: ClientRequestContext[ClientSession, Any],
    params: types.CreateMessageRequestParams,
) -> types.ErrorData:
    return types.ErrorData(code=types.INVALID_REQUEST, message=_REFUSAL)


async def _refuse_elicitation(
    context: ClientRequestContext[ClientSession, Any],
    params: types.ElicitRequestParams,
) -> types.ErrorData:
    return types.ErrorData(code=types.INVALID_REQUEST, message=_REFUSAL)


async def _refuse_list_roots(
    context: ClientRequestContext[ClientSession, Any],
) -> types.ErrorData:
    return types.ErrorData(code=types.INVALID_REQUEST, message=_REFUSAL)


# -- naming --------------------------------------------------------------------


def sanitize_mcp_name(name: str) -> str:
    """Map an arbitrary server/tool identifier onto dudamel's tool-name
    alphabet. Anything outside [a-zA-Z0-9_-] becomes '_'; an empty or
    all-invalid input becomes '_' rather than producing an empty string."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name) or "_"


def mcp_tool_name(server: str, tool: str) -> str:
    """Build the `{server}__{tool}` public name, sanitized and truncated to
    fit TOOL_NAME_RE's 64-char ceiling (truncating a valid-alphabet, non-empty
    string never produces something outside the alphabet or empty)."""
    return f"{sanitize_mcp_name(server)}__{sanitize_mcp_name(tool)}"[:64]


# -- thin schema adapter --------------------------------------------------------


class McpToolSchema:
    """Stands in for `ToolSchema` on mcp-origin tools. Deliberately NOT a
    ToolSchema-from-signature: there is no Python function signature to
    introspect, and the server -- not dudamel -- owns and validates its own
    input schema.

    `.json_schema` passes the server's `inputSchema` through unchanged (no
    forced `additionalProperties: False`, no rebuilding). `.validate()` only
    checks the JSON-RPC-boundary basics dudamel itself needs before it can
    call the tool at all: the args are a dict, and the schema's declared
    `required` keys are present. Anything deeper (types, enums, formats) is
    the server's job when it actually executes the call.
    """

    def __init__(self, input_schema: dict[str, Any]) -> None:
        self._schema = input_schema

    @property
    def json_schema(self) -> dict[str, Any]:
        # Deepcopy, not the live reference: `ToolSpec.from_tool` and friends
        # hand this straight to callers outside dudamel's control (LLM
        # provider payload construction, etc); a caller mutating the
        # returned dict must never corrupt the schema every future call to
        # this tool validates and describes itself with.
        return copy.deepcopy(self._schema)

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolValidationError("invalid arguments: expected an object")
        required = self._schema.get("required") or []
        missing = [k for k in required if k not in args]
        if missing:
            raise ToolValidationError(f"invalid arguments: missing required {missing}")
        return args


# -- tool call adapter -----------------------------------------------------------


def _make_call_fn(
    session: ClientSession, *, remote_name: str, local_name: str
) -> Callable[..., Awaitable[str]]:
    async def call(**kwargs: Any) -> str:
        result = await session.call_tool(remote_name, kwargs)
        text = "\n".join(
            block.text for block in result.content if isinstance(block, types.TextContent)
        )
        if result.is_error:
            # Raise rather than return: Router's execute path treats a raised
            # exception from tool.fn as an error tool-result (is_error=True,
            # text = "tool {name} raised RuntimeError: {detail}"). Returning
            # the text instead would produce an ok-looking result and lose
            # the server's is_error signal entirely.
            raise RuntimeError(text or f"mcp tool {local_name} reported an error with no detail")
        return text

    return call


def _build_tool(
    session: ClientSession,
    *,
    server_name: str,
    tool_name: str,
    mcp_tool: types.Tool,
    read_only: bool,
) -> Tool:
    description = f"[experimental MCP] {mcp_tool.description or mcp_tool.name}"
    input_schema = mcp_tool.input_schema or {"type": "object", "properties": {}}
    return Tool(
        name=tool_name,
        app_name=f"mcp:{server_name}",
        description=description,
        fn=_make_call_fn(session, remote_name=mcp_tool.name, local_name=tool_name),
        schema=McpToolSchema(input_schema),
        read_only=read_only,
        confirm=False,
        timeout=CALL_TIMEOUT,
        origin="mcp",
    )


# -- one mounted server ----------------------------------------------------------


class _MountedServer:
    def __init__(self, command: str, *, env_passthrough: Sequence[str] = ()) -> None:
        self.command = command
        self.session: ClientSession | None = None
        self.server_name: str = ""
        self._env_passthrough = list(env_passthrough)
        self._stack = AsyncExitStack()

    async def connect(self) -> None:
        argv = shlex.split(self.command)
        if not argv:
            raise ValueError(f"empty MCP command: {self.command!r}")
        # Env passthrough is explicit config (design doc), never ambient: the
        # SDK's own `stdio_client` already merges this dict OVER its safe
        # default environment (PATH etc.) when `env` is not None, so an
        # empty passthrough list behaves identically to omitting `env`
        # entirely -- nothing here needs to duplicate that merge.
        env = {var: os.environ[var] for var in self._env_passthrough if var in os.environ}
        params = StdioServerParameters(command=argv[0], args=argv[1:], env=env)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(
            ClientSession(
                read,
                write,
                sampling_callback=_refuse_sampling,
                elicitation_callback=_refuse_elicitation,
                list_roots_callback=_refuse_list_roots,
            )
        )
        init = await session.initialize()
        self.session = session
        self.server_name = sanitize_mcp_name(init.server_info.name)

    async def close(self) -> None:
        # Best-effort: a server that's already dead/misbehaving must not make
        # shutdown itself fail. Primary defense against cross-server
        # cancel-scope corruption is `MCPMount.close()` closing servers in
        # reverse (LIFO) mount order, matching anyio's per-task cancel-scope
        # stack discipline -- but this is defense in depth for whatever stray
        # cancel-scope misuse still slips through (this SDK's stdio transport
        # and ClientSession both open anyio task groups / cancel scopes via
        # `_stack`'s context managers). Such misuse surfaces as
        # `CancelledError`, a `BaseException` that a plain `except Exception`
        # would NOT catch -- it's shutdown plumbing, not a real cancellation,
        # so catch broadly here but still let a genuine
        # KeyboardInterrupt/SystemExit through.
        try:
            await self._stack.aclose()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.warning(
                "mcp: error closing server %r during shutdown (%s: %s) -- ignoring, best-effort",
                self.command,
                type(e).__name__,
                e,
            )


# -- per-server tool collection + MCP-vs-MCP dedupe ----------------------------


def _collect_server_tools(
    session: ClientSession,
    *,
    server_name: str,
    command: str,
    mcp_tools: Sequence[types.Tool],
    seen_names: set[str],
) -> list[Tool]:
    """Build dudamel `Tool`s from one server's `list_tools()` result,
    applying the MCP-vs-MCP collision policy inline: a tool whose
    sanitized/truncated `{server}__{tool}` name is already in `seen_names`
    -- another tool on the SAME server colliding after 64-char truncation,
    or an EARLIER server in this `mount()` call (including one spoofing an
    already-seen `serverInfo.name`) -- gets a warning and is dropped, first
    mount wins, never raised. `seen_names` is mutated in place so callers
    iterating multiple servers accumulate the claimed-name set correctly
    across calls. (Collision with a NATIVE tool is a separate, fail-loud
    check that lives in `Registry.add_mcp_tools` instead -- that's an
    operator config bug, not something an external MCP server should be
    able to trigger.)
    """
    tools: list[Tool] = []
    for mcp_tool in mcp_tools:
        name = mcp_tool_name(server_name, mcp_tool.name)
        if not TOOL_NAME_RE.match(name):
            logger.warning(
                "mcp: tool %r from server %r sanitizes to %r, which still doesn't "
                "match the tool-name pattern -- skipping this tool",
                mcp_tool.name,
                command,
                name,
            )
            continue
        if name in seen_names:
            logger.warning(
                "mcp: tool %r from server %r (serverInfo.name %r) maps to %r, which an "
                "earlier mcp tool already claimed -- dropping this one (first mount wins); "
                "this may be two long tool names colliding after 64-char truncation, or a "
                "server spoofing another's serverInfo.name",
                mcp_tool.name,
                command,
                server_name,
                name,
            )
            continue
        seen_names.add(name)
        read_only = bool(mcp_tool.annotations and mcp_tool.annotations.read_only_hint)
        tools.append(
            _build_tool(
                session,
                server_name=server_name,
                tool_name=name,
                mcp_tool=mcp_tool,
                read_only=read_only,
            )
        )
    return tools


# -- the mount --------------------------------------------------------------------


class MCPMount:
    """Spawns each configured MCP stdio server, discovers its tools, and
    hands back dudamel `Tool` objects (`origin="mcp"`) ready for
    `Registry.add_mcp_tools`. Each `commands[i]` is a shell-style command
    string (split with `shlex.split`) that launches one stdio server.

    A server that is unreachable, exits immediately, or fails the MCP
    handshake within `MOUNT_TIMEOUT` contributes zero tools and logs a
    warning -- it never raises out of `mount()`. Sibling servers, and core
    startup, are unaffected.

    A tool name colliding with another MCP-origin tool -- two tool names
    from the SAME server that sanitize/truncate to the same 64-char name,
    or two different servers (including a hostile or merely misconfigured
    server that spoofs another's `serverInfo.name`) producing the same
    `{server}__{tool}` name -- is a WARN-and-drop, first-mount-wins
    situation, never a raise: one MCP server must never be able to take
    down mounting for its siblings (or the whole process) just by
    colliding with a name, accidentally or on purpose. Colliding with a
    NATIVE tool is the one case that stays fail-loud, checked separately by
    `Registry.add_mcp_tools` -- that is an operator configuration bug, not
    something an external MCP server should be able to trigger.
    """

    def __init__(self, commands: Sequence[str], *, env_passthrough: Sequence[str] = ()) -> None:
        self._commands = list(commands)
        self._env_passthrough = list(env_passthrough)
        self._servers: list[_MountedServer] = []

    async def mount(self) -> list[Tool]:
        tools: list[Tool] = []
        seen_names: set[str] = set()
        for command in self._commands:
            server = _MountedServer(command, env_passthrough=self._env_passthrough)
            try:
                await asyncio.wait_for(server.connect(), timeout=MOUNT_TIMEOUT)
                listed = await asyncio.wait_for(server.session.list_tools(), timeout=MOUNT_TIMEOUT)  # type: ignore[union-attr]
            except Exception as e:
                logger.warning(
                    "mcp: server %r failed to mount (%s: %s) -- skipping; core unaffected",
                    command,
                    type(e).__name__,
                    e,
                )
                await server.close()
                continue
            self._servers.append(server)
            assert server.session is not None
            tools.extend(
                _collect_server_tools(
                    server.session,
                    server_name=server.server_name,
                    command=command,
                    mcp_tools=listed.tools,
                    seen_names=seen_names,
                )
            )
        return tools

    async def close(self) -> None:
        # Reverse (LIFO) mount order: anyio cancel scopes are stacked
        # per-task, in the order they were entered, across ALL mounted
        # servers (each server's own AsyncExitStack only guarantees LIFO
        # *within itself*) -- closing server 1 before server 2 tries to
        # exit a scope that isn't innermost anymore and anyio raises
        # `CancelledError` (a BaseException), which `contextlib.suppress`
        # calls used to let straight through the whole rest of shutdown.
        for server in reversed(self._servers):
            await server.close()
