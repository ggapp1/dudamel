"""Experimental MCP (Model Context Protocol) server mounting.

Two transports are supported, chosen per server by `MCPServerConfig`: stdio
(a subprocess command) and streamable HTTP (a URL, optionally with headers
for auth). A plain string in `Orchestrator(mcp=[...])` is still shorthand for
a stdio command -- `MCPServerConfig` is only needed for HTTP or for
per-server environment passthrough.

MCP support is EXPERIMENTAL. Everything here degrades gracefully: an
unreachable command or URL, a server that fails to speak the protocol, a tool
call that raises or times out -- none of it may ever crash `Runtime.start()`
or take down a chat turn. The one thing that is allowed to be a hard failure is
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

`destructive_hint` is honored too, but only in its EXPLICIT-true form: a
tool whose server sets `destructive_hint: true` becomes `Tool(confirm=True)`,
so the router stops and asks before running it. An absent or false
`destructive_hint` does NOT force confirm. This is a deliberate departure
from the MCP spec, which defaults `destructive_hint` to true when the
annotation is missing -- applying that default here would confirm-prompt
every unannotated mcp tool and make the whole feature unusable. The
compensating control for the tools this leaves un-prompted is the
`read_only_hint` taint gate described above: any mutating mcp tool (which is
most of them, since unannotated defaults to mutating) is already
confirm-gated once a turn has seen mcp output, regardless of
`destructive_hint`.

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
import json
import logging
import os
import re
import shlex
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.context import ClientRequestContext
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from dudamel.contract.types import TOOL_NAME_RE, Tool
from dudamel.exceptions import ToolValidationError

logger = logging.getLogger("dudamel.mcp")

CALL_TIMEOUT = 30.0  # per-tool-call timeout (Tool.timeout), enforced by Router
MOUNT_TIMEOUT = 15.0  # connect + list_tools budget per server, at mount time

# An MCP server's input_schema and description are embedded verbatim in every
# LLM request, so both are a token-budget cost AND a prompt-injection channel
# that fires at advertise time -- before any tool call, and therefore with no
# taint at all. 16 KiB is ~4-5k tokens against a default window_tokens of
# 8000, so a schema that wants a higher cap is already unusable here.
MAX_SCHEMA_BYTES = 16384
MAX_DESCRIPTION_CHARS = 1024

_REFUSAL = "MCP server-initiated callbacks are not supported in dudamel v1"


# -- server configuration -----------------------------------------------------


@dataclass(frozen=True)
class MCPServerConfig:
    """One configured MCP server: either a stdio `command` or an HTTP `url`.

    `headers` exists because `streamable_http_client` accepts no headers and
    no auth of its own -- a bare URL can never reach an authenticated server,
    which is most real remote ones. The token rides here and reaches the
    transport through mcp's own `create_mcp_http_client(headers=...)`, which
    keeps the SDK's HTTP stack an implementation detail rather than a
    dependency dudamel declares.

    `env` names environment variables passed through to a stdio server's
    subprocess -- per-server, replacing the single global passthrough list
    that `MCPMount(env_passthrough=...)` applies to plain string entries.
    """

    command: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    env: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if bool(self.command) == bool(self.url):
            raise ValueError("MCPServerConfig needs exactly one of `command` or `url`")

    @property
    def label(self) -> str:
        """Stable identifier for log messages, whichever transport this is."""
        return self.command or self.url or "<unconfigured>"


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
    server: _MountedServer, *, remote_name: str, local_name: str
) -> Callable[..., Awaitable[str]]:
    async def call(**kwargs: Any) -> str:
        # Resolved per call, never captured: Registry stores these Tool
        # objects permanently, so a session rebound by reconnect has to reach
        # tools that were built against the previous one.
        session = server.session
        if session is None:
            raise RuntimeError(f"mcp tool {local_name} has no live session")
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
    server: _MountedServer,
    *,
    server_name: str,
    tool_name: str,
    mcp_tool: types.Tool,
    read_only: bool,
    confirm: bool,
) -> Tool:
    raw_description = mcp_tool.description or mcp_tool.name
    # Truncate, not drop: a truncated description is still a description.
    if len(raw_description) > MAX_DESCRIPTION_CHARS:
        raw_description = raw_description[:MAX_DESCRIPTION_CHARS] + "..."
    description = f"[experimental MCP] {raw_description}"
    input_schema = mcp_tool.input_schema or {"type": "object", "properties": {}}
    return Tool(
        name=tool_name,
        app_name=f"mcp:{server_name}",
        description=description,
        fn=_make_call_fn(server, remote_name=mcp_tool.name, local_name=tool_name),
        schema=McpToolSchema(input_schema),
        read_only=read_only,
        confirm=confirm,
        timeout=CALL_TIMEOUT,
        origin="mcp",
    )


# -- one mounted server ----------------------------------------------------------


class _MountedServer:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.session: ClientSession | None = None
        self.server_name: str = ""
        self._stack = AsyncExitStack()

    async def _open_transport(self) -> tuple[Any, Any]:
        if self.config.url is not None:
            # mcp 2.0's streamable_http_client yields a 2-tuple (read,
            # write); 1.x (where the same function lives under the same
            # name) yields a 3-tuple with a get_session_id callable too.
            # Entering the context manager performs no network I/O -- it's
            # lazy, the connection attempt happens on the first real
            # operation inside connect() (initialize(), below), so
            # MOUNT_TIMEOUT wrapping connect() already covers it; no
            # separate timeout is needed here.
            #
            # The client is entered into `self._stack` here, not just
            # constructed: `streamable_http_client` only closes the
            # httpx2.AsyncClient it manages when IT created one internally
            # (`http_client=None`). Since we always pass one in explicitly,
            # streamable_http_client never closes it -- entering it into our
            # own stack, before the transport, is what makes it get closed
            # at all, and LIFO order tears the transport down first.
            http_client = await self._stack.enter_async_context(
                create_mcp_http_client(headers=self.config.headers)
            )
            read, write = await self._stack.enter_async_context(
                streamable_http_client(self.config.url, http_client=http_client)
            )
            return read, write
        argv = shlex.split(self.config.command or "")
        if not argv:
            raise ValueError(f"empty MCP command: {self.config.command!r}")
        # Env passthrough is explicit config, never ambient: the SDK's own
        # `stdio_client` already merges this dict OVER its safe default
        # environment (PATH etc.) when `env` is not None, so an empty
        # passthrough list behaves identically to omitting `env` entirely --
        # nothing here needs to duplicate that merge.
        env = {var: os.environ[var] for var in self.config.env if var in os.environ}
        params = StdioServerParameters(command=argv[0], args=argv[1:], env=env)
        return await self._stack.enter_async_context(stdio_client(params))

    async def connect(self) -> None:
        read, write = await self._open_transport()
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
                self.config.label,
                type(e).__name__,
                e,
            )


# -- per-server tool collection + MCP-vs-MCP dedupe ----------------------------


def _collect_server_tools(
    server: _MountedServer,
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
        raw_schema = mcp_tool.input_schema or {"type": "object", "properties": {}}
        try:
            schema_bytes = len(json.dumps(raw_schema).encode("utf-8"))
        except (RecursionError, ValueError, TypeError) as e:
            logger.warning(
                "mcp: tool %r from server %r has an input schema that cannot be "
                "serialized (%s) -- dropping this tool; the mount continues",
                mcp_tool.name,
                command,
                type(e).__name__,
            )
            continue
        if schema_bytes > MAX_SCHEMA_BYTES:
            # Drop, not truncate: a truncated JSON Schema is not a schema.
            logger.warning(
                "mcp: tool %r from server %r has a %d-byte input schema, over the "
                "%d-byte cap -- dropping this tool; the mount continues",
                mcp_tool.name,
                command,
                schema_bytes,
                MAX_SCHEMA_BYTES,
            )
            continue
        seen_names.add(name)
        annotations = mcp_tool.annotations
        read_only = bool(annotations and annotations.read_only_hint)
        # Only an EXPLICIT true forces confirm. The MCP spec defaults
        # destructive_hint to true when absent, but applying that default
        # would confirm-prompt every unannotated tool and make mcp mounting
        # unusable; mutating mcp tools are covered by the taint gate instead.
        destructive = bool(annotations and annotations.destructive_hint is True)
        tools.append(
            _build_tool(
                server,
                server_name=server_name,
                tool_name=name,
                mcp_tool=mcp_tool,
                read_only=read_only,
                confirm=destructive,
            )
        )
    return tools


# -- the mount --------------------------------------------------------------------


class MCPMount:
    """Spawns each configured MCP server (stdio subprocess or streamable
    HTTP), discovers its tools, and hands back dudamel `Tool` objects
    (`origin="mcp"`) ready for `Registry.add_mcp_tools`. Each entry in
    `servers` is either a shell-style command string (split with
    `shlex.split`, launched over stdio) or an `MCPServerConfig` (either
    transport, plus HTTP headers and/or per-server env).

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

    def __init__(
        self,
        servers: Sequence[str | MCPServerConfig],
        *,
        env_passthrough: Sequence[str] = (),
    ) -> None:
        # Normalization (string -> MCPServerConfig) is deliberately NOT done
        # here: MCPServerConfig("") -- e.g. from os.environ.get("VAR", "")
        # -- fails __post_init__'s "exactly one of command/url" check, and
        # raising out of __init__ would crash MCPMount(...) itself before
        # any server is even attempted, taking Runtime.start() down with it.
        # Normalizing per-entry inside mount()'s try (below) keeps a
        # malformed string entry a warn-and-skip, like any other bad config.
        self._entries: list[str | MCPServerConfig] = list(servers)
        self._env_passthrough = tuple(env_passthrough)
        self._servers: list[_MountedServer] = []

    async def mount(self) -> list[Tool]:
        tools: list[Tool] = []
        seen_names: set[str] = set()
        for entry in self._entries:
            label = entry if isinstance(entry, str) else entry.label
            server: _MountedServer | None = None
            try:
                # A plain string stays a stdio command, so existing configs
                # keep working. The global env_passthrough applies to string
                # entries only; an MCPServerConfig carries its own `env`.
                config = (
                    MCPServerConfig(command=entry, env=self._env_passthrough)
                    if isinstance(entry, str)
                    else entry
                )
                server = _MountedServer(config)
                await asyncio.wait_for(server.connect(), timeout=MOUNT_TIMEOUT)
                listed = await asyncio.wait_for(
                    server.session.list_tools(),  # type: ignore[union-attr]
                    timeout=MOUNT_TIMEOUT,
                )
                assert server.session is not None
                collected = _collect_server_tools(
                    server,
                    server_name=server.server_name,
                    command=config.label,
                    mcp_tools=listed.tools,
                    seen_names=seen_names,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                # Not just `except Exception`: measured on mcp 2.0.0, a
                # ClientSession whose background task group has already
                # failed (e.g. an HTTP transport that never connected)
                # raises a bare CancelledError -- a BaseException -- on its
                # NEXT operation, and this block calls session.initialize()
                # and list_tools(). An unreachable URL's own connect failure
                # is already Exception-derived (a native ExceptionGroup
                # wrapping httpx2.ConnectError) and would be caught either
                # way; this widening is for the session-level fallout, not
                # the initial connect. Same discipline, same anyio reason,
                # as close(). It also covers MCPServerConfig(...)
                # construction above raising ValueError on a malformed
                # string entry, before any server object even exists.
                logger.warning(
                    "mcp: server %r failed to mount (%s: %s) -- skipping; core unaffected",
                    label,
                    type(e).__name__,
                    e,
                )
                if server is not None:
                    await server.close()
                continue
            self._servers.append(server)
            tools.extend(collected)
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
