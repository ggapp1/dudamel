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

A server whose process dies is brought back automatically, but only within
tight limits. Each mounted server owns one long-lived supervisor task that
holds its `AsyncExitStack`; a failing tool call *signals* that task instead
of touching the stack itself, because the transport and `ClientSession` open
anyio cancel scopes that may only be exited by the task that entered them. A
reconnect always discards the whole transport + session pair and builds a
fresh one: once a session has seen its connection die, every later operation
on it fails too, so reusing it can never work. Reconnects are serialized per
server, so a burst of failing calls produces exactly one rebuild.

`MAX_RECONNECT_ATTEMPTS` with exponential backoff bounds one such burst, not
the process. If the whole burst fails, the server's tools fail fast for
`RECONNECT_COOLDOWN_SECONDS` and then one more burst is allowed. That is
deliberate: what a reconnect usually waits for is somebody else's
deployment, and an HTTP server being restarted, rolled, or cut over is
routinely gone for far longer than three backed-off attempts span --
disabling its tools permanently because it took a minute to come back would
defeat the feature in the case it exists for. The cooldown is what keeps
this bounded: a server that is gone for good costs a few attempts per
cooldown, not attempts on every tool call. A server that never mounted
successfully in the first place is different, and does stay skipped for the
rest of the process lifetime -- it contributed no tools, so there is nothing
to reconnect for.

What a reconnect deliberately does NOT do is re-run the call that failed,
unless re-running it is provably harmless. A read-only tool is retried and
the caller never learns anything happened. For a mutating tool the request
may well have been executed before the connection dropped, so the call
returns an error saying the outcome is UNKNOWN rather than saying it failed:
reporting failure invites a retry that performs the side effect twice, and
one confirmation from the user has to mean one execution.

Annotations are re-read on every reconnect, because a restarted server is
not required to be the same server. A tool whose `read_only_hint` or
`destructive_hint` changed is force-gated to `confirm=True` (and loses its
retry-safety) with a warning -- the classification it was registered under
is no longer something this server can be trusted to have kept. A tool that
is no longer advertised at all returns an error on its next call.

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
from mcp.shared.exceptions import MCPError

from dudamel.contract.types import TOOL_NAME_RE, Tool
from dudamel.exceptions import ToolValidationError

logger = logging.getLogger("dudamel.mcp")

CALL_TIMEOUT = 30.0  # per-tool-call timeout, enforced natively -- see `_make_call_fn`
MOUNT_TIMEOUT = 15.0  # connect + list_tools budget per server, at mount time
CLOSE_TIMEOUT = 10.0  # how long to wait for a server's supervisor task to exit

# Trap for a future test: `MCPMount`/`_MountedServer`'s `call_timeout`/
# `mount_timeout` keyword defaults are bound to these two constants at
# IMPORT time, an ordinary Python default-argument gotcha. A test that does
# `monkeypatch.setattr(mcp_mount, "MOUNT_TIMEOUT", ...)` and then constructs
# `MCPMount(...)` WITHOUT passing `mount_timeout` explicitly will NOT see the
# patched value -- the default was already resolved when this module was
# first imported. Pass the value explicitly instead of relying on a
# monkeypatched module constant to reach a default.

# `Tool.timeout` -- and so the Router's OWN `asyncio.wait_for(tool.fn(...),
# tool.timeout)` in router.py -- is deliberately set to call_timeout PLUS
# this margin, never to call_timeout itself. The two clocks do not start at
# the same instant: the Router's starts before `tool.fn` is even entered,
# while the SDK's `read_timeout_seconds` clock (passed into
# `session.call_tool`) only starts once the request has actually been
# serialized and dispatched. Setting them equal would therefore make the
# Router's outer timer expire FIRST, deterministically, on every genuinely
# slow call -- which cancels the task, and `call()`'s own cancellation
# handling (see `_cancellation_was_requested`) correctly treats that as "not
# a connection death" and re-raises it, so `wait_for` reports the bare,
# uncoded `TimeoutError` this whole design exists to move away from. This
# margin is what actually gives the native, coded `MCPError(REQUEST_TIMEOUT)`
# path room to fire first; without it, the Router layer isn't a backstop,
# it's the layer that wins.
ROUTER_TIMEOUT_MARGIN_SECONDS = 5.0

# How many times a dead server is rebuilt in one burst, and the base delay
# between those attempts (which doubles each time: 0s, 0.5s, 1.0s). Bounded
# on purpose -- a server that is gone must not turn every tool call into a
# multi-second stall. When a burst fails outright the server's tools fail
# fast for RECONNECT_COOLDOWN_SECONDS, and the next call after that gets a
# fresh burst: an HTTP server being restarted or rolled is routinely gone
# for much longer than one burst spans, so the budget bounds a burst rather
# than the process lifetime.
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_BACKOFF_SECONDS = 0.5
RECONNECT_COOLDOWN_SECONDS = 60.0

# How many times a server's supervisor may be cancelled by its own transport
# WITHIN ONE RECONNECT BURST before that server is abandoned. Guards against
# a cancel source stuck in a tight loop. The count is reset when a burst
# begins and when a connection succeeds, so it never accumulates across
# bursts -- what bounds the long run is the cooldown, not this. Sized above
# MAX_RECONNECT_ATTEMPTS because, measured, every attempt in a burst against
# a fully-down HTTP server ends in a cancellation from the connection it was
# building; a budget at one burst's worth would retire a server for a single
# failed burst.
MAX_SCOPE_RECOVERIES = 2 * MAX_RECONNECT_ATTEMPTS

# Sizing note, not a defect: a full burst against a server that accepts
# connections but never finishes `initialize()` costs up to
# MAX_RECONNECT_ATTEMPTS * MOUNT_TIMEOUT plus backoff (~46s), which exceeds
# CALL_TIMEOUT. The Router's own timeout fires first, so the tool call
# reports a TimeoutError rather than the reconnect's diagnosis -- and that
# timeout cancels the tool-call task, which cancels the rest of the burst
# too. Only the attempt already handed to the supervisor runs to completion;
# the remaining attempts are abandoned and no cooldown is recorded, so the
# next tool call simply starts a fresh burst.

# An MCP server's input_schema and description are embedded verbatim in every
# LLM request, so both are a token-budget cost AND a prompt-injection channel
# that fires at advertise time -- before any tool call, and therefore with no
# taint at all. 16 KiB is ~4-5k tokens against a default window_tokens of
# 8000, so a schema that wants a higher cap is already unusable here.
MAX_SCHEMA_BYTES = 16384
MAX_DESCRIPTION_CHARS = 1024

_REFUSAL = "MCP server-initiated callbacks are not supported in dudamel v1"

# Exact `MCPError` messages that mean "this HTTP session id is gone, start a
# new one" rather than "your request was malformed" -- see
# `_is_connection_death`.
_HTTP_SESSION_GONE = frozenset({"session terminated", "session not found", "session expired"})


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


# -- connection-death classification ---------------------------------------------


def _cancellation_was_requested() -> bool:
    """True when THIS task has an outstanding `Task.cancel()` against it.

    That is all it distinguishes, and the distinction is narrower than it
    looks: anyio delivers cancel-scope cancellation by calling
    `Task.cancel()` too, so a task that is a *member* of a cancelled anyio
    scope also answers True here. What answers False is a `CancelledError`
    object that merely propagated to this task -- e.g. from awaiting a
    future that someone else cancelled -- without any cancel request against
    this task.

    That is exactly the distinction the tool-call path needs. The Router
    runs every tool under `asyncio.wait_for`, which cancels the task running
    the tool; that cancellation must pass through untouched or `wait_for`
    never converts it into the `TimeoutError` the Router reports, and a
    merely slow tool would be misreported as a dead connection. A Router
    task, meanwhile, is never a member of an MCP session's cancel scopes:
    those are entered by -- and belong to -- the server's supervisor task,
    so a death arrives at the Router as a propagated `CancelledError` with
    no cancel request against it.

    It is NOT a usable test inside the supervisor task, which *is* the host
    of those scopes; see `_supervise`.
    """
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _is_connection_death(e: BaseException) -> bool:
    """True if `e` says the transport is gone and only a fresh connection
    could help; False for an ordinary tool-side failure, which a reconnect
    would not fix and must not be allowed to trigger one.

    Measured against this SDK generation rather than assumed:

    - anyio's `ClosedResourceError`/`BrokenResourceError` never reach a
      caller. The dispatcher converts them to `MCPError(CONNECTION_CLOSED)`,
      which is what the call that was in flight when the server died sees.
    - `MCPError(REQUEST_TIMEOUT)` is deliberately NOT death. A slow tool is
      not a dead transport, and reconnecting on every timeout would turn
      ordinary tool latency into connection churn.
    - The *next* call issued on a session that has already seen a death
      raises a bare `asyncio.CancelledError` instead -- a `BaseException`,
      invisible to `except Exception`. Call sites must therefore catch
      `BaseException`, and must rule out their own cancellation first (see
      `_cancellation_was_requested`), because the exception alone cannot say
      who asked for it.
    - A background failure that was deferred inside a task group surfaces as
      an exception group, so groups are searched member by member.

    Known false negative: an exception group whose members are raw transport
    errors (`httpx2.*`, `anyio.*`) rather than `MCPError`s is not recognized
    here, because matching them would mean importing two packages dudamel
    does not depend on directly. Those only surface while a transport is
    being torn down -- a path that is already guarded -- so the cost is that
    such a death is noticed one call later than it could be, not that it is
    missed.
    """
    if isinstance(e, MCPError):
        if e.code == types.CONNECTION_CLOSED:
            return True
        # A streamable-HTTP server that restarted is still reachable, so
        # nothing above fires -- but it no longer knows the session id we
        # hold, and answers 404. Measured, that reaches a caller as
        # INVALID_REQUEST with one of two messages: "Session terminated"
        # (synthesized by the client transport) or whatever JSON-RPC error
        # the server itself put in the body, "Session not found" for a
        # spec-correct one. The protocol's remedy for a 404 against a session
        # id is to start a new session, which is precisely a reconnect.
        #
        # Matched on the exact known messages rather than on INVALID_REQUEST
        # alone: that code is also how ordinary malformed-request errors
        # arrive, and reconnecting on those would be churn. An unrecognized
        # wording simply does not reconnect, which is no worse than before.
        if e.code == types.INVALID_REQUEST:
            message = str(getattr(e, "message", "") or e).strip().lower()
            return message in _HTTP_SESSION_GONE
        return False
    if isinstance(e, asyncio.CancelledError):
        return True
    if isinstance(e, BaseExceptionGroup):
        return any(_is_connection_death(sub) for sub in e.exceptions)
    return False


# -- tool call adapter -----------------------------------------------------------


def _make_call_fn(
    server: _MountedServer,
    *,
    remote_name: str,
    local_name: str,
    read_only: bool,
    call_timeout: float = CALL_TIMEOUT,
) -> Callable[..., Awaitable[str]]:
    """Build the `Tool.fn` for one remote tool, including its reconnect and
    retry policy.

    `read_only` is the classification the tool was registered under. It is
    only half the retry decision: `_MountedServer.is_retry_safe` can revoke
    it later if the server comes back advertising different annotations, and
    a revoked tool is never retried again.

    `call_timeout` bounds the call through the SDK's own
    `read_timeout_seconds`, NOT `asyncio.wait_for`. Measured against this SDK
    generation: a native timeout raises `MCPError(code=REQUEST_TIMEOUT,
    -32001)`, a coded exception `_is_connection_death` reads and deliberately
    excludes from "the connection is dead" (a slow tool is not a dead
    transport -- see that function's docstring). `asyncio.wait_for` instead
    raises a bare `TimeoutError` with no `.code`, indistinguishable from any
    other timeout by anything downstream, AND it works by cancelling this
    task -- which is exactly the `CancelledError` hazard `invoke()`'s callers
    already have to disentangle from a real connection death via
    `_cancellation_was_requested()`. The native path avoids manufacturing
    that ambiguity in the first place. (The Router's OWN `asyncio.wait_for`
    around `tool.fn` is unaffected by this and still applies -- see
    `Tool.timeout` on the built `Tool` -- it is a generic backstop for every
    tool, mcp-origin or not, and this local, coded timeout is expected to
    fire first.)
    """

    async def invoke(session: ClientSession, **kwargs: Any) -> str:
        result = await session.call_tool(remote_name, kwargs, read_timeout_seconds=call_timeout)
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

    def vanished() -> RuntimeError:
        return RuntimeError(
            f"mcp tool {local_name}: the server no longer advertises this tool after "
            f"reconnecting, so it cannot be called"
        )

    def no_session() -> RuntimeError:
        return RuntimeError(
            f"mcp tool {local_name} has no live session: the server is not connected and "
            f"could not be reconnected"
        )

    def demote(e: BaseException) -> RuntimeError:
        """Turn a `BaseException` into an ordinary error for the Router.

        Letting one escape `Tool.fn` would tear down the Router's task
        instead of producing an error tool-result the model can read.
        """
        return RuntimeError(f"mcp tool {local_name} failed: {type(e).__name__}: {e}")

    async def call(**kwargs: Any) -> str:
        # Resolved per call, never captured: Registry stores these Tool
        # objects permanently, so a session rebound by reconnect has to reach
        # tools that were built against the previous one.
        if server.is_vanished(remote_name):
            raise vanished()
        generation = server.generation
        session = server.session
        if session is None or not server.alive:
            # Provably pre-dispatch: nothing has been written to the server,
            # so rebuilding and calling once cannot double anything up. This
            # is the only case where a MUTATING tool is (re)dispatched by
            # this code at all.
            await server.reconnect(generation)
            if server.is_vanished(remote_name):
                raise vanished()
            generation = server.generation
            session = server.session
            if session is None:
                raise no_session()
        try:
            return await invoke(session, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            if _cancellation_was_requested():
                # The Router's own timeout, not the server dying. Passing it
                # through is what lets `wait_for` report a TimeoutError.
                raise
            if not _is_connection_death(e):
                if isinstance(e, Exception):
                    raise
                raise demote(e) from None
            await server.reconnect(generation)
            if read_only and server.is_retry_safe(remote_name):
                # Safe by construction: a read-only tool has no side effect
                # to double up on, so it does not matter whether the request
                # reached the server before the connection dropped.
                if server.is_vanished(remote_name):
                    raise vanished() from None
                retry_session = server.session
                if retry_session is None:
                    raise no_session() from None
                try:
                    return await invoke(retry_session, **kwargs)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as retry_error:
                    if _cancellation_was_requested():
                        raise
                    if isinstance(retry_error, Exception):
                        raise
                    raise demote(retry_error) from None
            # Anything mutating stops here, even though the failure looks
            # like the session was already dead before this call: the call
            # that is in flight when a server dies and the first call after
            # it are NOT reliably distinguishable from the exception, so the
            # only safe reading is that the side effect may have landed.
            # Saying "failed" would invite a retry that performs it twice.
            raise RuntimeError(
                f"mcp tool {local_name}: the server connection died during the call and "
                f"the outcome is UNKNOWN -- it may or may not have taken effect. Do not "
                f"retry automatically; check the server's state first."
            ) from None

    return call


def _build_tool(
    server: _MountedServer,
    *,
    server_name: str,
    tool_name: str,
    mcp_tool: types.Tool,
    read_only: bool,
    confirm: bool,
    call_timeout: float,
) -> Tool:
    raw_description = mcp_tool.description or mcp_tool.name
    # Truncate, not drop: a truncated description is still a description.
    if len(raw_description) > MAX_DESCRIPTION_CHARS:
        raw_description = raw_description[:MAX_DESCRIPTION_CHARS] + "..."
    description = f"[experimental MCP] {raw_description}"
    input_schema = mcp_tool.input_schema or {"type": "object", "properties": {}}
    tool = Tool(
        name=tool_name,
        app_name=f"mcp:{server_name}",
        description=description,
        fn=_make_call_fn(
            server,
            remote_name=mcp_tool.name,
            local_name=tool_name,
            read_only=read_only,
            call_timeout=call_timeout,
        ),
        schema=McpToolSchema(input_schema),
        read_only=read_only,
        confirm=confirm,
        # Primary enforcement is the transport-native path inside `invoke()`
        # (see `_make_call_fn`), not this. This is a true backstop, not a
        # duplicate: call_timeout + ROUTER_TIMEOUT_MARGIN_SECONDS, strictly
        # LARGER than call_timeout, so the Router's own
        # `asyncio.wait_for(tool.fn(...), timeout=tool.timeout)` in
        # router.py has no chance to win the race and pre-empt the native
        # timeout with an uncoded `TimeoutError` -- see the comment on
        # ROUTER_TIMEOUT_MARGIN_SECONDS for why an equal value would do
        # exactly that. If a future change to `invoke()` ever forgets to
        # pass a per-call timeout at all, this still bounds the call, just
        # later and without the coded MCPError the native path gives.
        timeout=call_timeout + ROUTER_TIMEOUT_MARGIN_SECONDS,
        origin="mcp",
    )
    # The server keeps a handle on every tool it actually contributed, so a
    # later reconnect can compare what comes back against what was
    # registered -- and force-gate or disable the tool in place, since the
    # Registry holds these exact objects for the process lifetime.
    server.register(mcp_tool.name, tool, read_only=read_only, destructive=confirm)
    return tool


# -- one mounted server ----------------------------------------------------------


class _SupervisorGone(RuntimeError):
    """A submitted operation was never run, because the supervisor holding
    it stopped first. Distinct from an operation that ran and failed: a lost
    one can be resubmitted, a failed one is an answer."""


@dataclass
class _RegisteredTool:
    """One dudamel `Tool` this server contributed, plus the safety
    annotations it was contributed under, so drift can be detected."""

    tool: Tool
    read_only: bool
    destructive: bool


class _MountedServer:
    """One MCP server's connection, owned by a single supervisor task.

    The transport and `ClientSession` are async context managers that open
    anyio cancel scopes, and anyio requires the task that entered a scope to
    be the task that exits it. Tool calls arrive on Router tasks, and startup
    and shutdown happen on the Runtime's task, so no caller may enter or
    exit that stack directly. Instead every caller *submits* an operation to
    the supervisor task, which owns the `AsyncExitStack` from the moment it
    is created until it exits.

    `session`, `server_name` and `alive` are rebound by the supervisor and
    read (never written) by callers; `generation` counts successful
    connections, and is how a caller says "I failed against connection N"
    without having to hold a lock while it makes its tool call.
    """

    def __init__(self, config: MCPServerConfig, *, mount_timeout: float = MOUNT_TIMEOUT) -> None:
        self.config = config
        self.mount_timeout = mount_timeout
        self.session: ClientSession | None = None
        self.server_name: str = ""
        # The self-reported name the tools were registered under, kept apart
        # from `server_name` so a restarted server renaming itself is a
        # warning rather than a silent divergence from the registered names.
        self._registered_server_name: str = ""
        self.listed_tools: list[types.Tool] = []
        self.alive = False
        # Bumped on every successful connection, including the first.
        self.generation = 0
        # Diagnostics, and what the tests assert coalescing against:
        # `reconnect_count` counts rebuild *cycles* (one per burst of failing
        # calls), `reconnect_attempts` counts individual connection attempts.
        self.reconnect_count = 0
        self.reconnect_attempts = 0
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()
        self._requests: asyncio.Queue[tuple[str, asyncio.Future[BaseException | None]]] = (
            asyncio.Queue()
        )
        self._supervisor: asyncio.Task[None] | None = None
        self._registered: dict[str, _RegisteredTool] = {}
        self._vanished: set[str] = set()
        self._drifted: set[str] = set()
        # `_mounted` gates reconnect on "this server worked at least once":
        # one that failed at startup contributed no tools and must stay
        # skipped for the process lifetime rather than being retried forever.
        self._mounted = False
        # Permanent, and set only by close(): the server is being shut down,
        # so the supervisor must stop and nothing may reconnect it. Kept
        # strictly apart from the reconnect budget below, which is temporary.
        self._closed = False
        # Deadline before which no further reconnect burst may start. A spent
        # budget is a cooldown, not a retirement -- see `reconnect`.
        self._cooldown_until = 0.0
        # Consecutive supervisor stand-downs with no successful connection in
        # between; the bound on `MAX_SCOPE_RECOVERIES`.
        self._standdowns = 0
        # Set when those stand-downs are exhausted: the supervisor is not
        # respawned again, because something is cancelling it in a loop.
        self._supervisor_failed = False

    # -- what callers read ----------------------------------------------------

    def register(self, remote_name: str, tool: Tool, *, read_only: bool, destructive: bool) -> None:
        self._registered[remote_name] = _RegisteredTool(
            tool=tool, read_only=read_only, destructive=destructive
        )

    def is_vanished(self, remote_name: str) -> bool:
        """True if the server stopped advertising this tool across a
        reconnect. Calling it would be a request into the void."""
        return remote_name in self._vanished

    def is_retry_safe(self, remote_name: str) -> bool:
        """False once a tool's safety annotations have changed underneath
        us. Transparent retries are justified by a read-only classification;
        a server that has already contradicted its own classification has
        forfeited that justification."""
        return remote_name not in self._drifted

    # -- the supervisor task --------------------------------------------------

    async def _submit(self, op: str) -> BaseException | None:
        """Ask the supervisor to run `op` and wait for the outcome.

        Returns the failure instead of raising it: this is called from tool
        calls and from shutdown, and neither may be handed a `BaseException`
        it did not ask for.

        Respawns the supervisor if a previous one stood down after absorbing
        a cancellation. A stood-down supervisor has already released its
        stack, so there is nothing to inherit and a fresh one starts clean.

        Tried at most twice. A supervisor can stand down while already
        holding this request, which answers it without ever running it --
        measured against a fully-down HTTP server, that was two of every
        three reconnect attempts, each reported as a failed attempt that in
        truth was never made. A lost request is not a failed operation, so
        it is resubmitted once to a fresh supervisor. Safe to repeat: the
        only ops are "connect", which rebuilds from scratch and is
        idempotent, and "stop", which `close()` submits directly.
        """
        error = await self._submit_once(op)
        if isinstance(error, _SupervisorGone):
            error = await self._submit_once(op)
        return error

    async def _submit_once(self, op: str) -> BaseException | None:
        if self._closed or self._supervisor_failed:
            return _SupervisorGone("mcp: server supervisor is not running")
        supervisor = self._supervisor
        if supervisor is None:
            return _SupervisorGone("mcp: server supervisor is not running")
        if supervisor.done():
            supervisor = self._spawn_supervisor()
        return await self._submit_to(supervisor, op)

    def _spawn_supervisor(self) -> asyncio.Task[None]:
        supervisor = asyncio.create_task(
            self._supervise(), name=f"mcp-supervisor:{self.config.label}"
        )
        self._supervisor = supervisor
        return supervisor

    async def _submit_to(self, supervisor: asyncio.Task[None], op: str) -> BaseException | None:
        if supervisor.done():
            return _SupervisorGone("mcp: server supervisor is not running")
        fut: asyncio.Future[BaseException | None] = asyncio.get_running_loop().create_future()
        await self._requests.put((op, fut))
        # Wait on the supervisor too: if it ever dies mid-operation, its
        # future would never resolve and this would hang the caller forever.
        await asyncio.wait([fut, supervisor], return_when=asyncio.FIRST_COMPLETED)
        if fut.done():
            return fut.result()
        fut.cancel()
        return _SupervisorGone("mcp: server supervisor exited without answering")

    @staticmethod
    def _resolve(fut: asyncio.Future[BaseException | None], value: BaseException | None) -> None:
        if not fut.done():
            fut.set_result(value)

    async def _supervise(self) -> None:
        """Own this server's `AsyncExitStack` for its whole lifetime.

        Operations are serialized here by construction -- one task, one
        queue -- so a connect can never overlap a teardown, and every cancel
        scope is entered and exited by this task and no other.

        This task is the *host* of the transport's and the `ClientSession`'s
        anyio task groups, because it is the task that entered them. Measured
        consequence: when one of those groups fails in the background -- an
        HTTP transport losing its connection is the ordinary case -- anyio
        cancels its host, which lands here as a `CancelledError` on whatever
        this task is awaiting -- the idle `_requests.get()`, or a connection
        attempt in flight inside `_rebuild`. Letting that kill the supervisor
        outright would leak the whole stack (nothing would ever `aclose()`
        it) and strand the server, since every later `_submit` would find no
        supervisor to talk to.

        So a cancellation is caught HERE, wherever in the loop it arose --
        `_rebuild` deliberately re-raises rather than handling its own -- the
        stack is unwound inline, and this task then **stands down**: it
        returns rather than re-arming, and `_submit` starts a fresh
        supervisor when there is work again. It does not re-arm in place
        because re-arming makes this task uncancellable: `asyncio.run`'s
        shutdown cancels each remaining task exactly once and then waits for
        it, so a supervisor that absorbs that one cancellation and goes back
        to `_requests.get()` never finishes and loop teardown hangs forever
        (measured, on both paths). A hang is a worse failure than the leak
        this whole path exists to prevent. Standing down keeps both
        properties: the stack is always released, and the task always
        completes.

        Consecutive stand-downs with no successful connection in between are
        bounded by `MAX_SCOPE_RECOVERIES`; past that the supervisor is not
        respawned, so a cancel source stuck in a loop cannot spin.
        """
        pending: asyncio.Future[BaseException | None] | None = None
        try:
            while True:
                try:
                    op, fut = await self._requests.get()
                    pending = fut
                    if op == "stop":
                        await self._teardown()
                        self._resolve(fut, None)
                        return
                    self._resolve(fut, await self._rebuild())
                    pending = None
                except asyncio.CancelledError as e:
                    if self._closed:
                        # close() gave up waiting and cancelled us. Stop.
                        raise
                    if pending is not None:
                        self._resolve(pending, e)
                        pending = None
                    self._standdowns += 1
                    exhausted = self._standdowns > MAX_SCOPE_RECOVERIES
                    # Release the stack either way -- that is the whole point
                    # of catching this -- and only then decide whether this
                    # server is worth serving again.
                    await self._stand_down(final=exhausted)
                    if exhausted:
                        self._supervisor_failed = True
                    return
        except (KeyboardInterrupt, SystemExit) as e:
            # Resolve before re-raising, or whoever is waiting on this
            # operation waits for a task that is never coming back.
            if pending is not None:
                self._resolve(pending, e)
            raise
        finally:
            # Anything still queued behind a failed supervisor gets an
            # answer rather than a hang.
            if pending is not None:
                self._resolve(pending, _SupervisorGone("mcp: server supervisor stopped"))
            while not self._requests.empty():
                _, queued = self._requests.get_nowait()
                self._resolve(queued, _SupervisorGone("mcp: server supervisor stopped"))

    async def _stand_down(self, *, final: bool) -> None:
        """Absorb a cancellation delivered by one of this server's own anyio
        scopes and release everything it was holding, so this task can finish.

        `_teardown()` is awaited inline and NOT under `asyncio.shield`.
        Shield runs its argument in a *new* task, and exiting an anyio cancel
        scope from a task other than the one that entered it is exactly the
        corruption this whole structure exists to prevent -- measured, it
        fails with "Attempted to exit cancel scope in a different task than
        it was entered in". Unwinding the scope inline is also what clears
        the cancellation: anyio's `__aexit__` uncancels the host task on its
        way out and re-raises the deferred background error, which
        `_teardown` logs.
        """
        if final:
            logger.warning(
                "mcp: server %r has had its connection scope cancelled %d times in a row "
                "without a working connection in between -- giving up on it; its tools "
                "stay unavailable for the rest of this process. Core unaffected.",
                self.config.label,
                self._standdowns,
            )
        else:
            logger.warning(
                "mcp: server %r cancelled its own connection scope -- its transport failed "
                "in the background. Discarding the connection; the next tool call will "
                "reconnect.",
                self.config.label,
            )
        await self._teardown()
        # Defensive: anyio uncancels the host task as it unwinds, but if
        # there was no stack left to unwind nothing did that for us, and a
        # task carrying a pending cancel request aborts its very next await.
        task = asyncio.current_task()
        while task is not None and task.cancelling() > 0:
            task.uncancel()

    async def _rebuild(self) -> BaseException | None:
        """Discard whatever connection exists and build a brand-new one.

        Never a partial rebuild. A session that has seen its connection die
        stays dead: every subsequent operation on it fails, including ones
        that have nothing to do with the failure, so rebinding a fresh
        transport onto the old session (or retrying against it) cannot work.
        """
        await self._teardown()
        stack = AsyncExitStack()
        self._stack = stack
        try:
            await asyncio.wait_for(self._connect(stack), timeout=self.mount_timeout)
        except (KeyboardInterrupt, SystemExit):
            raise
        except asyncio.CancelledError:
            # ALWAYS re-raised, never turned into a return value. Measured on
            # 3.11+: `wait_for` reports its own expiry as `TimeoutError` and
            # never as `CancelledError`, so a cancellation arriving here is
            # always someone else's -- an anyio scope this half-built
            # connection opened, or the event loop shutting the process down.
            # Absorbing it and returning normally would send `_supervise`
            # back to its idle await carrying a cancellation that was meant
            # to stop it, which is exactly how an unclosed mount used to hang
            # `asyncio.run`'s teardown forever (measured: no exit, SIGKILL at
            # 25s, with the supervisor parked inside this call).
            #
            # `_supervise`'s stand-down handler is the single owner of this
            # path. It releases `self._stack` -- already installed above, so
            # the half-built connection is not leaked -- lets the task
            # finish, and counts the stand-down against
            # `MAX_SCOPE_RECOVERIES`. This is deliberately NOT a second,
            # overlapping layer: a `CancelledError` handled here instead of
            # there would defeat that guarantee rather than duplicate it.
            raise
        except BaseException as e:
            await self._teardown()
            return e
        self.alive = True
        self.generation += 1
        # A working connection clears the stand-down budget (the reconnect
        # cooldown is cleared separately, by `reconnect` itself). It exists
        # to stop a pathological loop, not to accumulate over a long uptime.
        self._standdowns = 0
        return None

    async def _teardown(self) -> None:
        stack, self._stack = self._stack, None
        self.session = None
        self.alive = False
        if stack is None:
            return
        # Best-effort: a server that's already dead/misbehaving must not make
        # shutdown -- or the next reconnect -- fail. Unwinding a failed
        # transport routinely raises here: `CancelledError` from a cancel
        # scope, or the exception group anyio defers until teardown carrying
        # the real background error. Both are `BaseException`s a plain
        # `except Exception` would miss, and neither may propagate -- this is
        # the one place that guarantees the stack is released, so it must
        # always run to completion. Only KeyboardInterrupt/SystemExit escape.
        try:
            await stack.aclose()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.warning(
                "mcp: error closing server %r (%s: %s) -- ignoring, best-effort",
                self.config.label,
                type(e).__name__,
                e,
            )

    async def _open_transport(self, stack: AsyncExitStack) -> tuple[Any, Any]:
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
            # The client is entered into `stack` here, not just constructed:
            # `streamable_http_client` only closes the httpx2.AsyncClient it
            # manages when IT created one internally (`http_client=None`).
            # Since we always pass one in explicitly, streamable_http_client
            # never closes it -- entering it into our own stack, before the
            # transport, is what makes it get closed at all, and LIFO order
            # tears the transport down first.
            http_client = await stack.enter_async_context(
                create_mcp_http_client(headers=self.config.headers)
            )
            read, write = await stack.enter_async_context(
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
        return await stack.enter_async_context(stdio_client(params))

    async def _connect(self, stack: AsyncExitStack) -> None:
        """Handshake and discovery, always as one unit.

        `list_tools()` runs here rather than at the call site so a session
        whose connection died between `initialize()` and discovery fails the
        whole connection attempt, inside the supervisor, instead of raising
        a bare `CancelledError` at whoever asked for it.
        """
        read, write = await self._open_transport(stack)
        session = await stack.enter_async_context(
            ClientSession(
                read,
                write,
                sampling_callback=_refuse_sampling,
                elicitation_callback=_refuse_elicitation,
                list_roots_callback=_refuse_list_roots,
            )
        )
        init = await session.initialize()
        listed = await session.list_tools()
        self.session = session
        self.server_name = sanitize_mcp_name(init.server_info.name)
        self.listed_tools = list(listed.tools)

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        """Launch the supervisor and perform the first connection.

        Raises on failure so `MCPMount.mount()`'s per-server handler can log
        and skip the server, exactly as it did when connecting inline.
        """
        self._spawn_supervisor()
        error = await self._submit("connect")
        if error is not None:
            if isinstance(error, Exception):
                raise error
            # Never re-raise a BaseException at a caller that only asked for
            # a connection -- describe it instead.
            raise RuntimeError(f"{type(error).__name__}: {error}")
        self._registered_server_name = self.server_name
        self._mounted = True

    async def reconnect(self, seen_generation: int) -> bool:
        """Rebuild this connection, or report that it cannot be rebuilt.

        `seen_generation` is the `generation` the caller was working against
        when it failed. If the connection has already been rebuilt since
        then, some other caller in the same burst of failures did the work
        and this one returns immediately -- that, plus the lock, is what
        makes a batch of simultaneously-failing calls produce exactly one
        reconnect instead of one per call.

        `MAX_RECONNECT_ATTEMPTS` bounds one *burst*, not the process. A burst
        that fails outright starts a `RECONNECT_COOLDOWN_SECONDS` cooldown
        during which calls fail fast; after it, the next call is allowed a
        fresh burst. The bound has to work this way because the thing being
        waited for is usually somebody else's deployment: an HTTP server
        being restarted, rolled, or cut over is routinely gone for far longer
        than three backed-off attempts span, and permanently disabling its
        tools because it took a minute to come back would make the whole
        reconnect path useless in exactly the case it was built for. The
        cooldown is what keeps that from becoming an unbounded retry loop --
        a permanently dead server costs a handful of attempts per cooldown,
        not one per tool call.

        Never raises, and never enters or exits a context itself: it only
        signals the supervisor task that owns them.
        """
        if not self._mounted or self._closed or self._supervisor_failed:
            # Never worked at all, being shut down, or its supervisor gave
            # up. All three stay down for the rest of the process lifetime.
            return False
        async with self._lock:
            if self._closed or self._supervisor_failed:
                return False
            if self.generation != seen_generation and self.alive:
                return True
            now = asyncio.get_running_loop().time()
            if now < self._cooldown_until:
                # A burst already failed recently. Fail fast rather than
                # making every tool call pay for three more attempts.
                return False
            self.reconnect_count += 1
            # A new burst starts with a fresh stand-down budget. Measured
            # against a fully-down HTTP server, every attempt in a burst ends
            # in a scope cancellation and so costs a stand-down; letting
            # those accumulate across bursts would spend the budget after
            # two or three cooldowns and retire the server permanently --
            # reintroducing, by a slower route, exactly the write-off the
            # cooldown exists to prevent. `MAX_SCOPE_RECOVERIES` is meant to
            # stop a supervisor being cancelled in a tight loop *within* one
            # burst; what bounds the long run is the cooldown, which already
            # caps a permanently dead server at one burst per minute forever.
            self._standdowns = 0
            delay = 0.0
            for attempt in range(MAX_RECONNECT_ATTEMPTS):
                if delay:
                    await asyncio.sleep(delay)
                if self._closed:  # close() landed while we were backing off
                    return False
                self.reconnect_attempts += 1
                error = await self._submit("connect")
                if error is None:
                    logger.info(
                        "mcp: reconnected to server %r (attempt %d)",
                        self.config.label,
                        attempt + 1,
                    )
                    self._cooldown_until = 0.0
                    self._apply_reconnect_drift()
                    return True
                logger.warning(
                    "mcp: reconnect attempt %d/%d for server %r failed (%s: %s)",
                    attempt + 1,
                    MAX_RECONNECT_ATTEMPTS,
                    self.config.label,
                    type(error).__name__,
                    error,
                )
                delay = RECONNECT_BACKOFF_SECONDS * (2**attempt)
            self._cooldown_until = asyncio.get_running_loop().time() + RECONNECT_COOLDOWN_SECONDS
            logger.warning(
                "mcp: server %r could not be reconnected in %d attempts -- its tools fail "
                "fast for the next %.0fs, then one more burst will be tried; core unaffected",
                self.config.label,
                MAX_RECONNECT_ATTEMPTS,
                RECONNECT_COOLDOWN_SECONDS,
            )
            return False

    def _apply_reconnect_drift(self) -> None:
        """Reconcile the freshly discovered tool list with what was
        registered. A restarted server is not obliged to be the same server,
        and the `Tool` objects the Registry holds were built from promises
        the old one made."""
        if self.server_name != self._registered_server_name:
            logger.warning(
                "mcp: server %r now reports itself as %r instead of %r after reconnecting; "
                "its tools keep the names they were registered under",
                self.config.label,
                self.server_name,
                self._registered_server_name,
            )
            self._registered_server_name = self.server_name
        fresh = {tool.name: tool for tool in self.listed_tools}
        for remote_name, registered in self._registered.items():
            advertised = fresh.get(remote_name)
            if advertised is None:
                if remote_name not in self._vanished:
                    logger.warning(
                        "mcp: tool %r is no longer advertised by server %r after reconnecting "
                        "-- calls to %r will return an error until it comes back",
                        remote_name,
                        self.config.label,
                        registered.tool.name,
                    )
                self._vanished.add(remote_name)
                continue
            self._vanished.discard(remote_name)
            annotations = advertised.annotations
            read_only = bool(annotations and annotations.read_only_hint)
            destructive = bool(annotations and annotations.destructive_hint is True)
            if read_only == registered.read_only and destructive == registered.destructive:
                continue
            logger.warning(
                "mcp: tool %r on server %r came back from a reconnect with different safety "
                "annotations (read_only %s -> %s, destructive %s -> %s) -- forcing "
                "confirmation on %r and disabling its automatic retry",
                remote_name,
                self.config.label,
                registered.read_only,
                read_only,
                registered.destructive,
                destructive,
                registered.tool.name,
            )
            # One-way: a tool that has contradicted its own classification
            # once is never quietly trusted again, even if a later reconnect
            # restores the original annotations.
            registered.tool.confirm = True
            registered.tool.read_only = False
            registered.read_only = read_only
            registered.destructive = destructive
            self._drifted.add(remote_name)

    async def close(self) -> None:
        """Stop the supervisor and let it tear its own stack down.

        Safe to call more than once, and safe to call on a server that never
        started. Never raises: shutdown must not be able to fail because a
        server is misbehaving.
        """
        self._closed = True
        supervisor, self._supervisor = self._supervisor, None
        if supervisor is None or supervisor.done():
            # Either it never started, or it already stood down after a
            # cancellation -- in which case it released its stack on the way
            # out and there is nothing left to stop.
            return
        error = await self._submit_to(supervisor, "stop")
        if error is not None:
            logger.warning(
                "mcp: server %r did not shut down cleanly (%s: %s) -- ignoring, best-effort",
                self.config.label,
                type(error).__name__,
                error,
            )
        if not supervisor.done():
            # It answered "stop" but has not finished unwinding. Give it a
            # bounded grace period, then stop waiting -- a wedged server may
            # not hold up the rest of shutdown.
            await asyncio.wait([supervisor], timeout=CLOSE_TIMEOUT)
            if not supervisor.done():
                logger.warning(
                    "mcp: server %r did not finish shutting down within %.0fs -- abandoning it",
                    self.config.label,
                    CLOSE_TIMEOUT,
                )
                supervisor.cancel()


# -- per-server tool collection + MCP-vs-MCP dedupe ----------------------------


def _collect_server_tools(
    server: _MountedServer,
    *,
    server_name: str,
    command: str,
    mcp_tools: Sequence[types.Tool],
    seen_names: set[str],
    call_timeout: float = CALL_TIMEOUT,
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
                call_timeout=call_timeout,
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
        call_timeout: float = CALL_TIMEOUT,
        mount_timeout: float = MOUNT_TIMEOUT,
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
        self._call_timeout = call_timeout
        self._mount_timeout = mount_timeout
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
                server = _MountedServer(config, mount_timeout=self._mount_timeout)
                # Connect + discovery happen inside the server's supervisor
                # task, under this mount's configured mount_timeout;
                # `start()` re-raises whatever went wrong for the handler
                # below.
                await server.start()
                collected = _collect_server_tools(
                    server,
                    server_name=server.server_name,
                    command=config.label,
                    mcp_tools=server.listed_tools,
                    seen_names=seen_names,
                    call_timeout=self._call_timeout,
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
        # Reverse (LIFO) mount order. Each server now enters and exits its
        # own cancel scopes inside its own supervisor task, so scopes are no
        # longer stacked across servers in one task and this ordering is no
        # longer what keeps anyio happy -- but shutting down in the reverse
        # of startup order is still the right default when servers depend on
        # each other, and it costs nothing to keep.
        for server in reversed(self._servers):
            await server.close()
