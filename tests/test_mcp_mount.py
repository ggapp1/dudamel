"""Acceptance tests: MCP mounting (experimental).

These spawn a real stdio MCP server subprocess (tests/fixtures/mcp_echo_server.py)
via the mcp SDK's client transport. That's the whole point -- they ARE the
acceptance tests for discovery/namespacing/read_only mapping, e2e chat, taint,
and degraded-mount behavior -- so they are intentionally NOT marked "slow" or
excluded from the default run; they run every time with the rest of the suite.
"""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path

import mcp.types as types
import pytest

from dudamel import App, Orchestrator, Runtime
from dudamel.config import McpConfig, RouterConfig, Settings, TierConfig
from dudamel.contract.types import TOOL_NAME_RE, Tool
from dudamel.exceptions import RegistryError, ToolValidationError
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call
from dudamel.mcp_mount import (
    MCPMount,
    McpToolSchema,
    _collect_server_tools,
    _make_call_fn,
    _refuse_elicitation,
    _refuse_list_roots,
    _refuse_sampling,
    mcp_tool_name,
    sanitize_mcp_name,
)
from dudamel.registry import Registry

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
FIXTURE_CMD = shlex.join([sys.executable, str(FIXTURE)])


def fixture_cmd(name: str) -> str:
    """A fixture command that self-reports serverInfo.name = `name` instead
    of the default "fixture" -- for tests that need two mounted servers with
    DISTINCT identities (as opposed to tests that mount FIXTURE_CMD twice on
    purpose to exercise the same-identity collision/spoofing path)."""
    return shlex.join([sys.executable, str(FIXTURE), name])


def make_settings(
    tmp_path: Path,
    *,
    router: RouterConfig | None = None,
    mcp: McpConfig | None = None,
    **tiers: TierConfig,
) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/mcp.db",
        data_dir=tmp_path,
        llm_tiers=tiers or {"standard": TierConfig(provider="fake", model="f")},
        router=router or RouterConfig(),
        mcp=mcp or McpConfig(),
    )


# -- discovery / namespacing / read_only mapping ------------------------------


async def test_mount_discovers_fixture_tools_namespaced_and_annotated() -> None:
    mount = MCPMount([FIXTURE_CMD])
    try:
        tools = await mount.mount()
    finally:
        await mount.close()
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"fixture__echo", "fixture__mutate", "fixture__read_env"}
    assert by_name["fixture__echo"].read_only is True
    assert by_name["fixture__mutate"].read_only is False  # unannotated -> mutating
    assert by_name["fixture__read_env"].read_only is True
    assert all(t.origin == "mcp" for t in by_name.values())
    assert all(t.description.startswith("[experimental MCP]") for t in by_name.values())
    assert all(t.timeout == 30.0 for t in by_name.values())
    assert all(t.confirm is False for t in by_name.values())


async def test_mounted_tool_fn_round_trips_through_the_server() -> None:
    mount = MCPMount([FIXTURE_CMD])
    try:
        tools = await mount.mount()
        echo = next(t for t in tools if t.name == "fixture__echo")
        assert await echo.fn(text="hello") == "hello"
    finally:
        await mount.close()


# -- e2e chat via FakeProvider through the mounted tool ------------------------


async def test_e2e_chat_calls_mounted_tool(tmp_path: Path) -> None:
    orc = Orchestrator(apps=[], mcp=[FIXTURE_CMD])
    rt = Runtime(
        orc,
        make_settings(tmp_path),
        providers={
            "standard": FakeProvider(
                [fake_tool_call("fixture__echo", {"text": "hi from fake"}), fake_text("done")]
            )
        },
    )
    await rt.start()
    try:
        assert "fixture__echo" in orc.registry.tools
        reply = await rt.chat("t:1", "echo hi", user_id="u1")
        assert reply.text == "done"
    finally:
        await rt.stop()


# -- taint through the mounted path --------------------------------------------


async def test_native_mutation_gated_after_mounted_readonly_mcp_call(tmp_path: Path) -> None:
    mutated: list[str] = []
    app = App("notes", description="d")

    @app.tool
    async def save_note(text: str) -> str:
        """Save a note (mutating)."""
        mutated.append(text)
        return "saved"

    orc = Orchestrator(apps=[app], mcp=[FIXTURE_CMD])
    rt = Runtime(
        orc,
        make_settings(tmp_path),
        providers={
            "standard": FakeProvider(
                [
                    fake_tool_call("fixture__echo", {"text": "hi"}, id="m1"),
                    fake_tool_call("save_note", {"text": "injected!"}, id="m2"),
                ]
            )
        },
    )
    await rt.start()
    try:
        reply = await rt.chat("t:1", "echo then save", user_id="u1")
        assert reply.pending_confirmation_id is not None  # gated!
        assert mutated == []
    finally:
        await rt.stop()


async def test_unannotated_mcp_tool_call_succeeds_and_still_taints(tmp_path: Path) -> None:
    """fixture__mutate carries no annotations, so Tool.read_only is False --
    it is treated as mutating. Calling it as the FIRST thing a clean turn
    does must still succeed: nothing untrusted has been seen yet, so there
    is nothing injected to act on, and gating here would confirm-prompt
    every mcp write. It must also taint the turn, so a following mutation
    -- native or mcp -- is gated."""
    mutated: list[str] = []
    app = App("notes", description="d")

    @app.tool
    async def save_note(text: str) -> str:
        """Save a note (mutating)."""
        mutated.append(text)
        return "saved"

    orc = Orchestrator(apps=[app], mcp=[FIXTURE_CMD])
    rt = Runtime(
        orc,
        make_settings(tmp_path),
        providers={
            "standard": FakeProvider(
                [
                    fake_tool_call("fixture__mutate", {"value": "x"}, id="m1"),
                    fake_tool_call("save_note", {"text": "sneaky"}, id="m2"),
                ]
            )
        },
    )
    await rt.start()
    try:
        reply = await rt.chat("t:1", "mutate then save", user_id="u1")
        assert reply.pending_confirmation_id is not None
        assert mutated == []
    finally:
        await rt.stop()


# -- unreachable command degrades: warning + skip, core unaffected ------------


async def test_unreachable_command_warns_and_yields_no_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mount = MCPMount(["nonexistent-cmd-xyz --flag"])
    tools = await mount.mount()
    assert tools == []
    await mount.close()  # must not raise even though nothing connected
    assert any("failed to mount" in r.message for r in caplog.records)


async def test_runtime_start_succeeds_with_unreachable_mcp_server(tmp_path: Path) -> None:
    orc = Orchestrator(apps=[], mcp=["nonexistent-cmd-xyz"])
    rt = Runtime(
        orc, make_settings(tmp_path), providers={"standard": FakeProvider([fake_text("hi")])}
    )
    await rt.start()  # must NOT raise -- broken mcp server never breaks core
    assert orc.registry.tools == {}
    reply = await rt.chat("t:1", "hi", user_id="u1")
    assert reply.text == "hi"
    await rt.stop()


async def test_mixed_reachable_and_unreachable_servers(tmp_path: Path) -> None:
    """One dead server alongside one healthy one: the healthy one's tools
    still mount; the dead one just contributes nothing."""
    orc = Orchestrator(apps=[], mcp=["nonexistent-cmd-xyz", FIXTURE_CMD])
    rt = Runtime(
        orc, make_settings(tmp_path), providers={"standard": FakeProvider([fake_text("hi")])}
    )
    await rt.start()
    try:
        assert set(orc.registry.tools) == {
            "fixture__echo",
            "fixture__mutate",
            "fixture__read_env",
        }
    finally:
        await rt.stop()


# -- name sanitization ----------------------------------------------------------


def test_sanitize_mcp_name_replaces_invalid_chars() -> None:
    assert sanitize_mcp_name("my server!") == "my_server_"
    assert sanitize_mcp_name("") == "_"
    assert sanitize_mcp_name("a-b_c9") == "a-b_c9"  # already valid, unchanged


def test_mcp_tool_name_joins_sanitizes_and_stays_in_pattern() -> None:
    name = mcp_tool_name("weird.server name", "weird tool/name")
    assert TOOL_NAME_RE.match(name)
    assert name == "weird_server_name__weird_tool_name"


def test_mcp_tool_name_truncates_to_64() -> None:
    long_name = mcp_tool_name("s" * 40, "t" * 40)
    assert len(long_name) == 64
    assert TOOL_NAME_RE.match(long_name)


async def test_registry_add_mcp_tools_rejects_native_collision() -> None:
    app = App("web", description="d")

    @app.tool
    async def fetch(url: str) -> str:
        """Fetch."""
        return "x"

    registry = Registry([app])

    async def fake_fn(**kwargs: object) -> str:
        return "x"

    mcp_tool = Tool(
        name="fetch",
        app_name="mcp:web",
        description="[experimental MCP] collide",
        fn=fake_fn,
        schema=McpToolSchema({"type": "object"}),
        read_only=True,
        confirm=False,
        timeout=30.0,
        origin="mcp",
    )
    with pytest.raises(RegistryError, match="collides"):
        registry.add_mcp_tools([mcp_tool])
    assert registry.tools["fetch"].origin == "native"  # unaffected by the rejected batch


def test_registry_add_mcp_tools_rejects_invalid_name() -> None:
    registry = Registry([])

    async def fake_fn(**kwargs: object) -> str:
        return "x"

    bad = Tool(
        name="bad name!",
        app_name="mcp:x",
        description="d",
        fn=fake_fn,
        schema=McpToolSchema({}),
        read_only=True,
        confirm=False,
        timeout=30.0,
        origin="mcp",
    )
    with pytest.raises(RegistryError, match="must match"):
        registry.add_mcp_tools([bad])
    assert registry.tools == {}


# -- schema adapter: thin passthrough, not signature-derived -------------------


def test_mcp_tool_schema_passes_input_schema_through_unchanged() -> None:
    raw = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}
    schema = McpToolSchema(raw)
    assert schema.json_schema == raw
    assert "additionalProperties" not in schema.json_schema  # server owns its schema, not forced


def test_mcp_tool_schema_validate_checks_required_and_dict_shape() -> None:
    schema = McpToolSchema({"type": "object", "required": ["a"], "properties": {"a": {}}})
    assert schema.validate({"a": 1, "extra": "allowed"}) == {"a": 1, "extra": "allowed"}
    with pytest.raises(ToolValidationError, match="missing required"):
        schema.validate({})
    with pytest.raises(ToolValidationError, match="expected an object"):
        schema.validate("not a dict")  # type: ignore[arg-type]


# -- server-side is_error -> raised so the router marks the result an error ---


async def test_call_fn_raises_runtime_error_on_server_side_is_error() -> None:
    import mcp.types as types

    class _FakeSession:
        async def call_tool(self, name: str, args: dict) -> types.CallToolResult:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="upstream boom")], isError=True
            )

    fn = _make_call_fn(_FakeSession(), remote_name="whatever", local_name="fixture__whatever")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="upstream boom"):
        await fn()


async def test_call_fn_joins_multiple_text_blocks() -> None:
    import mcp.types as types

    class _FakeSession:
        async def call_tool(self, name: str, args: dict) -> types.CallToolResult:
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text="a"),
                    types.TextContent(type="text", text="b"),
                ],
                isError=False,
            )

    fn = _make_call_fn(_FakeSession(), remote_name="whatever", local_name="local")  # type: ignore[arg-type]
    assert await fn() == "a\nb"


# -- refused callbacks: never a hang, always an immediate ErrorData -----------


async def test_server_initiated_callbacks_are_refused_immediately() -> None:
    import mcp.types as types

    sampling = await _refuse_sampling(None, None)  # type: ignore[arg-type]
    elicitation = await _refuse_elicitation(None, None)  # type: ignore[arg-type]
    roots = await _refuse_list_roots(None)  # type: ignore[arg-type]
    for result in (sampling, elicitation, roots):
        assert isinstance(result, types.ErrorData)
        assert "not supported in dudamel v1" in result.message


# -- close order must be reversed (LIFO), or a 2nd+ mounted server's ---------
# cancel-scope teardown raises CancelledError (a BaseException) that escapes
# `contextlib.suppress(Exception)`, poisoning the task and crashing
# Runtime.stop()/serve(). Repeated 3x in-test: this is a
# subprocess-timing-sensitive regression, so a flaky pass on run 1 that fails
# on run 2/3 would otherwise slip through.


async def test_mcp_mount_close_reversed_order_survives_two_servers() -> None:
    """anyio cancel scopes are stacked per-task, in mount order, across ALL
    mounted servers -- each server's own AsyncExitStack only guarantees LIFO
    *within itself*. Closing servers in forward mount order tries to exit a
    scope that is no longer innermost, and anyio raises CancelledError.
    fixture_cmd gives the two servers DISTINCT identities on purpose: this
    test isolates the close-order fix from the separate MCP-vs-MCP
    collision/dedupe policy exercised elsewhere."""
    for _ in range(3):
        mount = MCPMount([fixture_cmd("alpha"), fixture_cmd("beta")])
        tools = await mount.mount()
        assert {t.app_name for t in tools} == {"mcp:alpha", "mcp:beta"}
        await mount.close()  # must not raise


async def test_runtime_start_stop_survives_two_mounted_servers_repeatedly(
    tmp_path: Path,
) -> None:
    """Same regression at the actual reported crash site: Runtime.stop()
    (and by the same code path, serve()'s shutdown sequence)."""
    for _ in range(3):
        orc = Orchestrator(apps=[], mcp=[fixture_cmd("alpha"), fixture_cmd("beta")])
        rt = Runtime(
            orc, make_settings(tmp_path), providers={"standard": FakeProvider([fake_text("hi")])}
        )
        await rt.start()
        assert set(orc.registry.tools) == {
            "alpha__echo",
            "alpha__mutate",
            "alpha__read_env",
            "beta__echo",
            "beta__mutate",
            "beta__read_env",
        }
        await rt.stop()  # must not raise


# -- MCP-vs-MCP collisions warn + drop, never raise --------------------------
# (collision with a NATIVE tool stays fail-loud -- RegistryError, see the
# `test_registry_add_mcp_tools_rejects_native_collision` test above and the
# e2e version below).


async def test_truncation_collision_within_one_server_drops_second_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two tool names >=55 chars from the SAME server whose `{server}__{tool}`
    form collides after 64-char truncation: the first one mounted wins, the
    second is dropped with a warning -- an MCP server picking its own tool
    names must never be able to crash mounting via a naming accident."""
    import mcp.types as types

    long_a = "a" * 55 + "AAAA"
    long_b = "a" * 55 + "BBBB"
    # Sanity check: they DO collide once server+tool are joined and truncated.
    assert mcp_tool_name("fixture", long_a) == mcp_tool_name("fixture", long_b)
    tool_a = types.Tool(name=long_a, description="first", inputSchema={"type": "object"})
    tool_b = types.Tool(name=long_b, description="second", inputSchema={"type": "object"})

    caplog.set_level("WARNING")
    seen: set[str] = set()
    tools = _collect_server_tools(
        None,  # type: ignore[arg-type]  # never called: fn is a closure, not invoked here
        server_name="fixture",
        command="fixture-cmd",
        mcp_tools=[tool_a, tool_b],
        seen_names=seen,
    )
    assert len(tools) == 1
    assert tools[0].description.endswith("first")
    assert any("dropping this one" in r.message for r in caplog.records)


async def test_spoofed_server_identity_drops_colliding_tool_and_start_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two REAL servers reporting the SAME serverInfo.name (the simplest
    spoof: two instances of the same fixture, default name) produce
    identical `{server}__{tool}` names for their identical tool sets --
    first mount wins, second server's overlapping tools are dropped with a
    warning, and Runtime.start() succeeds instead of raising."""
    caplog.set_level("WARNING")
    orc = Orchestrator(apps=[], mcp=[FIXTURE_CMD, FIXTURE_CMD])
    rt = Runtime(
        orc, make_settings(tmp_path), providers={"standard": FakeProvider([fake_text("hi")])}
    )
    await rt.start()  # must NOT raise
    try:
        assert set(orc.registry.tools) == {"fixture__echo", "fixture__mutate", "fixture__read_env"}
        assert any("dropping this one" in r.message for r in caplog.records)
    finally:
        await rt.stop()


async def test_runtime_start_raises_registry_error_when_mcp_tool_collides_with_native(
    tmp_path: Path,
) -> None:
    """Unlike an MCP-vs-MCP collision, a mounted mcp tool colliding with a
    NATIVE tool stays fail-loud -- that's an operator configuration bug, not
    something an external MCP server should be able to shrug off."""
    app = App("web", description="d")

    @app.tool
    async def fixture__echo(text: str) -> str:
        """Collides on purpose with the mounted mcp tool's name."""
        return "x"

    orc = Orchestrator(apps=[app], mcp=[FIXTURE_CMD])
    rt = Runtime(
        orc, make_settings(tmp_path), providers={"standard": FakeProvider([fake_text("hi")])}
    )
    with pytest.raises(RegistryError, match="collides"):
        await rt.start()
    await rt.stop()


# -- env passthrough is explicit config, never ambient -----------------------


async def test_env_passthrough_forwards_configured_var_to_mcp_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUDAMEL_TEST_MCP_PASSTHROUGH_VAR", "secret-value")
    orc = Orchestrator(apps=[], mcp=[FIXTURE_CMD])
    settings = make_settings(
        tmp_path, mcp=McpConfig(env_passthrough=["DUDAMEL_TEST_MCP_PASSTHROUGH_VAR"])
    )
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([fake_text("hi")])})
    await rt.start()
    try:
        read_env = orc.registry.tools["fixture__read_env"]
        assert await read_env.fn(name="DUDAMEL_TEST_MCP_PASSTHROUGH_VAR") == "secret-value"
    finally:
        await rt.stop()


async def test_env_passthrough_absent_var_stays_absent_without_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUDAMEL_TEST_MCP_PASSTHROUGH_VAR", "secret-value")
    orc = Orchestrator(apps=[], mcp=[FIXTURE_CMD])
    settings = make_settings(tmp_path)  # McpConfig() default: nothing passed through
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([fake_text("hi")])})
    await rt.start()
    try:
        read_env = orc.registry.tools["fixture__read_env"]
        assert await read_env.fn(name="DUDAMEL_TEST_MCP_PASSTHROUGH_VAR") == ""
    finally:
        await rt.stop()


# -- post-mount max_tools ceiling drops mcp tools, never crashes -------------
# The ceiling exists because small models' tool selection collapses past a
# modest tool count. But a mounted server's tool count is not the operator's
# to control -- it is whatever that server advertises today -- so enforcing
# the ceiling by raising would let any server take the whole assistant down,
# which is precisely what this module promises can never happen. Excess
# mcp tools are therefore dropped with a warning. Native over-registration
# still raises, in Router.__init__: that IS the operator's own code.


async def test_mount_exceeding_max_tools_drops_mcp_tools_instead_of_crashing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    orc = Orchestrator(apps=[], mcp=[FIXTURE_CMD])  # fixture mounts 3 tools
    settings = make_settings(tmp_path, router=RouterConfig(max_tools=1))
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([fake_text("hi")])})
    with caplog.at_level(logging.WARNING, logger="dudamel.router"):
        await rt.start()
    try:
        assert len(orc.registry.tools) == 1
        assert all(t.origin == "mcp" for t in orc.registry.tools.values())
        assert "max_tools" in caplog.text
    finally:
        await rt.stop()


async def test_native_tools_are_never_dropped_for_mcp_tools(tmp_path: Path) -> None:
    """The operator's own app is what they actually asked for; a mounted
    server's tools are opportunistic. When the ceiling forces a choice, the
    native ones survive."""
    app = App("gym", description="d")

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record a workout."""
        return "ok"

    orc = Orchestrator(apps=[app], mcp=[FIXTURE_CMD])
    settings = make_settings(tmp_path, router=RouterConfig(max_tools=2))
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([fake_text("hi")])})
    await rt.start()
    try:
        names = set(orc.registry.tools)
        assert "log_workout" in names
        assert len(names) == 2
    finally:
        await rt.stop()


# -- advertised schema/description caps --------------------------------------
# Both are embedded verbatim in every LLM request, so both are a token-budget
# cost AND a prompt-injection channel that fires at advertise time, before any
# tool call -- the taint gate that protects tool *results* never engages here.
# A schema is dropped (a truncated JSON Schema is not a schema); a description
# is truncated (a truncated description is still a description).


def _mcp_tool(name: str, *, schema: dict | None = None, description: str = "d") -> types.Tool:
    return types.Tool(
        name=name,
        description=description,
        inputSchema=schema if schema is not None else {"type": "object", "properties": {}},
    )


def test_oversized_input_schema_drops_the_tool_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    huge = {"type": "object", "properties": {"x": {"description": "y" * 20000}}}
    seen: set[str] = set()
    with caplog.at_level(logging.WARNING):
        tools = _collect_server_tools(
            None,  # type: ignore[arg-type]
            server_name="fixture",
            command="cmd",
            mcp_tools=[_mcp_tool("big", schema=huge), _mcp_tool("small")],
            seen_names=seen,
        )
    assert [t.name for t in tools] == ["fixture__small"]
    assert "big" in caplog.text


def test_dropped_oversized_tool_does_not_burn_its_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool dropped for an oversized schema must not claim its name, so a
    later same-named tool from another server can still mount."""
    huge = {"type": "object", "properties": {"x": {"description": "y" * 20000}}}
    seen: set[str] = set()
    with caplog.at_level(logging.WARNING):
        _collect_server_tools(
            None,  # type: ignore[arg-type]
            server_name="fixture",
            command="cmd",
            mcp_tools=[_mcp_tool("dup", schema=huge)],
            seen_names=seen,
        )
    assert seen == set()


def test_oversized_description_is_truncated_not_dropped() -> None:
    seen: set[str] = set()
    tools = _collect_server_tools(
        None,  # type: ignore[arg-type]
        server_name="fixture",
        command="cmd",
        mcp_tools=[_mcp_tool("chatty", description="z" * 5000)],
        seen_names=seen,
    )
    assert len(tools) == 1
    assert len(tools[0].description) < 1200
    assert tools[0].description.startswith("[experimental MCP] zzz")


def test_unserializable_schema_drops_the_tool_rather_than_crashing() -> None:
    """A hostile deeply-nested schema must drop one tool, not kill the mount."""
    nested: dict = {"type": "object"}
    cursor = nested
    for _ in range(5000):
        cursor["items"] = {"type": "object"}
        cursor = cursor["items"]
    seen: set[str] = set()
    tools = _collect_server_tools(
        None,  # type: ignore[arg-type]
        server_name="fixture",
        command="cmd",
        mcp_tools=[_mcp_tool("nested", schema=nested), _mcp_tool("fine")],
        seen_names=seen,
    )
    assert [t.name for t in tools] == ["fixture__fine"]
