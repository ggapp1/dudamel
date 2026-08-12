# dudamel

A framework for running a personal AI assistant on your own hardware: apps
are ordinary typed Python, safety rules are enforced by the framework
instead of relied on from the model, and the dashboard keeps working even
when the model doesn't.

## What dudamel is

dudamel splits an assistant into two planes. The **command plane** is the
chat loop: a message comes in, the model picks tools to call, dudamel runs
them and feeds results back until it has an answer. The **data plane** —
the dashboard, widgets, scheduled jobs, database migrations — does not
depend on the model at all. If your LLM endpoint is down, the dashboard
still renders, because widgets are plain `async def` functions that read
the database directly; nothing about them calls the model.

Apps are code, not configuration:

- a **tool** is an `async def` with type hints — its signature becomes the
  JSON schema the model is offered, its docstring becomes the description
  the model sees;
- a **widget** is an `async def` returning a small typed payload (a stat, a
  table, or markdown) that the dashboard renders;
- a **job** is a scheduled `async def` (cron or interval) that can call the
  model and push a notification, independent of any chat turn.

Safety is enforced in front of every tool call, not left to a system
prompt: **confirm gates** stop and wait for explicit approval before
running anything marked as needing it; the **taint rule** forces the same
approval step onto mutating tools once a turn has seen output from
a less-trusted source (currently: MCP); and per-tier **token budgets** are
checked server-side before each model call, not just documented as a
limit.

## Quickstart

```
uvx dudamel new my-assistant
cd my-assistant
uv run dudamel db migrate -m init
uv run dudamel run
```

`dudamel new` scaffolds a project with one example app already wired up
(`workouts`, shown in full below), a `dudamel.toml`, and a generated web
token in `.env`. The scaffold is local-first by default — both LLM tiers
point at a local Ollama server, so the whole thing runs offline once
you've pulled a model:

```toml
[llm.tiers.standard]
provider = "openai-compatible"
base_url = "http://localhost:11434/v1"
model = "qwen3.5:9b"

[llm.tiers.fast]
provider = "openai-compatible"
base_url = "http://localhost:11434/v1"
model = "qwen3.5:1.5b"
```

Point a tier at a hosted provider instead by setting `provider =
"anthropic"` and an API key env var. `uv run dudamel doctor` checks every
configured tier is actually reachable, along with the database, migration
state, and web/Telegram token configuration.

The dashboard comes up at `http://127.0.0.1:8787`. Log in with the token
`dudamel new` generated in `.env` (`DUDAMEL_WEB_TOKEN`); rotate it any time
with `uv run dudamel token rotate`, which rewrites only that one line.

### The whole example app

This is the complete `workouts` app the scaffold ships — a database model,
a tool, a widget, and a scheduled job:

```python
from datetime import datetime

from dudamel import App

app = App("workouts", description="Log and review gym workouts")


class WorkoutSet(app.Model, table="sets"):
    exercise: str
    sets: int
    reps: int
    weight_kg: float
    logged_at: datetime = app.now()


@app.tool
async def log_workout(exercise: str, sets: int, reps: int, weight_kg: float) -> str:
    """Record one exercise from today's session."""
    async with app.db() as db:
        db.add(WorkoutSet(exercise=exercise, sets=sets, reps=reps, weight_kg=weight_kg))
    return f"Logged: {exercise} {sets}x{reps} @ {weight_kg}kg"


@app.widget(title="This week", renderer="stat")
async def week_volume() -> dict:
    async with app.db() as db:
        from sqlalchemy import func, select

        total = (await db.execute(select(func.sum(WorkoutSet.weight_kg)))).scalar() or 0
    return {"label": "Weekly volume", "value": total, "unit": "kg"}


@app.job(cron="0 20 * * *")
async def evening_summary() -> None:
    text = await app.llm("Summarize today's training", tier="fast")
    await app.notify(text)
```

That's the entire app: a typed model, a tool the LLM can call, a widget the
dashboard renders without touching the model at all, and a nightly job
that summarizes the day and sends a notification. `dudamel db migrate`
picks the model up automatically — no separate schema file.

## Backends without native tool calling

Some local models — especially smaller ones served through an
OpenAI-compatible endpoint — either reject a request that includes `tools`
or accept it and never emit a tool call. Set `tool_calling = "prompted"` on
that tier:

```toml
[llm.tiers.standard]
provider = "openai-compatible"
base_url = "http://localhost:11434/v1"
model = "some-small-model"
tool_calling = "prompted"
```

This replaces the native wire format with a prompted fallback: tool
descriptions and prior tool results are flattened into plain prompt text,
and the model is instructed to reply with a small JSON envelope instead of
a native tool call. Run `uv run dudamel doctor --probe-tools` after
changing this — the probe wraps the same way the router will, so a passing
probe means the prompted round-trip actually works for that model, not
just that native tool calling is (as expected) absent.

Trust caveats:

- Tool *selection* is exactly as gated as the native path — a parsed call
  still has to name a tool the registry knows about, and an MCP-origin
  tool still taints the turn the same way whether the call arrived natively
  or was parsed from prompted text. This setting changes how a call is
  *recognized*, not what an accepted call is allowed to do.
- A pending confirmation can never be approved by anything the model
  outputs, prompted or native — approval is a separate, out-of-band step
  the user takes through an interface, not something reachable from parsed
  text.
- The fallback is more permissive about malformed output than a native
  backend would ever produce: expect it to occasionally reply in plain
  prose instead of calling a tool it should have. That degrades to a
  normal text reply, never a crash. If instead it emits a well-formed
  envelope that asks for nothing runnable — an empty call list, or entries
  missing a tool name — the reply you see is a short neutral apology rather
  than the model's raw JSON; the envelope and the reason are logged at
  WARNING for the operator.

## Remote access

The dashboard binds to `127.0.0.1` by default: reachable only from the
machine it runs on. Two ways to reach it from elsewhere:

- **Tailscale (recommended)** — put the machine on your tailnet and reach
  the dashboard at its Tailscale address; `dudamel doctor` detects a
  running Tailscale client and prints the address to use. No inbound port
  needs to be opened, and traffic never leaves your own mesh network.
  `dudamel doctor` prints that address as `http://…`, and reaching the
  dashboard over plain HTTP at a non-loopback address is exactly the case
  `cookie_secure`'s auto-derivation (see below) marks `Secure` — a browser
  will not store a `Secure` cookie on a plain-HTTP, non-localhost origin,
  which shows up as the login page redirecting back to itself instead of
  an error. If you're reaching the dashboard this way, set `[web]
  cookie_secure = false` in `dudamel.toml`.
- **Telegram** — works from anywhere with zero network configuration,
  because dudamel's Telegram interface polls Telegram's API outward rather
  than listening for inbound connections. Set `DUDAMEL_TELEGRAM_TOKEN` and
  an allowed-user-id list in `dudamel.toml` and it works from behind any
  NAT, on cellular, wherever.

Forwarding a port on your router to expose the dashboard directly to the
public internet is not recommended — it puts the dashboard's auth layer
directly in front of arbitrary internet traffic, which Tailscale and
Telegram's outbound polling both avoid needing to do at all.

Running a reverse proxy (nginx, Caddy) in front of the dashboard on the
same host needs one more setting: list its address in `[web]
trusted_proxies` in `dudamel.toml`, or dudamel ignores `X-Forwarded-For`
entirely and every request looks like it came from the proxy itself
instead of the real client.

### HTTPS behind a reverse proxy

If the proxy terminates TLS and forwards to the dashboard over plain HTTP
on the same host, a few settings need attention:

- **`cookie_secure`** — `[web] cookie_secure` in `dudamel.toml` is `None`
  (auto) by default: the session cookie gets the `Secure` flag whenever the
  configured `[web] host` isn't a loopback address, and stays plain on
  `127.0.0.1` / `localhost` / `::1` since those are secure contexts to a
  browser even over HTTP. Setting `cookie_secure` explicitly always wins
  over that derivation, in either direction. When `Secure` is on, the
  cookie is also renamed to `__Host-dudamel_session` — a browser-enforced
  prefix that refuses the cookie unless it's `Secure`, scoped to `Path=/`,
  and carries no `Domain` attribute. The derivation looks at `[web] host`,
  not at what the browser sees: a proxy that terminates TLS and forwards
  to a loopback bind leaves `[web] host` at `127.0.0.1`, so the
  auto-derivation resolves to `False` even though the deployment is
  HTTPS end-to-end. In that setup, set `cookie_secure = true` explicitly.
  `dudamel doctor` prints the resolved posture — the value, whether it was
  set explicitly or derived, and (when derived) the remedy for the topology
  it looks wrong for — right under the dashboard URL it judges it against.
- **`trusted_proxies`** — only ever list a proxy's address here if you
  actually run it: anything in this list is a peer dudamel will believe
  can claim any client address via `X-Forwarded-For`. Leaving it empty
  (the default) means forwarded headers are ignored and requests are
  attributed to whichever address actually opened the connection — the
  proxy itself, not its clients.
- **HSTS** — dudamel does not set `Strict-Transport-Security` itself; that
  header belongs to whatever terminates TLS. Set it on the proxy.
- **`allowed_hosts`** — `[web] allowed_hosts` defaults to `["localhost",
  "127.0.0.1"]`. Every request's `Host` header is checked against this
  list; a request whose `Host` isn't on it gets a `400` before it reaches
  any route, `/health` included. A proxy forwarding its own external
  hostname will 400 every single request until that hostname is added to
  `allowed_hosts`. Do not set it to `["*"]` to make the 400s go away —
  that disables the host check entirely and reopens the DNS-rebinding
  attack it exists to close; add the specific hostname(s) instead.

### Running as a service

A personal assistant is only useful if it's actually running. `dudamel new`
writes a launchd plist and a systemd user unit into `<project>/deploy/`,
both with the project's own path already filled in:

- **macOS** — `deploy/dudamel.plist`, loaded with `launchctl load
  ~/Library/LaunchAgents/dudamel.plist` (after copying it there). Sleep
  stops everything, though: run `caffeinate` while it matters, or enable
  Power Nap / "Wake for network access" in Energy Saver settings so
  scheduled jobs and Telegram still fire while the machine is asleep.
- **Linux** — `deploy/dudamel.service`, a systemd `--user` unit with
  `Restart=always`; enable it with `systemctl --user enable --now
  dudamel.service` and run `loginctl enable-linger $USER` so it keeps
  running after you log out.

Both templates restart `dudamel run` automatically if it exits — see the
comments at the top of each file for the exact install steps.

`auto_migrate` (a top-level key in `dudamel.toml`, default `true`) controls
whether `dudamel run` upgrades the database schema in place on every
startup. That's convenient for local development, but a service that
restarts on its own — a crash, a `systemctl --user restart`, a machine
reboot — should not be able to mutate the schema just by starting up. Set
`auto_migrate = false` for a production deployment. With it off, a
startup against a schema that's behind its migration scripts refuses to
start instead of upgrading, and tells you to run `dudamel db migrate -m
<message>` — which both autogenerates the pending migration and applies
it (core and app schemas), not apply-only.

App migrations resolve from `project_dir` (a `dudamel.toml`/`Settings` key,
default the current directory) — the same `migrations/` that `dudamel db
migrate` writes — not from `data_dir` (where the SQLite database and other
runtime state live). Keep them the same unless you deliberately split them:
migrations placed under `data_dir` are ignored.

## Security

- **Confirm gates**: a tool registered with `confirm=True` always stops
  and returns a `pending_confirmation_id` before running, regardless of
  what the model asked for. The same user has to approve it explicitly.
- **The taint rule**: tool output is treated as data, not instruction. Once
  a turn has seen a result from a less-trusted source (an MCP tool), any
  further mutating tool call in that turn — native or MCP — that isn't
  marked `read_only=True` is forced through a confirm gate too, even if it
  wasn't registered with `confirm=True`. A tool with no safety annotation
  defaults to "mutating" until proven otherwise. Approving a gated MCP call
  taints the rest of that turn as well — its output is untrusted whether it
  arrived through the gate or not. Where taint is re-derived from history
  (`taint_mode = "window"`, and a summary's stored taint flag), a call whose
  tool the registry can no longer resolve — an MCP server dropped from the
  config, a renamed tool, or a name the model invented — counts as
  untrusted: unknown provenance is not trusted provenance.
- **Token budgets**: `[llm.budget] daily_tokens` in `dudamel.toml` sets a
  hard per-day ceiling per tier, enforced before each call — a runaway
  loop or a misbehaving job can't spend past it.
- **Persona**: `[router] persona` in `dudamel.toml` customizes the assistant's
  system prompt identity line (the "You are..." part). The installed apps
  block and tool-use instructions are always present — a persona cannot
  disable tool use by accident. Leave unset for the default identity.
- Every `/api/*` route requires a bearer token or an authenticated session
  cookie; `/health` is intentionally unauthenticated (it exists for
  infrastructure checks) and never returns anything beyond up/down status.
- dudamel refuses to bind the dashboard to a non-loopback host unless a web
  token is configured, so it can't end up unauthenticated on your network
  by accident.

## Conversation compaction (opt-in)

Every turn assembles a context window from the conversation's history under
a token budget (`[router] window_tokens`); once a long-running conversation
doesn't fit, the oldest turns are cut, always at turn boundaries, and the
model simply never sees them. `[router] compact_dropped_turns` (default
`false`) turns on a summarizer that condenses the turns a window build is
about to drop into one row in the `summaries` table, so the assistant keeps
"the gist" of a long conversation instead of silently forgetting it.
Turning it on requires `[router] compaction_tier`, naming one of
`[llm.tiers]` — dudamel refuses to start if it's missing or names a tier
that isn't configured.

The summary is prepended to the window as a `role="user"` message, worded
as background context rather than as an instruction — never as
`role="system"`, because the anthropic provider folds every system message
into the single top-level `system` request parameter, alongside the
operator's own instructions, and a summary of the conversation's own
history (which can include text an MCP tool put into it) does not belong
there. Its token cost is subtracted from the budget handed to the window
builder, so compaction never adds headroom pressure beyond an uncompacted
turn — it does not guarantee the model call stays within `window_tokens`
overall: the window builder always includes the newest turn even when
that turn alone exceeds whatever budget it's given, with or without a
summary competing for the same budget.
Summarization runs at most once per turn — not once per loop iteration —
and reuses the newest summary already covering the dropped span instead of
calling the model again. A summarization failure (including a budget
error) is logged and the turn proceeds with the uncompacted window; it
never fails the turn.

**Cost**: each summarizer call condenses the whole dropped span from
scratch, not "the previous summary plus the new turns". Once a
conversation has outgrown its budget the span grows every turn, so
compaction adds one model call per turn whose prompt grows with the span:
measured on a simulated steady state, roughly 2k characters of prompt by
turn 10 and 16k by turn 60 with small tool results, and ~66k and ~479k
respectively when tool results run at the full `tool_result_cap`. The
200-message horizon bounds the worst case near 1.6 MB (~400k tokens),
which can exceed a provider's context limit — that failure is logged and
the turn proceeds uncompacted, like any other summarizer failure.

A summarized turn's taint (whether it saw output from a less-trusted MCP
tool) is computed from the summarized rows' own provenance, never from the
summarizer's output, and is carried forward: a new turn seeds its taint
state from the newest summary's flag, under every `taint_mode` except
`"off"`, including the default `"turn"` mode.

**Scope**: `ConversationStore.recent()` reads at most the newest 200
messages in a conversation. Compaction only ever sees, and only ever
covers, that same window — anything older than the 200-message horizon is
gone regardless of whether compaction is enabled.

## MCP (experimental)

dudamel can mount external [MCP](https://modelcontextprotocol.io) servers
as additional tools, configured in code alongside your apps. A plain string
is a stdio command, launched as a subprocess:

```python
orchestrator = Orchestrator(apps=[workouts_app], mcp=["npx -y @some/mcp-server"])
```

For an HTTP server, or a stdio server that needs its own environment
variables, use `MCPServerConfig` instead of a string:

```python
from dudamel import MCPServerConfig

orchestrator = Orchestrator(
    apps=[workouts_app],
    mcp=[
        MCPServerConfig(
            url="https://mcp.example.com/mcp",
            headers={"Authorization": f"Bearer {token}"},
        ),
    ],
)
```

`MCPServerConfig` takes exactly one of `command` (a stdio command string)
or `url` (a streamable-HTTP endpoint) — never both, never neither.
`headers` are sent with every HTTP request, which is the only way to reach
an authenticated server: the underlying client sends no headers of its
own, so a bare URL string can't carry a token. `env` names environment
variables (already set in your own process) to pass through to a stdio
server's subprocess, per server — this replaces the plain-string form's
global `[mcp] env_passthrough` list for that one server.

MCP support is **experimental**. Mounted tools are treated as less trusted
than native ones: their results feed the taint rule described above, tool
names are sanitized and namespaced (`{server}__{tool}`), and a server that
fails to start or doesn't speak the protocol correctly is skipped with a
warning rather than blocking the rest of the assistant from starting. A
tool a server annotates `destructiveHint: true` is registered with
`confirm=True`, so it gets a confirm prompt on every call, the same as a
native tool registered that way. A mounted server asking dudamel for
sampling, elicitation, or roots gets an explicit refusal rather than a
hang or a silent no-op. If a mounted server pushes the tool count past
`[router] max_tools`, nothing is dropped permanently: every native tool
stays offered on every turn, and each turn instead picks a relevant
subset of the mcp-origin tools — ranked by overlap between the tool's
name/description and the current message, plus any mcp tool already
called earlier in the same turn — to fill the remaining slots. Startup
still refuses to start if *native* tool registration alone exceeds
`max_tools`, since that's the operator's own code to fix; a busy mcp
mount just means a given turn won't see every mounted tool, logged once
at mount time and once per turn when a tool is left out — a turn that
suspends on a confirm prompt and later resumes logs at most once for each
half, since the resumed half can be offered a different subset.

A server whose connection dies is reconnected automatically, but only
within limits: a bounded number of attempts with growing backoff. If that
burst is exhausted the server's tools fail fast for a cooldown period
rather than being disabled for the rest of the process — the next call
after the cooldown gets a fresh burst, because a real deployment being
restarted or rolled routinely takes longer to come back than one burst
spans. If a **mutating** tool's call dies mid-flight — the connection drops,
or the call outlives `call_timeout` — dudamel reports the outcome as
**unknown** rather than as a failure: the side effect may already have
happened, and reporting failure would invite a retry that runs it twice.
That wording is aimed at the model, so what actually keeps one approval to
one execution is the confirm gate in front of it, and that gate is
conditional: a tool the server annotates `destructiveHint: true` is always
gated, but an unannotated one is gated only once MCP output has already
been seen (in the turn, or in the window — that's what `[router]
taint_mode` selects), and not at all under `taint_mode = "off"`. Ungated,
the model is free to call the tool again on its own.

Two `[mcp]` settings in `dudamel.toml` control how long this is allowed to
take: `call_timeout` (default 30 seconds) bounds a single tool call, and
`mount_timeout` (default 15 seconds) bounds connecting to and listing
tools from a server at startup and on each reconnect. Both must be
positive; anything else is rejected at startup.

Three more size the reconnect burst itself: `reconnect_attempts` (default
3) is how many connection attempts one burst may spend,
`reconnect_backoff_seconds` (default 0.5) is the delay before the second
attempt and doubles for each one after it, and
`reconnect_cooldown_seconds` (default 60) is how long that server's tools
fail fast after a burst is exhausted, before the next call earns a fresh
burst. All three must be positive; anything else is rejected at startup.

## Breaking changes for next release

- `[llm.budget] daily_usd` has been removed and is no longer accepted.
  If your `dudamel.toml` sets this key, it will raise a validation error
  on startup. Use `daily_tokens` for budget enforcement — it is the only
  enforced limit in v1 and later.
- An MCP tool a server annotates `destructiveHint: true` is now
  registered with `confirm=True` automatically, so it gets a confirm
  prompt on every call the same as a native tool registered that way.
  Previously this annotation was ignored. If you rely on such a tool
  running without a confirmation step, register it explicitly or adjust
  the server's annotations.

## Testing your apps

`dudamel.llm.testing.FakeProvider` scripts a model's responses so tool
flows can be tested deterministically, without a real model running:

```python
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call

provider = FakeProvider([
    fake_tool_call("log_workout", {"exercise": "bench", "sets": 3, "reps": 5, "weight_kg": 100}),
    fake_text("Logged it."),
])
```

Hand `provider` to `Runtime`/`serve()` in place of a real one and drive
your app's tools, widgets, and jobs against a real database (typically
`tmp_path` and SQLite) — no network calls, no model, fully deterministic
output to assert against.

## Why "dudamel"?

[check this out](https://www.youtube.com/watch?v=jfDprp0NlQ4)

## License

MIT — see [LICENSE](LICENSE).
