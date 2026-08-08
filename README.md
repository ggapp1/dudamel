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

## Remote access

The dashboard binds to `127.0.0.1` by default: reachable only from the
machine it runs on. Two ways to reach it from elsewhere:

- **Tailscale (recommended)** — put the machine on your tailnet and reach
  the dashboard at its Tailscale address; `dudamel doctor` detects a
  running Tailscale client and prints the address to use. No inbound port
  needs to be opened, and traffic never leaves your own mesh network.
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

## Security

- **Confirm gates**: a tool registered with `confirm=True` always stops
  and returns a `pending_confirmation_id` before running, regardless of
  what the model asked for. The same user has to approve it explicitly.
- **The taint rule**: tool output is treated as data, not instruction. Once
  a turn has seen a result from a less-trusted source (an MCP tool), any
  further mutating tool call in that turn — native or MCP — that isn't
  marked `read_only=True` is forced through a confirm gate too, even if it
  wasn't registered with `confirm=True`. A tool with no safety annotation
  defaults to "mutating" until proven otherwise.
- **Token budgets**: `[llm.budget] daily_tokens` in `dudamel.toml` sets a
  hard per-day ceiling per tier, enforced before each call — a runaway
  loop or a misbehaving job can't spend past it.
- Every `/api/*` route requires a bearer token or an authenticated session
  cookie; `/health` is intentionally unauthenticated (it exists for
  infrastructure checks) and never returns anything beyond up/down status.
- dudamel refuses to bind the dashboard to a non-loopback host unless a web
  token is configured, so it can't end up unauthenticated on your network
  by accident.

## MCP (experimental)

dudamel can mount external [MCP](https://modelcontextprotocol.io) servers
as additional tools, configured in code alongside your apps:

```python
orchestrator = Orchestrator(apps=[workouts_app], mcp=["npx -y @some/mcp-server"])
```

MCP support is **experimental**. Mounted tools are treated as less trusted
than native ones: their results feed the taint rule described above, tool
names are sanitized and namespaced (`{server}__{tool}`), and a server that
fails to start or doesn't speak the protocol correctly is skipped with a
warning rather than blocking the rest of the assistant from starting.
Only the stdio transport is supported; a mounted server asking dudamel for
sampling, elicitation, or roots gets an explicit refusal rather than a
hang or a silent no-op. If a mounted server pushes the tool count past
`[router] max_tools`, the excess mcp tools — even ones you explicitly
configured — are dropped from the model's tool list with only a log line,
rather than failing startup.

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
