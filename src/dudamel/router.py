"""The command plane's tool-calling loop. Defensive by design: small local
models misbehave, so every malformed thing they emit is fed back to them as
an error tool-result instead of crashing the turn."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from dudamel.activity import json_safe, log_activity
from dudamel.config import RouterConfig
from dudamel.contract.types import Tool
from dudamel.convo import ConversationStore
from dudamel.db import Database
from dudamel.exceptions import (
    BudgetExceededError,
    LLMError,
    RegistryError,
    ToolValidationError,
)
from dudamel.llm.client import LLMClient
from dudamel.llm.provider import ToolSpec
from dudamel.llm.types import Message, ToolCall
from dudamel.models_core import Message as MessageRow
from dudamel.models_core import PendingConfirmation
from dudamel.registry import Registry
from dudamel.window import build_window

logger = logging.getLogger("dudamel.router")


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class ChatReply:
    text: str
    pending_confirmation_id: str | None = None


@dataclass
class _BatchOutcome:
    results: list[Message] = field(default_factory=list)  # completed + skipped
    pending_call: ToolCall | None = None  # first confirm-gated call, if any
    executed_any: bool = False
    saw_mcp_result: bool = False


class Router:
    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: Registry,
        convo: ConversationStore,
        db: Database,
        config: RouterConfig,
    ) -> None:
        if len(registry.tools) > config.max_tools:
            raise RegistryError(
                f"{len(registry.tools)} tools registered but router max_tools is "
                f"{config.max_tools} — small models' tool selection collapses beyond "
                "this; raise [router].max_tools deliberately or split apps"
            )
        self._llm = llm
        self._registry = registry
        self._convo = convo
        self._db = db
        self._config = config
        self._specs = [ToolSpec.from_tool(t) for t in registry.tools.values()]
        self._locks: dict[int, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # -- public ---------------------------------------------------------------
    async def handle(
        self,
        *,
        channel: str,
        text: str,
        user_id: str,
        client_msg_id: str | None = None,
        tier: str = "standard",
    ) -> ChatReply:
        conv_id = await self._convo.get_or_create(channel)
        async with await self._lock_for(conv_id):
            if client_msg_id is not None:
                # Read-only duplicate check BEFORE auto-decline: an interface
                # retry of the very message that created a pending confirmation
                # must not auto-decline its own confirmation just to be deduped
                # a moment later. The append-dedupe below remains the backstop
                # for racing concurrent writes of the same client_msg_id.
                async with self._db.session() as s:
                    dup = (
                        await s.execute(
                            select(MessageRow.id).where(
                                MessageRow.conversation_id == conv_id,
                                MessageRow.client_msg_id == client_msg_id,
                            )
                        )
                    ).first()
                if dup is not None:
                    return ChatReply(text="")
            await self._auto_decline_pending(conv_id)
            appended = await self._convo.append(
                conv_id, Message(role="user", text=text), client_msg_id=client_msg_id
            )
            if not appended:
                return ChatReply(text="")  # duplicate delivery; interfaces drop empties
            return await self._loop(
                conv_id,
                tier=tier,
                user_id=user_id,
                start_iteration=0,
                executed_any=False,
            )

    # -- internals ------------------------------------------------------------
    async def _lock_for(self, conv_id: int) -> asyncio.Lock:
        async with self._locks_guard:
            if conv_id not in self._locks:
                self._locks[conv_id] = asyncio.Lock()
            return self._locks[conv_id]

    def _system_message(self) -> Message:
        apps = "\n".join(f"- {a.name}: {a.description}" for a in self._registry.apps.values())
        return Message(
            role="system",
            text=(
                "You are dudamel, a personal assistant orchestrator.\n"
                f"Installed apps:\n{apps}\n"
                "Use the available tools to act or fetch data; otherwise answer "
                "directly and concisely."
            ),
        )

    async def _loop(
        self,
        conv_id: int,
        *,
        tier: str,
        user_id: str,
        start_iteration: int,
        executed_any: bool,
        initial_taint: bool = False,
    ) -> ChatReply:
        # DEVIATION (Task 11 review): resumes seed taint from the suspended
        # turn's stored flag so MCP taint survives the suspension gap.
        turn_tainted = initial_taint
        for iteration in range(start_iteration, self._config.iteration_cap):
            history = await self._convo.recent(conv_id)
            window = [self._system_message()] + build_window(
                history,
                token_budget=self._config.window_tokens,
                tool_result_cap=self._config.tool_result_cap,
            )
            dropped = len(history) - (len(window) - 1)
            if dropped > 0:
                # spec: truncation is surfaced, never silent
                logger.info(
                    "conversation %s: context window dropped %d older messages",
                    conv_id,
                    dropped,
                )
            if self._config.taint_mode == "window":
                turn_tainted = turn_tainted or self._window_tainted(window)
            try:
                completion = await self._llm.complete(
                    window, tier=tier, tools=self._specs, conversation_id=conv_id
                )
            except BudgetExceededError as e:
                return ChatReply(text=str(e))
            except LLMError as e:
                if executed_any:
                    return ChatReply(
                        text=f"I completed the action(s), but couldn't produce a "
                        f"summary — the model failed afterwards ({e})."
                    )
                return ChatReply(text=f"The model is unavailable: {e}")
            msg = completion.message
            if not msg.tool_calls:
                await self._convo.append(conv_id, msg)
                return ChatReply(text=msg.text or "(no reply)")
            outcome = await self._execute_batch(conv_id, msg.tool_calls, turn_tainted=turn_tainted)
            executed_any = executed_any or outcome.executed_any
            turn_tainted = turn_tainted or outcome.saw_mcp_result
            if outcome.pending_call is not None:
                return await self._suspend(
                    conv_id,
                    assistant=msg,
                    outcome=outcome,
                    iteration=iteration,
                    tier=tier,
                    user_id=user_id,
                    executed_any=executed_any,
                    turn_tainted=turn_tainted,
                )
            await self._convo.append(conv_id, msg)
            for r in outcome.results:
                await self._convo.append(conv_id, r)
        return ChatReply(
            text="I couldn't finish within the step limit — the request may be "
            "too complex; try narrowing it."
        )

    def _window_tainted(self, window: list[Message]) -> bool:
        for m in window:
            for tc in m.tool_calls:
                tool = self._registry.tools.get(tc.name)
                if tool is not None and tool.origin == "mcp":
                    return True
        return False

    def _needs_confirm(self, tool: Tool, *, turn_tainted: bool, batch_has_mcp: bool) -> bool:
        if tool.confirm:
            return True
        if self._config.taint_mode == "off":
            return False
        return tool.origin == "native" and not tool.read_only and (turn_tainted or batch_has_mcp)

    async def _execute_batch(
        self, conv_id: int, calls: list[ToolCall], *, turn_tainted: bool
    ) -> _BatchOutcome:
        outcome = _BatchOutcome()
        batch_has_mcp = any(
            (t := self._registry.tools.get(c.name)) is not None and t.origin == "mcp" for c in calls
        )
        plan: list[tuple[ToolCall, str]] = []  # (call, action: run|pending|skip|...)
        for call in calls:
            tool = self._registry.tools.get(call.name)
            if tool is None:
                plan.append((call, "unknown"))
            elif self._needs_confirm(tool, turn_tainted=turn_tainted, batch_has_mcp=batch_has_mcp):
                if outcome.pending_call is None:
                    outcome.pending_call = call
                    plan.append((call, "pending"))
                else:
                    plan.append((call, "skip"))
            else:
                plan.append((call, "run"))

        async def run_one(call: ToolCall) -> Message:
            tool = self._registry.tools[call.name]
            try:
                kwargs = tool.schema.validate(call.args)
            except ToolValidationError as e:
                await log_activity(
                    self._db,
                    tool=call.name,
                    args=call.args,
                    status="error",
                    result_preview=str(e),
                    conversation_id=conv_id,
                )
                return Message(role="tool", text=str(e), tool_call_id=call.id, is_error=True)
            try:
                result = await asyncio.wait_for(tool.fn(**kwargs), tool.timeout)
            except TimeoutError:
                detail = f"tool {call.name} timed out after {tool.timeout}s"
                await log_activity(
                    self._db,
                    tool=call.name,
                    args=call.args,
                    status="error",
                    result_preview=detail,
                    conversation_id=conv_id,
                )
                return Message(role="tool", text=detail, tool_call_id=call.id, is_error=True)
            except Exception as e:  # tool bugs must not kill the conversation
                detail = f"tool {call.name} raised {type(e).__name__}: {e}"
                await log_activity(
                    self._db,
                    tool=call.name,
                    args=call.args,
                    status="error",
                    result_preview=detail,
                    conversation_id=conv_id,
                )
                return Message(role="tool", text=detail, tool_call_id=call.id, is_error=True)
            text = result if isinstance(result, str) else json.dumps(json_safe(result))
            await log_activity(
                self._db,
                tool=call.name,
                args=call.args,
                status="ok",
                result_preview=text,
                conversation_id=conv_id,
            )
            outcome.executed_any = True
            if self._registry.tools[call.name].origin == "mcp":
                outcome.saw_mcp_result = True
            return Message(role="tool", text=text, tool_call_id=call.id)

        tasks = {
            idx: asyncio.create_task(run_one(call))
            for idx, (call, action) in enumerate(plan)
            if action == "run"
        }
        for idx, (call, action) in enumerate(plan):
            if action == "run":
                outcome.results.append(await tasks[idx])
            elif action == "unknown":
                available = ", ".join(sorted(self._registry.tools))
                outcome.results.append(
                    Message(
                        role="tool",
                        text=f"unknown tool {call.name!r}; available tools: {available}",
                        tool_call_id=call.id,
                        is_error=True,
                    )
                )
            elif action == "skip":
                outcome.results.append(
                    Message(
                        role="tool",
                        text="not executed: another call in this batch awaits confirmation",
                        tool_call_id=call.id,
                        is_error=True,
                    )
                )
            # "pending" gets its result at resolution time (Task 12)
        return outcome

    async def _suspend(
        self,
        conv_id: int,
        *,
        assistant: Message,
        outcome: _BatchOutcome,
        iteration: int,
        tier: str,
        user_id: str,
        executed_any: bool,
        turn_tainted: bool,
    ) -> ChatReply:
        call = outcome.pending_call
        assert call is not None
        confirmation_id = uuid.uuid4().hex[:32]
        async with self._db.session() as s:
            s.add(
                PendingConfirmation(
                    id=confirmation_id,
                    conversation_id=conv_id,
                    user_id=user_id,
                    tool=call.name,
                    args=json_safe(call.args),
                    loop_state={
                        "assistant": assistant.to_dict(),
                        "results": [m.to_dict() for m in outcome.results],
                        "pending_call_id": call.id,
                        "iteration": iteration,
                        "tier": tier,
                        # DEVIATION (Task 11 review): persist the loop's
                        # executed_any and taint at suspension so resume can
                        # report honestly and keep MCP taint across the gap.
                        "executed_any": executed_any,
                        "turn_tainted": turn_tainted,
                    },
                    status="pending",
                    expires_at=_utcnow() + timedelta(seconds=self._config.confirm_ttl_seconds),
                )
            )
        summary = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
        return ChatReply(
            text=f"Confirm: run {call.name}({summary})? This action requires approval.",
            pending_confirmation_id=confirmation_id,
        )

    # -- confirmation resolution ----------------------------------------------
    async def _auto_decline_pending(self, conv_id: int) -> None:
        """An intervening user message (or lazy expiry) declines the pending
        action and closes its turn WITHOUT a model call — a dangling
        tool_calls message must never persist."""
        async with self._db.session() as s:
            rows = (
                (
                    await s.execute(
                        select(PendingConfirmation)
                        .where(
                            PendingConfirmation.conversation_id == conv_id,
                            PendingConfirmation.status == "pending",
                        )
                        .order_by(PendingConfirmation.created_at)
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                row.status = "expired" if row.expires_at < _utcnow() else "declined"
        for row in rows:
            await self._close_suspended_turn(row, note="declined (superseded)")
            await log_activity(
                self._db,
                tool=row.tool,
                args=row.args,
                status=row.status,
                result_preview="auto-declined",
                conversation_id=conv_id,
            )

    async def _close_suspended_turn(self, row: PendingConfirmation, *, note: str) -> None:
        """Append the suspended assistant turn, its already-run results, and a
        final error result for the un-run pending call — in that order — so the
        persisted history never carries an assistant tool_call without a
        matching tool result. Appends commit per-message (see self-review notes
        on the crash window); the ordering guarantees a valid pairing on the
        success path."""
        state = row.loop_state
        await self._convo.append(row.conversation_id, Message.from_dict(state["assistant"]))
        for d in state["results"]:
            await self._convo.append(row.conversation_id, Message.from_dict(d))
        await self._convo.append(
            row.conversation_id,
            Message(
                role="tool",
                text=note,
                tool_call_id=state["pending_call_id"],
                is_error=True,
            ),
        )

    async def resolve_confirmation(
        self, confirmation_id: str, *, approved: bool, user_id: str
    ) -> ChatReply:
        # Fast unauthenticated existence check outside the lock; the read under
        # the lock below is the authoritative one that gates the transition.
        async with self._db.session() as s:
            row = (
                await s.execute(
                    select(PendingConfirmation).where(PendingConfirmation.id == confirmation_id)
                )
            ).scalar_one_or_none()
        if row is None:
            return ChatReply(text="Unknown confirmation — it may already be resolved.")
        async with await self._lock_for(row.conversation_id):
            async with self._db.session() as s:
                row = (
                    await s.execute(
                        select(PendingConfirmation).where(PendingConfirmation.id == confirmation_id)
                    )
                ).scalar_one()
                if row.status != "pending":
                    return ChatReply(text=f"Unknown confirmation — already {row.status}.")
                if row.user_id != user_id:
                    return ChatReply(text="Only the requester can resolve this confirmation.")
                if row.expires_at < _utcnow():
                    row.status = "expired"
                    expired = True
                else:
                    row.status = "confirmed" if approved else "declined"
                    expired = False
            state = row.loop_state
            # DEVIATION (Task 11 review): honest resume state — the stored
            # executed_any (pre-suspension side effects) OR any successful
            # result in the suspended batch; taint carried across the gap.
            stored_executed_any = state.get("executed_any", False)
            results_have_success = any(not d.get("is_error", False) for d in state["results"])
            initial_taint = state.get("turn_tainted", False)
            if expired:
                await self._close_suspended_turn(row, note="declined (expired)")
                await log_activity(
                    self._db,
                    tool=row.tool,
                    args=row.args,
                    status="declined",
                    result_preview="expired",
                    conversation_id=row.conversation_id,
                )
                return ChatReply(text="That confirmation expired; nothing was done.")
            if not approved:
                await self._close_suspended_turn(row, note="declined by user")
                await log_activity(
                    self._db,
                    tool=row.tool,
                    args=row.args,
                    status="declined",
                    conversation_id=row.conversation_id,
                )
                return await self._loop(
                    row.conversation_id,
                    tier=state["tier"],
                    user_id=user_id,
                    start_iteration=state["iteration"] + 1,
                    executed_any=stored_executed_any or results_have_success,
                    initial_taint=initial_taint,
                )
            # approved: execute now, then resume. Order matters — the tool
            # result is produced before the assistant turn + prior results +
            # final result are appended, so no dangling tool_call persists.
            call = ToolCall(id=state["pending_call_id"], name=row.tool, args=dict(row.args))
            result = await self._execute_confirmed(row.conversation_id, call)
            await self._convo.append(row.conversation_id, Message.from_dict(state["assistant"]))
            for d in state["results"]:
                await self._convo.append(row.conversation_id, Message.from_dict(d))
            await self._convo.append(row.conversation_id, result)
            # DEVIATION (Fix round 1 review): mirror the deny path's honesty —
            # executed_any must reflect whether ANY tool actually succeeded
            # (pre-suspension successes, suspended-batch successes, or this
            # just-run confirmed call), not unconditionally True. A confirmed
            # tool that raised must not make a post-resume LLMError claim
            # "I completed the action(s)" when nothing actually succeeded.
            executed_any = stored_executed_any or results_have_success or (not result.is_error)
            return await self._loop(
                row.conversation_id,
                tier=state["tier"],
                user_id=user_id,
                start_iteration=state["iteration"] + 1,
                executed_any=executed_any,
                initial_taint=initial_taint,
            )

    async def _execute_confirmed(self, conv_id: int, call: ToolCall) -> Message:
        tool = self._registry.tools.get(call.name)
        if tool is None:
            return Message(
                role="tool",
                text=f"tool {call.name!r} no longer exists",
                tool_call_id=call.id,
                is_error=True,
            )
        try:
            # Revalidate through the schema: loop_state round-trips args through
            # JSON, so enum values must be re-coerced back to members here.
            kwargs = tool.schema.validate(call.args)
            result = await asyncio.wait_for(tool.fn(**kwargs), tool.timeout)
        except Exception as e:  # noqa: BLE001 — surfaced to the model, never fatal
            detail = f"confirmed tool {call.name} failed: {type(e).__name__}: {e}"
            await log_activity(
                self._db,
                tool=call.name,
                args=call.args,
                status="error",
                result_preview=detail,
                conversation_id=conv_id,
            )
            return Message(role="tool", text=detail, tool_call_id=call.id, is_error=True)
        text = result if isinstance(result, str) else json.dumps(json_safe(result))
        await log_activity(
            self._db,
            tool=call.name,
            args=call.args,
            status="confirmed",
            result_preview=text,
            conversation_id=conv_id,
        )
        return Message(role="tool", text=text, tool_call_id=call.id)
