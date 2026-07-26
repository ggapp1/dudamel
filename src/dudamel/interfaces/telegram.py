"""Telegram interface (Plan 3 Task 5). THIN by Global Constraints: every
handler parses an Update, authorizes the sender, then calls exactly one
`Runtime` method — zero business logic, zero LLM calls live here.

Auth rules (Global Constraints, binding):
  - per-USER allowlist (`from_user.id`), deny-by-default
  - groups rejected unless `allow_groups`
  - callback queries are authorized the same way as text messages, PLUS the
    Router's own originating-user check on `resolve_confirmation` (belt and
    suspenders — see `Router.resolve_confirmation`)
  - a stranger (not on the allowlist, in an otherwise-authorized chat) gets
    their numeric ID echoed back once per hour, never more often
  - `drop_pending_updates=True` on start — a restart must not replay a
    backlog of messages queued while the process was down

Lifecycle is PTB's low-level API (`initialize`/`start`/`updater.start_polling`)
rather than `run_polling`: the assembly (Plan 3 Task 6) owns the event loop,
running this alongside uvicorn and the scheduler in the same process.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from dudamel.config import Settings
from dudamel.runtime import Runtime

logger = logging.getLogger("dudamel.interfaces.telegram")

__all__ = ["TelegramInterface", "resolve_token"]

_MAX_MESSAGE_LEN = 4096
_STRANGER_COOLDOWN = timedelta(hours=1)


def resolve_token(settings: Settings) -> str | None:
    """The configured Telegram bot token, or None if its env var is unset/empty."""
    return os.environ.get(settings.telegram.token_env) or None


def _split_message(text: str, limit: int = _MAX_MESSAGE_LEN) -> list[str]:
    """Hard-slice `text` into <=`limit`-char chunks. Telegram rejects any
    single message over 4096 chars outright, so a long model reply must be
    sent as several messages rather than truncated or dropped."""
    if not text:
        return []
    return [text[i : i + limit] for i in range(0, len(text), limit)]


_TRUNCATION_MARKER = "… [truncated]"


def _fit_single_message(text: str, limit: int = _MAX_MESSAGE_LEN) -> str:
    """Fit `text` inside Telegram's per-message hard limit by truncating,
    for the two call sites that must stay a SINGLE message rather than being
    split across several the way `_send_text`/`notify` can: an inline
    keyboard lives on exactly one message, and editing a message can't fan
    out into more than the one it's editing. A tool call's args summary
    (arbitrary user/model-supplied values -- see `Router._request_
    confirmation`) is the realistic way this gets hit."""
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


class TelegramInterface:
    """Wraps a PTB `Application`. Construction builds the Application and
    registers handlers (cheap, no network — network only happens in
    `start()`/`stop()`, which the assembly calls)."""

    def __init__(self, runtime: Runtime, settings: Settings) -> None:
        token = resolve_token(settings)
        if token is None:
            raise RuntimeError(
                "refusing to start Telegram interface: set the "
                f"{settings.telegram.token_env} environment variable"
            )
        self._runtime = runtime
        self._settings = settings
        self._last_stranger_reply: dict[int, datetime] = {}
        self._app: Application = ApplicationBuilder().token(token).build()
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))
        self._app.add_handler(CallbackQueryHandler(self._on_callback))
        self._app.add_error_handler(self._on_error)

    # -- lifecycle --------------------------------------------------------------
    async def start(self) -> None:
        await self._app.initialize()
        await self._app.start()
        if self._app.updater is None:
            raise RuntimeError("PTB application has no updater")
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self._app.updater is not None and self._app.updater.running:
            await self._app.updater.stop()
        if self._app.running:
            await self._app.stop()
        await self._app.shutdown()

    # -- outbound (used by Runtime.bind_notify) ----------------------------------
    async def notify(self, text: str) -> None:
        """Send `text` to the first allow-listed user. Bound to
        `Runtime.bind_notify` by the assembly once this interface is up,
        replacing the WARN-log fallback Runtime binds at construction."""
        allowed = self._settings.telegram.allowed_user_ids
        if not allowed:
            logger.warning("telegram notify with no allowed_user_ids configured: %s", text)
            return
        await self._send_text(allowed[0], text)

    async def _send_text(self, chat_id: int, text: str) -> None:
        for chunk in _split_message(text):
            try:
                await self._app.bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")
            except BadRequest:
                # A model/user-authored reply is not guaranteed to be valid
                # Markdown — degrade to plain text rather than swallow it.
                await self._app.bot.send_message(chat_id=chat_id, text=chunk)

    async def _send_confirmation(self, chat_id: int, text: str, confirmation_id: str) -> None:
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅", callback_data=f"confirm:{confirmation_id}:yes"),
                    InlineKeyboardButton("❌", callback_data=f"confirm:{confirmation_id}:no"),
                ]
            ]
        )
        text = _fit_single_message(text)
        try:
            await self._app.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)
        except BadRequest as e:
            # The args summary embedded in `text` is arbitrary user/model
            # data (see `Router._request_confirmation`) and the API can
            # reject it for reasons the length check above doesn't cover --
            # degrade to a short, guaranteed-safe prompt rather than lose
            # the confirm/cancel buttons entirely.
            logger.warning("telegram confirmation send rejected (%s); using plain fallback", e)
            await self._app.bot.send_message(
                chat_id=chat_id,
                text="Confirmation required (original prompt could not be delivered).",
                reply_markup=keyboard,
            )

    # -- authorization ------------------------------------------------------------
    def _is_allowed(self, user_id: int) -> bool:
        return user_id in self._settings.telegram.allowed_user_ids

    def _chat_authorized(self, chat_type: str) -> bool:
        return chat_type == "private" or self._settings.telegram.allow_groups

    def _rate_limited_stranger(self, user_id: int) -> bool:
        """True if `user_id` already got their ID-reply within the cooldown
        window. Side-effecting: records `user_id` as replied-to now when
        returning False, so call this exactly once per stranger message."""
        now = datetime.now(UTC)
        last = self._last_stranger_reply.get(user_id)
        if last is not None and now - last < _STRANGER_COOLDOWN:
            return True
        self._last_stranger_reply[user_id] = now
        # Bound the dict: prune entries older than 1 hour when size exceeds 1000.
        if len(self._last_stranger_reply) > 1000:
            self._last_stranger_reply = {
                uid: ts
                for uid, ts in self._last_stranger_reply.items()
                if now - ts < _STRANGER_COOLDOWN
            }
        return False

    # -- handlers -------------------------------------------------------------------
    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None or not message.text or message.from_user is None:
            return
        chat = message.chat
        user = message.from_user
        if not self._chat_authorized(chat.type):
            return  # group rejection: total silence, not even a stranger reply
        if not self._is_allowed(user.id):
            # Stranger reply only in private chats; silently ignore in groups.
            if chat.type == "private" and not self._rate_limited_stranger(user.id):
                await self._send_text(
                    chat.id,
                    f"Your Telegram user ID is {user.id}. Ask the administrator to "
                    "allow-list it before I'll respond.",
                )
            return
        reply = await self._runtime.chat(
            f"telegram:{chat.id}",
            message.text,
            user_id=str(user.id),
            client_msg_id=str(update.update_id),
        )
        if not reply.text:
            return  # dedupe: interfaces drop empty ChatReply text
        if reply.pending_confirmation_id is not None:
            await self._send_confirmation(chat.id, reply.text, reply.pending_confirmation_id)
        else:
            await self._send_text(chat.id, reply.text)

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return
        user = query.from_user
        message = query.message
        chat_type = message.chat.type if message is not None else "private"
        if not (self._chat_authorized(chat_type) and self._is_allowed(user.id)):
            await self._app.bot.answer_callback_query(
                query.id, text="Not authorized.", show_alert=True
            )
            return
        parts = query.data.split(":")
        if len(parts) != 3 or parts[0] != "confirm" or parts[2] not in ("yes", "no"):
            await self._app.bot.answer_callback_query(query.id)
            return
        _, confirmation_id, action = parts
        # Belt and suspenders: Router.resolve_confirmation independently
        # rejects a user_id that doesn't match the confirmation's requester,
        # regardless of the allowlist check above.
        reply = await self._runtime.resolve_confirmation(
            confirmation_id, approved=(action == "yes"), user_id=str(user.id)
        )
        if message is not None:
            edit_text = _fit_single_message(reply.text)
            try:
                await self._app.bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    text=edit_text,
                    reply_markup=None,
                )
            except BadRequest as e:
                logger.warning("telegram confirmation edit rejected (%s); using plain fallback", e)
                await self._app.bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    text="Done (original reply could not be delivered).",
                    reply_markup=None,
                )
        await self._app.bot.answer_callback_query(query.id)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Registered via `Application.add_error_handler` — PTB routes any
        exception raised inside `_on_message`/`_on_callback` (or its own
        polling internals) here instead of letting it propagate and kill the
        update-processing loop. Logged, not silently swallowed, so a handler
        bug shows up rather than just going dark."""
        logger.error("telegram handler error: %s", context.error)
