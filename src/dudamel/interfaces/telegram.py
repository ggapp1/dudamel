"""Telegram interface. Intentionally thin: every handler parses an Update,
authorizes the sender, then calls exactly one `Runtime` method — zero
business logic, zero LLM calls live here.

Auth rules (binding for this interface):
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
rather than `run_polling`: the single-process assembly (dudamel.serve.serve)
owns the event loop, running this alongside uvicorn and the scheduler in the
same process.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from typing import Any

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from dudamel.config import Settings
from dudamel.contract.renderers import UNSAFE_DISPLAY_CHARS
from dudamel.exceptions import UnknownActionError
from dudamel.runtime import Runtime

logger = logging.getLogger("dudamel.interfaces.telegram")

__all__ = ["TelegramInterface", "resolve_token"]

_MAX_MESSAGE_LEN = 4096
_STRANGER_COOLDOWN = timedelta(hours=1)
_MAX_STRANGER_ENTRIES = 1000
_ACTION_TOKEN_TTL = timedelta(seconds=3600)
_MAX_ACTION_TOKENS = 512
_MAX_ACTION_BUTTONS = 20
_MAX_DIGEST_LINE = 256


class _ActionTokens:
    """Short-lived handles for homescreen action buttons.

    Telegram caps callback_data at 64 bytes, so a button cannot carry its
    tool's arguments the way a confirmation button carries an id. It carries
    a token instead and this maps it back.

    Bounded two ways on purpose: entries expire AND the map is capped by LRU
    eviction. Age alone cannot bound it -- enough distinct buttons issued
    inside the TTL window leaves every entry fresh, so an age filter removes
    nothing. This is the same two-bound shape `_last_stranger_reply` uses.
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[str, tuple[str, dict[str, Any], int, datetime]] = OrderedDict()

    def issue(self, tool: str, args: dict[str, Any], user_id: int) -> str:
        token = secrets.token_urlsafe(8)
        self._entries[token] = (tool, args, user_id, datetime.now(UTC))
        self._entries.move_to_end(token)
        while len(self._entries) > _MAX_ACTION_TOKENS:
            self._entries.popitem(last=False)
        return token

    def consume(self, token: str, user_id: int) -> tuple[str, dict[str, Any]] | None:
        """Take the entry for `token` if it belongs to `user_id` and is live.

        Removal happens HERE, before the caller runs anything, so a double-tap
        cannot execute twice: the second call finds nothing. Consuming after
        execution instead would leave the window open for a second tap while
        the first call is still awaiting. There is no await between the lookup
        and the delete, so concurrent callbacks on one event loop cannot both
        succeed.

        A user mismatch deliberately does NOT consume: otherwise any
        allow-listed user could disarm another's buttons by tapping them,
        turning an authorization failure into a denial of service.
        """
        entry = self._entries.get(token)
        if entry is None:
            return None
        tool, args, owner, issued_at = entry
        if owner != user_id:
            return None
        del self._entries[token]
        if datetime.now(UTC) - issued_at > _ACTION_TOKEN_TTL:
            return None
        return tool, args


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
    (arbitrary user/model-supplied values -- see `Router._suspend`) is the
    realistic way this gets hit."""
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


# Characters an app must not be able to put on a digest line: C0/C1 controls
# and DEL (they break the line structure of a surface that escapes nothing),
# the Unicode line/paragraph separators U+2028/U+2029 (which clients render as
# line breaks and which `ListItem`'s ASCII-only url check does not catch), and
# the bidi overrides, which can reorder a line into something it does not say.
# Imported, not redefined -- see the note on UNSAFE_DISPLAY_CHARS in
# contract/renderers.py. The digest and the contract must strip the same set or
# the two surfaces draw the same row differently -- the defect this closed, one
# field to the left of the action label.
_UNSAFE_DIGEST_CHARS = UNSAFE_DISPLAY_CHARS
# Square brackets delimit a button's anchor, so app text containing them could
# forge one: a row reading "Buy milk [1 · Done]" in its own title would make an
# inert line look actionable and make the anchor ambiguous. Folded to
# parentheses rather than dropped, so the text still reads as its author wrote.
#
# What this buys and what it does not: the machine-checkable property (every
# button's label appears intact, exactly once, on the line it acts on) holds
# unconditionally, because a forged anchor cannot be spelled with ASCII
# brackets after this. What it does NOT buy is the READING of that property by
# a human -- lookalike delimiters survive, so "［1 · Delete］" (fullwidth) or
# "【1 · Delete】" in a synced-in title can still make an inert line LOOK
# actionable, which matters because one tap is consent. A blacklist is
# unwinnable here: Unicode's Ps/Pe categories run to hundreds of pairs. The
# structural fix is to move the anchor to a line PREFIX, where app text can
# never reach position 0 of a rendered line; that is a rendering change, not a
# sanitizing one, and is deliberately not attempted here.
_ANCHOR_DELIMITERS = str.maketrans({"[": "(", "]": ")"})


def _plain(value: object) -> str:
    """App-authored text, made safe to place on a digest line.

    Every fragment an app controls -- titles, subtitles, urls, stat labels,
    table cells, error messages, action labels -- goes through this. The
    digest is plain text with no escaping of any kind, so this is the only
    place that can hold the line structure and the anchor syntax.
    """
    return _UNSAFE_DIGEST_CHARS.sub("", str(value)).translate(_ANCHOR_DELIMITERS)


def _card_lines(card: dict[str, Any]) -> list[str]:
    """One card as plain-text lines, one per renderer shape.

    Every app-authored fragment is passed through `_plain`, so no title,
    subtitle, url or cell can break a line in two or forge a button anchor.
    """
    return [_plain(line) for line in _raw_card_lines(card)]


def _raw_card_lines(card: dict[str, Any]) -> list[str]:
    if "error" in card:
        return [f"⚠️ {card['title']} — {card['error']}"]
    renderer, data = card["renderer"], card["data"]
    if renderer == "stat":
        value = f"{data['value']}"
        if data.get("unit"):
            value += f" {data['unit']}"
        if data.get("delta") is not None:
            value += f" (Δ {data['delta']})"
        return [f"{data['label']}: {value}"]
    if renderer == "list":
        lines = []
        for item in data:
            line = f"• {item['title']}"
            if item.get("subtitle"):
                line += f" — {item['subtitle']}"
            if item.get("url"):
                line += f" {item['url']}"
            lines.append(line)
        return lines or ["(empty)"]
    if renderer == "table":
        rows = [" | ".join(str(c) for c in data["columns"])]
        rows += [" | ".join(str(c) for c in row) for row in data["rows"]]
        return rows
    return str(data).splitlines() or ["(empty)"]


def _button_label(action: dict[str, Any], number: int) -> str:
    """The label one tap-target carries, in the keyboard AND in the digest text.

    Numbered because a keyboard offers no other way to say which line a button
    acts on: a list where two items both offer "Done" would otherwise render
    two identical buttons whose only distinguishing feature is their order,
    and one tap is consent, so a mis-read there completes the wrong item. The
    number makes every label in a message unique and repeats it, in brackets,
    on the line the button acts on.

    A `confirm=True` action is marked. On the web that flag raises a browser
    dialog before the POST; a Telegram tap has no dialog, and one tap is the
    human decision either way, so the marker -- not a second tap -- is what
    replaces it. Applied here, where the button is built, so the resolved
    descriptor the web renders is untouched.

    The label is app-authored, so it goes through `_plain` as well: a bracket
    inside it would break the anchor it is about to be wrapped in, and a
    control character would split that anchor across lines. Its LENGTH is
    bounded upstream by `ItemAction`'s contract cap -- a label long enough to
    outgrow a line has no good rendering here, only a less obvious failure.
    """
    warn = "⚠️ " if action.get("confirm") else ""
    return f"{number} · {warn}{_plain(action['label'])}"


def _pack(
    entries: list[tuple[str, dict[str, Any] | None]], header: str
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group (text, action) pairs into messages, splitting at item boundaries.

    Guarantee: **every button in a message has its label present, intact and
    exactly once, on the line it acts on -- for any app-supplied title,
    subtitle, url or label, of any length or content.**

    Two things buy that, and both live here rather than in the caller:

    - The anchor is composed AFTER the item text is truncated, so the cap can
      only eat app text and never the anchor. Composing it upstream put the
      anchor at the end of a line that was then cut to `_MAX_DIGEST_LINE`, and
      a 300-character title -- nothing in the contract bounds one -- silently
      ate it, leaving a numbered button whose number appeared nowhere.
    - Splitting happens at item boundaries, never mid-message: an inline
      keyboard belongs to exactly one message and cannot be truncated
      alongside its text, so `_fit_single_message` must never be used on a
      keyboard-bearing digest. It would leave buttons on a message whose lines
      had been cut -- a button with no visible referent, the worst possible
      affordance on a destructive action.

    The remaining half of the guarantee -- that app text cannot forge or break
    an anchor -- is `_plain`'s, applied to every fragment before it gets here.
    """
    messages: list[tuple[str, list[dict[str, Any]]]] = []
    # The header gets the same treatment as any other line, and for the same
    # reason: `HomeSection.title` is an unconstrained string, so an over-long
    # one would push every message of the section past the limit at once, a
    # newline in it would split into two lines, and brackets in it would forge
    # an anchor -- next to EVERY button, since the header repeats on every
    # message. It comes from the operator's own `[[home.section]]` rather than
    # from an app, but it is already being length-capped for exactly this class
    # of reason; treating it as trusted for the other half would be arbitrary.
    header = _plain(header)[:_MAX_DIGEST_LINE]
    lines = [header]
    actions: list[dict[str, Any]] = []
    for raw_line, action in entries:
        line = raw_line[:_MAX_DIGEST_LINE]
        if action is not None:
            line = f"{line}  [{action['label']}]"
        too_long = len("\n".join([*lines, line])) > _MAX_MESSAGE_LEN
        too_many = action is not None and len(actions) >= _MAX_ACTION_BUTTONS
        if (too_long or too_many) and len(lines) > 1:
            messages.append(("\n".join(lines), actions))
            lines, actions = [header, line], ([action] if action is not None else [])
            continue
        lines.append(line)
        if action is not None:
            actions.append(action)
    if len(lines) > 1:
        messages.append(("\n".join(lines), actions))
    return messages


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
        self._last_stranger_reply: OrderedDict[int, datetime] = OrderedDict()
        self._action_tokens = _ActionTokens()
        self._app: Application = ApplicationBuilder().token(token).build()
        self._app.add_handler(CommandHandler("home", self._on_home))
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
            # data (see `Router._suspend`) and the API can
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
        self._last_stranger_reply.move_to_end(user_id)
        # Bound the dict by LRU eviction. Age-based pruning alone cannot bound
        # it: a flood of >1000 DISTINCT strangers within the cooldown window
        # leaves every entry fresh, so an age filter removes nothing. Evicting
        # the oldest entries caps the size unconditionally; a stranger whose
        # entry is evicted simply gets one more ID-reply than the ideal one
        # per hour, which is harmless.
        while len(self._last_stranger_reply) > _MAX_STRANGER_ENTRIES:
            self._last_stranger_reply.popitem(last=False)
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

    async def _on_home(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None or message.from_user is None:
            return
        chat, user = message.chat, message.from_user
        if not (self._chat_authorized(chat.type) and self._is_allowed(user.id)):
            return
        for section in await self._runtime.render_home():
            entries: list[tuple[str, dict[str, Any] | None]] = []
            numbered = 0
            for card in section.cards:
                # No Markdown anywhere in the digest: card titles are plain, so
                # the whole thing goes out under one parser. The alternative --
                # bolding titles -- would have widget-authored item text
                # Markdown-interpreted in the button-less messages and literal
                # in the keyboard-bearing ones, i.e. the same app-controlled
                # string read two different ways inside one /home.
                entries.append((_plain(card["title"]), None))
                lines = _card_lines(card)
                items = card.get("data") if card["renderer"] == "list" else None
                for index, line in enumerate(lines):
                    action = None
                    if items is not None and index < len(items):
                        action = items[index].get("action")
                    if action is None:
                        entries.append((line, None))
                        continue
                    numbered += 1
                    # Only the numbered label is decided here. `_pack` writes it
                    # onto the line, AFTER truncating the app's text, so no
                    # title can be long enough to push its own button's anchor
                    # off the end.
                    entries.append((line, {**action, "label": _button_label(action, numbered)}))
                for action in card["actions"]:
                    numbered += 1
                    # No text of its own: `_pack` renders this as the anchor
                    # alone, indented under the card it belongs to.
                    entries.append(("", {**action, "label": _button_label(action, numbered)}))
            for text, actions in _pack(entries, section.title or "Home"):
                await self._send_digest(chat.id, user.id, text, actions)

    async def _send_digest(
        self, chat_id: int, user_id: int, text: str, actions: list[dict[str, Any]]
    ) -> None:
        """Send one packed digest message, plain (never Markdown — see `_on_home`).

        `_pack` already fits the text inside the per-message limit, so this
        never splits: splitting is what would tear a keyboard off its lines.
        """
        keyboard = (
            InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            action["label"],
                            callback_data=(
                                "act:"
                                + self._action_tokens.issue(action["tool"], action["args"], user_id)
                            ),
                        )
                    ]
                    for action in actions
                ]
            )
            if actions
            else None
        )
        try:
            await self._app.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)
        except BadRequest as e:
            # Widget-authored text can be rejected for reasons the length cap
            # doesn't cover. Degrade to a short note, and -- unlike
            # `_send_confirmation`, which keeps its buttons because losing them
            # strands the user mid-turn -- DROP the keyboard: a button whose
            # lines were not delivered is exactly the referent-less button
            # `_pack` exists to prevent, and /home regenerates it anyway.
            logger.warning("telegram digest send rejected (%s); using plain fallback", e)
            await self._app.bot.send_message(
                chat_id=chat_id,
                text="A homescreen section could not be delivered. Send /home to retry.",
            )

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
        if query.data.startswith("act:"):
            resolved = self._action_tokens.consume(query.data.removeprefix("act:"), user.id)
            if resolved is None:
                await self._answer(query.id, "That button expired — send /home again.", alert=True)
                return
            tool, args = resolved
            try:
                await self._runtime.run_action(tool, args, actor=str(user.id), source="telegram")
            except UnknownActionError:
                # The realistic case is a digest outliving the app that owned
                # the button (disabled, or its action label removed); "Failed"
                # would misdescribe it as a tool that ran and broke.
                await self._answer(query.id, "That action is no longer available.", alert=True)
                return
            except Exception as e:
                await self._answer(query.id, f"Failed: {e}", alert=True)
                return
            # The digest is deliberately not edited. Editing adds failure modes
            # (message too old, edit rejected) for something /home regenerates,
            # and the stale keyboard is inert: THIS token is spent. Note that
            # says nothing about the action -- two /home calls leave two live
            # buttons for the same item, and tapping both runs it twice, so
            # whether that is safe is the app's business, not this module's.
            await self._answer(query.id, "Done — send /home for a fresh view.")
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
            if reply.pending_confirmation_id is not None:
                # Resolving this confirmation resumed the turn, whose model
                # then requested ANOTHER confirm-gated call. Strip the now-stale
                # buttons off the just-resolved prompt, then surface the
                # follow-up as its own message with its own buttons + id --
                # dropping it here would leave the second action unapprovable.
                await self._edit_confirmation_result(
                    message, "Approved." if action == "yes" else "Declined."
                )
                await self._send_confirmation(
                    message.chat.id, reply.text, reply.pending_confirmation_id
                )
            else:
                await self._edit_confirmation_result(message, reply.text)
        await self._app.bot.answer_callback_query(query.id)

    async def _answer(self, query_id: str, text: str, *, alert: bool = False) -> None:
        """Answer a callback query with text Telegram will actually accept.

        `answerCallbackQuery.text` is capped at 200 characters and PTB does not
        truncate it for us, so an unbounded message (a tool's exception string
        -- a SQLAlchemy IntegrityError runs to hundreds of characters) is
        rejected by the API. That rejection would surface as a BadRequest from
        a handler that has already run the tool, leaving the operator told
        nothing at the moment they most need telling.
        """
        await self._app.bot.answer_callback_query(
            query_id,
            text=_fit_single_message(text, CallbackQuery.MAX_ANSWER_TEXT_LENGTH),
            show_alert=alert,
        )

    async def _edit_confirmation_result(self, message: Message, text: str) -> None:
        """Replace a confirmation prompt's text and drop its inline keyboard,
        degrading to a short guaranteed-safe note if the API rejects the edit
        (the resolved reply is arbitrary model/user text — see `Router._suspend`)."""
        edit_text = _fit_single_message(text)
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

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Registered via `Application.add_error_handler` — PTB routes any
        exception raised inside `_on_message`/`_on_callback` (or its own
        polling internals) here instead of letting it propagate and kill the
        update-processing loop. Logged, not silently swallowed, so a handler
        bug shows up rather than just going dark."""
        logger.error("telegram handler error: %s", context.error, exc_info=context.error)
