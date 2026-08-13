"""Telegram interface tests. No network: handlers are called directly with
hand-built telegram.Update/Message/User/CallbackQuery objects, and
`TelegramInterface._app.bot` is swapped for a stub that records calls
instead of hitting the Bot API.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from telegram import CallbackQuery, Chat, InlineKeyboardMarkup, Message, Update, User
from telegram.error import BadRequest

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TelegramConfig, TierConfig
from dudamel.interfaces.telegram import (
    _ACTION_TOKEN_TTL,
    _MAX_ACTION_BUTTONS,
    _MAX_ACTION_TOKENS,
    _MAX_MESSAGE_LEN,
    TelegramInterface,
    _ActionTokens,
    _button_label,
    _card_lines,
    _pack,
    resolve_token,
)
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call

TOKEN = "123456:FAKETOKENFAKETOKENFAKETOKENFAKETOK"  # noqa: S105 — test fixture, not a real secret


# --- stub bot / application --------------------------------------------------


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.edited: list[dict[str, Any]] = []
        self.answered: list[dict[str, Any]] = []

    async def send_message(
        self, *, chat_id: int, text: str, parse_mode: str | None = None, reply_markup: Any = None
    ) -> None:
        self.sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
            }
        )

    async def edit_message_text(
        self, *, chat_id: int, message_id: int, text: str, reply_markup: Any = None
    ) -> None:
        self.edited.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "reply_markup": reply_markup,
            }
        )

    async def answer_callback_query(
        self, callback_query_id: str, *, text: str | None = None, show_alert: bool | None = None
    ) -> None:
        self.answered.append({"id": callback_query_id, "text": text, "show_alert": show_alert})


class MarkdownRejectingBot(FakeBot):
    """Rejects the first (Markdown) attempt for every send — exercises the
    parse-mode degrade-to-plain path."""

    async def send_message(
        self, *, chat_id: int, text: str, parse_mode: str | None = None, reply_markup: Any = None
    ) -> None:
        if parse_mode is not None:
            raise BadRequest("Can't parse entities: bad markdown")
        await super().send_message(
            chat_id=chat_id, text=text, parse_mode=parse_mode, reply_markup=reply_markup
        )


class ConfirmationRejectingBot(FakeBot):
    """Rejects the first `send_message` call unconditionally, then behaves
    normally — exercises `_send_confirmation`'s BadRequest-degrade retry,
    independent of message length."""

    def __init__(self) -> None:
        super().__init__()
        self._raised = False

    async def send_message(
        self, *, chat_id: int, text: str, parse_mode: str | None = None, reply_markup: Any = None
    ) -> None:
        if not self._raised:
            self._raised = True
            raise BadRequest("simulated failure")
        await super().send_message(
            chat_id=chat_id, text=text, parse_mode=parse_mode, reply_markup=reply_markup
        )


class EditRejectingBot(FakeBot):
    """Rejects the first `edit_message_text` call unconditionally, then
    behaves normally — exercises the confirmation callback's
    BadRequest-degrade retry."""

    def __init__(self) -> None:
        super().__init__()
        self._raised = False

    async def edit_message_text(
        self, *, chat_id: int, message_id: int, text: str, reply_markup: Any = None
    ) -> None:
        if not self._raised:
            self._raised = True
            raise BadRequest("simulated failure")
        await super().edit_message_text(
            chat_id=chat_id, message_id=message_id, text=text, reply_markup=reply_markup
        )


class FakeUpdater:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.running = False
        self.polling_kwargs: dict[str, Any] | None = None

    async def start_polling(self, **kwargs: Any) -> None:
        self.polling_kwargs = kwargs
        self.running = True

    async def stop(self) -> None:
        self.calls.append("stop")
        self.running = False


class FakeApplication:
    """Stand-in for PTB's Application used only to unit-test the lifecycle
    methods without touching the network (real Application.initialize()
    calls Bot.get_me())."""

    def __init__(self) -> None:
        self.bot = FakeBot()
        self.updater = FakeUpdater()
        self.calls: list[str] = []
        self.running = False

    async def initialize(self) -> None:
        self.calls.append("initialize")

    async def start(self) -> None:
        self.calls.append("start")
        self.running = True

    async def stop(self) -> None:
        self.calls.append("stop")
        self.running = False

    async def shutdown(self) -> None:
        self.calls.append("shutdown")


# --- fixtures -----------------------------------------------------------------


def make_orc() -> Orchestrator:
    app = App("gym", description="d")

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record."""
        return f"ok {exercise}"

    @app.tool(confirm=True)
    async def wipe(reason: str) -> str:
        """Delete stuff."""
        return "wiped"

    @app.tool(action="Done")
    async def complete(id: int) -> str:
        """Complete a task."""
        return "done"

    @app.tool(action="Break")
    async def explode() -> str:
        """Fail on purpose."""
        # Long on purpose: a real failure message (a database integrity error,
        # say) routinely runs to hundreds of characters, and a callback answer
        # is capped at 200.
        raise RuntimeError("kaboom " + "E" * 500)

    @app.tool(action="Nuke", confirm=True)
    async def nuke() -> str:
        """Destroy everything."""
        return "nuked"

    @app.widget(title="Tasks", renderer="list", actions=["explode", "nuke"])
    async def today() -> list[dict[str, object]]:
        return [
            {"title": "Buy milk", "action": {"tool": "complete", "args": {"id": 4}}},
            {"title": "Call mum"},
            {"title": "Walk dog", "action": {"tool": "complete", "args": {"id": 9}}},
        ]

    return Orchestrator(apps=[app])


def make_hostile_orc() -> Orchestrator:
    """An app whose widget text is as hostile as the contract permits.

    Nothing bounds a list item's title, subtitle or url, none of the three is
    checked for the characters a plain-text surface cares about, and a per-row
    label may carry anything up to its length cap. Every value here is
    reachable by an ordinary app author, deliberately or by accident.
    """
    app = App("edge", description="d")

    @app.tool(action="Done")
    async def complete(id: int) -> str:
        """Complete a task."""
        return "done"

    @app.tool(action="Sweep")
    async def sweep() -> str:
        """Sweep up."""
        return "swept"

    @app.widget(title="Edge [cases]", renderer="list", actions=["sweep"])
    async def rows() -> list[dict[str, object]]:
        return [
            {"title": "L" * 300, "action": {"tool": "complete", "args": {"id": 1}}},
            {"title": "Decoy [1 · Done] row"},  # forges an anchor in its own text
            {
                "title": "Newline label",
                "action": {"tool": "complete", "args": {"id": 3}, "label": "Do\nne"},
            },
            {
                "title": "Bracket label",
                "action": {"tool": "complete", "args": {"id": 4}, "label": "[x]"},
            },
            {
                "title": "Separator url",
                "url": "https://x.test/a\u2028b",  # the url check is ASCII-only
                "action": {"tool": "complete", "args": {"id": 5}},
            },
            {
                "title": "L" * 300,
                "subtitle": "S" * 300,
                "url": "https://x.test/" + "u" * 300,
                "action": {"tool": "complete", "args": {"id": 6}},
            },
        ]

    return Orchestrator(apps=[app])


def make_settings(tmp_path: Path, *, telegram: TelegramConfig | None = None) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/tg.db",
        data_dir=tmp_path,
        llm_tiers={"standard": TierConfig(provider="fake", model="f")},
        telegram=telegram or TelegramConfig(allowed_user_ids=[111]),
    )


async def build(
    tmp_path: Path,
    script: list,
    *,
    telegram: TelegramConfig | None = None,
    bot: FakeBot | None = None,
    orc: Orchestrator | None = None,
) -> tuple[Runtime, TelegramInterface]:
    orc = orc or make_orc()
    settings = make_settings(tmp_path, telegram=telegram)
    rt = Runtime(orc, settings, providers={"standard": FakeProvider(script)})
    await rt.start()
    interface = TelegramInterface(rt, settings)
    interface._app.bot = bot or FakeBot()
    return rt, interface


def make_user(id: int = 111, username: str = "alice") -> User:
    return User(id=id, first_name="Test", is_bot=False, username=username)


def make_chat(id: int = 111, type: str = "private") -> Chat:
    return Chat(id=id, type=type)


def make_message(
    text: str | None,
    *,
    message_id: int = 1,
    chat: Chat | None = None,
    from_user: User | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        date=datetime.now(UTC),
        chat=chat or make_chat(),
        from_user=from_user or make_user(),
        text=text,
    )


def make_update(
    *,
    message: Message | None = None,
    callback_query: CallbackQuery | None = None,
    update_id: int = 1,
) -> Update:
    return Update(update_id=update_id, message=message, callback_query=callback_query)


@pytest.fixture
def token_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DUDAMEL_TELEGRAM_TOKEN", TOKEN)
    return TOKEN


def spy_chat(rt: Runtime) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    original = rt.chat

    async def wrapper(channel: str, text: str, *, user_id: str, client_msg_id: str | None = None):
        calls.append(
            {"channel": channel, "text": text, "user_id": user_id, "client_msg_id": client_msg_id}
        )
        return await original(channel, text, user_id=user_id, client_msg_id=client_msg_id)

    rt.chat = wrapper  # type: ignore[method-assign]
    return calls


def spy_resolve_confirmation(rt: Runtime) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    original = rt.resolve_confirmation

    async def wrapper(confirmation_id: str, *, approved: bool, user_id: str):
        calls.append({"confirmation_id": confirmation_id, "approved": approved, "user_id": user_id})
        return await original(confirmation_id, approved=approved, user_id=user_id)

    rt.resolve_confirmation = wrapper  # type: ignore[method-assign]
    return calls


def spy_run_action(rt: Runtime) -> list[dict[str, Any]]:
    """Record every action execution that gets past the button's token.

    Same wrap-the-Runtime-method shape as `spy_chat`: the real tool still
    runs, so the recorded calls are executions, not merely intents.
    """
    calls: list[dict[str, Any]] = []
    original = rt.run_action

    async def wrapper(tool_name: str, args: dict[str, Any], *, actor: str, source: str):
        calls.append({"tool": tool_name, "args": args, "actor": actor, "source": source})
        return await original(tool_name, args, actor=actor, source=source)

    rt.run_action = wrapper  # type: ignore[method-assign]
    return calls


# --- resolve_token / construction ---------------------------------------------


def test_resolve_token_none_when_env_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DUDAMEL_TELEGRAM_TOKEN", raising=False)
    settings = make_settings(tmp_path)
    assert resolve_token(settings) is None


def test_resolve_token_reads_configured_env(tmp_path: Path, token_env: str) -> None:
    settings = make_settings(tmp_path)
    assert resolve_token(settings) == token_env


def test_construction_raises_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DUDAMEL_TELEGRAM_TOKEN", raising=False)
    orc = make_orc()
    settings = make_settings(tmp_path)
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([])})
    with pytest.raises(RuntimeError, match="DUDAMEL_TELEGRAM_TOKEN"):
        TelegramInterface(rt, settings)


# --- allowlisted text ----------------------------------------------------------


async def test_allowlisted_text_calls_chat_with_right_args(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [fake_text("hi there")])
    calls = spy_chat(rt)
    msg = make_message("hello", chat=make_chat(id=555), from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg, update_id=42), None)

    assert calls == [
        {"channel": "telegram:555", "text": "hello", "user_id": "111", "client_msg_id": "42"}
    ]
    bot: FakeBot = interface._app.bot
    assert bot.sent == [
        {"chat_id": 555, "text": "hi there", "parse_mode": "Markdown", "reply_markup": None}
    ]
    await rt.stop()


async def test_empty_reply_sends_nothing(tmp_path: Path, token_env: str) -> None:
    """Router returns ChatReply(text="") on a deduped client_msg_id — the
    interface must not send an empty message."""
    rt, interface = await build(tmp_path, [fake_text("first")])
    msg = make_message("hello", chat=make_chat(id=555), from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg, update_id=1), None)
    # Same client_msg_id (derived from update_id) replayed -> dedupe -> "".
    await interface._on_message(make_update(message=msg, update_id=1), None)
    bot: FakeBot = interface._app.bot
    assert len(bot.sent) == 1  # only the first call produced output
    await rt.stop()


# --- stranger rate limiting ------------------------------------------------------


async def test_stranger_gets_id_reply_once_then_silence_within_hour(
    tmp_path: Path, token_env: str
) -> None:
    rt, interface = await build(tmp_path, [])
    calls = spy_chat(rt)
    stranger = make_user(id=999)
    msg1 = make_message("hi", chat=make_chat(id=999), from_user=stranger, message_id=1)
    await interface._on_message(make_update(message=msg1, update_id=1), None)
    msg2 = make_message("hi again", chat=make_chat(id=999), from_user=stranger, message_id=2)
    await interface._on_message(make_update(message=msg2, update_id=2), None)

    bot: FakeBot = interface._app.bot
    assert calls == []  # runtime.chat is never called for a stranger
    assert len(bot.sent) == 1
    assert "999" in bot.sent[0]["text"]

    # Simulate the cooldown expiring: a new reply is allowed afterwards.
    interface._last_stranger_reply[999] -= timedelta(hours=2)
    msg3 = make_message("hi thrice", chat=make_chat(id=999), from_user=stranger, message_id=3)
    await interface._on_message(make_update(message=msg3, update_id=3), None)
    assert len(bot.sent) == 2
    await rt.stop()


async def test_stranger_reply_dict_is_bounded_under_flood(tmp_path: Path, token_env: str) -> None:
    """A flood of distinct strangers within the cooldown window must not grow
    the tracking dict without bound: age-based pruning alone can't shrink it
    (every entry is fresh), so LRU eviction has to cap it."""
    from dudamel.interfaces.telegram import _MAX_STRANGER_ENTRIES

    rt, interface = await build(tmp_path, [])
    for uid in range(_MAX_STRANGER_ENTRIES + 500):
        interface._rate_limited_stranger(uid)
    assert len(interface._last_stranger_reply) <= _MAX_STRANGER_ENTRIES
    await rt.stop()


# --- group rejection ---------------------------------------------------------------


async def test_group_message_rejected_when_allow_groups_false(
    tmp_path: Path, token_env: str
) -> None:
    rt, interface = await build(tmp_path, [])
    calls = spy_chat(rt)
    msg = make_message("hello", chat=make_chat(id=-100, type="group"), from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: FakeBot = interface._app.bot
    assert calls == []
    assert bot.sent == []  # total silence, not even a stranger reply
    await rt.stop()


async def test_group_message_allowed_when_allow_groups_true(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(
        tmp_path,
        [fake_text("ack")],
        telegram=TelegramConfig(allowed_user_ids=[111], allow_groups=True),
    )
    calls = spy_chat(rt)
    msg = make_message("hello", chat=make_chat(id=-100, type="group"), from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    assert len(calls) == 1
    assert calls[0]["channel"] == "telegram:-100"
    await rt.stop()


async def test_stranger_in_group_with_allow_groups_true_gets_no_reply(
    tmp_path: Path, token_env: str
) -> None:
    """Critical: when allow_groups=True, a stranger's group message must NOT
    trigger the ID-reveal reply (which would be posted into the group).
    Strangers should only get replies in private chats."""
    rt, interface = await build(
        tmp_path,
        [],
        telegram=TelegramConfig(allowed_user_ids=[111], allow_groups=True),
    )
    calls = spy_chat(rt)
    stranger = make_user(id=999)
    msg = make_message("hello", chat=make_chat(id=-100, type="group"), from_user=stranger)
    await interface._on_message(make_update(message=msg), None)
    bot: FakeBot = interface._app.bot
    assert calls == []  # runtime.chat not called
    assert bot.sent == []  # no reply sent into the group
    await rt.stop()


# --- long reply splitting --------------------------------------------------------


async def test_long_reply_split_across_messages(tmp_path: Path, token_env: str) -> None:
    long_text = "a" * 5000
    rt, interface = await build(tmp_path, [fake_text(long_text)])
    msg = make_message("hello", chat=make_chat(id=555), from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: FakeBot = interface._app.bot
    assert len(bot.sent) == 2
    assert all(len(part["text"]) <= 4096 for part in bot.sent)
    assert "".join(part["text"] for part in bot.sent) == long_text
    await rt.stop()


# --- markdown degrade ------------------------------------------------------------


async def test_markdown_parse_error_degrades_to_plain(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [fake_text("*unbalanced")], bot=MarkdownRejectingBot())
    msg = make_message("hello", chat=make_chat(id=555), from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: MarkdownRejectingBot = interface._app.bot
    assert bot.sent == [
        {"chat_id": 555, "text": "*unbalanced", "parse_mode": None, "reply_markup": None}
    ]
    await rt.stop()


# --- confirmation flow ----------------------------------------------------------


async def test_confirmation_sent_with_inline_keyboard(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [fake_tool_call("wipe", {"reason": "clean"})])
    msg = make_message("wipe it", chat=make_chat(id=777), from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: FakeBot = interface._app.bot
    assert len(bot.sent) == 1
    keyboard = bot.sent[0]["reply_markup"]
    assert isinstance(keyboard, InlineKeyboardMarkup)
    yes_button, no_button = keyboard.inline_keyboard[0]
    assert yes_button.text == "✅" and yes_button.callback_data.endswith(":yes")
    assert no_button.text == "❌" and no_button.callback_data.endswith(":no")
    await rt.stop()


async def test_confirm_button_roundtrip_approved_by_originating_user(
    tmp_path: Path, token_env: str
) -> None:
    rt, interface = await build(
        tmp_path, [fake_tool_call("wipe", {"reason": "clean"}), fake_text("All done")]
    )
    resolve_calls = spy_resolve_confirmation(rt)
    original_chat = make_chat(id=777)
    msg = make_message("wipe it", chat=original_chat, from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: FakeBot = interface._app.bot
    keyboard: InlineKeyboardMarkup = bot.sent[0]["reply_markup"]
    callback_data = keyboard.inline_keyboard[0][0].callback_data  # the "yes" button

    sent_message = make_message(
        None, chat=original_chat, from_user=make_user(id=111), message_id=99
    )
    query = CallbackQuery(
        id="cbq1",
        from_user=make_user(id=111),
        chat_instance="ci1",
        message=sent_message,
        data=callback_data,
    )
    await interface._on_callback(make_update(callback_query=query, update_id=2), None)

    assert resolve_calls == [
        {"confirmation_id": callback_data.split(":")[1], "approved": True, "user_id": "111"}
    ]
    assert bot.edited == [
        {"chat_id": 777, "message_id": 99, "text": "All done", "reply_markup": None}
    ]
    assert [a["id"] for a in bot.answered] == ["cbq1"]
    await rt.stop()


async def test_callback_chained_confirmation_is_surfaced(tmp_path: Path, token_env: str) -> None:
    """Approving one confirmation can resume a turn whose model then requests
    ANOTHER confirm-gated tool. The callback handler must surface that
    follow-up confirmation (its buttons + a fresh id), not drop it, or the
    second action is unapprovable and the user is stranded."""
    rt, interface = await build(
        tmp_path,
        [
            fake_tool_call("wipe", {"reason": "first"}, id="tc1"),
            fake_tool_call("wipe", {"reason": "second"}, id="tc2"),
            fake_text("All done"),
        ],
    )
    original_chat = make_chat(id=777)
    msg = make_message("wipe it", chat=original_chat, from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: FakeBot = interface._app.bot
    first_cb = bot.sent[0]["reply_markup"].inline_keyboard[0][0].callback_data
    first_id = first_cb.split(":")[1]

    sent_message = make_message(
        None, chat=original_chat, from_user=make_user(id=111), message_id=99
    )
    query = CallbackQuery(
        id="cbq1",
        from_user=make_user(id=111),
        chat_instance="ci1",
        message=sent_message,
        data=first_cb,
    )
    await interface._on_callback(make_update(callback_query=query, update_id=2), None)

    # A second confirmation must have been sent, with buttons and a NEW id.
    chained = [s for s in bot.sent[1:] if isinstance(s["reply_markup"], InlineKeyboardMarkup)]
    assert len(chained) == 1
    assert chained[0]["chat_id"] == 777
    second_cb = chained[0]["reply_markup"].inline_keyboard[0][0].callback_data
    second_id = second_cb.split(":")[1]
    assert second_id != first_id

    # And the surfaced second confirmation is itself resolvable end-to-end.
    sent_message2 = make_message(
        None, chat=original_chat, from_user=make_user(id=111), message_id=100
    )
    query2 = CallbackQuery(
        id="cbq2",
        from_user=make_user(id=111),
        chat_instance="ci2",
        message=sent_message2,
        data=second_cb,
    )
    await interface._on_callback(make_update(callback_query=query2, update_id=3), None)
    assert any(e["text"] == "All done" for e in bot.edited)
    await rt.stop()


async def test_confirm_button_from_unauthorized_user_does_not_resolve(
    tmp_path: Path, token_env: str
) -> None:
    rt, interface = await build(tmp_path, [fake_tool_call("wipe", {"reason": "clean"})])
    resolve_calls = spy_resolve_confirmation(rt)
    original_chat = make_chat(id=777)
    msg = make_message("wipe it", chat=original_chat, from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: FakeBot = interface._app.bot
    callback_data = bot.sent[0]["reply_markup"].inline_keyboard[0][0].callback_data

    sent_message = make_message(
        None, chat=original_chat, from_user=make_user(id=111), message_id=99
    )
    stranger = make_user(id=999)
    query = CallbackQuery(
        id="cbq2", from_user=stranger, chat_instance="ci2", message=sent_message, data=callback_data
    )
    await interface._on_callback(make_update(callback_query=query, update_id=3), None)

    assert resolve_calls == []
    assert bot.edited == []
    assert bot.answered == [{"id": "cbq2", "text": "Not authorized.", "show_alert": True}]
    await rt.stop()


# --- confirmation hardening ------------------------------------------------------


async def test_confirmation_send_badrequest_degrades_to_plain_fallback(
    tmp_path: Path, token_env: str
) -> None:
    rt, interface = await build(
        tmp_path, [fake_tool_call("wipe", {"reason": "clean"})], bot=ConfirmationRejectingBot()
    )
    msg = make_message("wipe it", chat=make_chat(id=777), from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: ConfirmationRejectingBot = interface._app.bot
    assert len(bot.sent) == 1  # the rejected first attempt is never recorded
    assert bot.sent[0]["text"] == "Confirmation required (original prompt could not be delivered)."
    assert isinstance(bot.sent[0]["reply_markup"], InlineKeyboardMarkup)
    await rt.stop()


async def test_confirmation_prompt_over_limit_is_truncated(tmp_path: Path, token_env: str) -> None:
    huge_reason = "x" * 5000
    rt, interface = await build(tmp_path, [fake_tool_call("wipe", {"reason": huge_reason})])
    msg = make_message("wipe it", chat=make_chat(id=777), from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: FakeBot = interface._app.bot
    assert len(bot.sent) == 1
    assert len(bot.sent[0]["text"]) <= 4096
    assert bot.sent[0]["text"].endswith("[truncated]")
    await rt.stop()


async def test_confirm_edit_badrequest_degrades_to_plain_fallback(
    tmp_path: Path, token_env: str
) -> None:
    rt, interface = await build(
        tmp_path,
        [fake_tool_call("wipe", {"reason": "clean"}), fake_text("All done")],
        bot=EditRejectingBot(),
    )
    resolve_calls = spy_resolve_confirmation(rt)
    original_chat = make_chat(id=777)
    msg = make_message("wipe it", chat=original_chat, from_user=make_user(id=111))
    await interface._on_message(make_update(message=msg), None)
    bot: EditRejectingBot = interface._app.bot
    callback_data = bot.sent[0]["reply_markup"].inline_keyboard[0][0].callback_data

    sent_message = make_message(
        None, chat=original_chat, from_user=make_user(id=111), message_id=99
    )
    query = CallbackQuery(
        id="cbq3",
        from_user=make_user(id=111),
        chat_instance="ci3",
        message=sent_message,
        data=callback_data,
    )
    await interface._on_callback(make_update(callback_query=query, update_id=2), None)

    assert resolve_calls == [
        {"confirmation_id": callback_data.split(":")[1], "approved": True, "user_id": "111"}
    ]
    assert bot.edited == [
        {
            "chat_id": 777,
            "message_id": 99,
            "text": "Done (original reply could not be delivered).",
            "reply_markup": None,
        }
    ]
    await rt.stop()


# --- notify ----------------------------------------------------------------------


async def test_notify_targets_first_allowed_id(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[555, 111]))
    await interface.notify("scheduled job failed")
    bot: FakeBot = interface._app.bot
    assert bot.sent == [
        {
            "chat_id": 555,
            "text": "scheduled job failed",
            "parse_mode": "Markdown",
            "reply_markup": None,
        }
    ]
    await rt.stop()


async def test_notify_with_no_allowed_users_does_not_crash(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[]))
    await interface.notify("hello")  # must not raise
    bot: FakeBot = interface._app.bot
    assert bot.sent == []
    await rt.stop()


# --- lifecycle -------------------------------------------------------------------


async def test_error_handler_registered_with_the_application(
    tmp_path: Path, token_env: str
) -> None:
    rt, interface = await build(tmp_path, [])
    assert interface._on_error in interface._app.error_handlers
    await rt.stop()


async def test_error_handler_logs_handler_exceptions(
    tmp_path: Path, token_env: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Invoked directly with a stub context, mirroring how PTB itself would
    call it after catching an exception raised inside `_on_message`/
    `_on_callback`."""
    rt, interface = await build(tmp_path, [])

    class _StubContext:
        error = RuntimeError("boom")

    await interface._on_error(None, _StubContext())  # type: ignore[arg-type]
    matching = [
        r for r in caplog.records if "telegram handler error" in r.message and "boom" in r.message
    ]
    assert matching
    # The traceback must be attached (exc_info), not just the str(exc) --
    # a bare message hides where the failure came from.
    assert any(r.exc_info is not None for r in matching)
    await rt.stop()


async def test_start_stop_use_lifecycle_api_not_run_polling(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [])
    fake_app = FakeApplication()
    interface._app = fake_app  # type: ignore[assignment]

    await interface.start()
    assert fake_app.calls == ["initialize", "start"]
    assert fake_app.updater.polling_kwargs == {"drop_pending_updates": True}

    await interface.stop()
    assert fake_app.updater.calls == ["stop"]
    assert fake_app.calls == ["initialize", "start", "stop", "shutdown"]
    await rt.stop()


# --- homescreen action tokens -------------------------------------------------


def test_a_token_resolves_once() -> None:
    tokens = _ActionTokens()
    token = tokens.issue("complete", {"id": 4}, user_id=7)
    assert tokens.consume(token, user_id=7) == ("complete", {"id": 4})
    assert tokens.consume(token, user_id=7) is None


def test_a_token_belongs_to_the_user_it_was_issued_to() -> None:
    tokens = _ActionTokens()
    token = tokens.issue("complete", {"id": 4}, user_id=7)
    assert tokens.consume(token, user_id=8) is None
    # and the rightful owner's button still works: a wrong-user tap must not
    # disarm it, or any allow-listed user could deny service to another
    assert tokens.consume(token, user_id=7) == ("complete", {"id": 4})


def test_an_expired_token_does_not_resolve() -> None:
    tokens = _ActionTokens()
    token = tokens.issue("complete", {"id": 4}, user_id=7)
    entry = tokens._entries[token]
    tokens._entries[token] = (*entry[:3], entry[3] - _ACTION_TOKEN_TTL - timedelta(seconds=1))
    assert tokens.consume(token, user_id=7) is None


def test_the_map_is_bounded_by_lru_eviction() -> None:
    """Age alone cannot bound the map: every entry issued inside the TTL
    window is fresh, so an age filter removes nothing. LRU eviction caps it
    unconditionally."""
    tokens = _ActionTokens()
    first = tokens.issue("complete", {"id": 0}, user_id=7)
    for n in range(1, _MAX_ACTION_TOKENS + 1):
        tokens.issue("complete", {"id": n}, user_id=7)
    assert tokens.consume(first, user_id=7) is None
    assert len(tokens._entries) == _MAX_ACTION_TOKENS


def test_an_unknown_token_does_not_resolve() -> None:
    assert _ActionTokens().consume("nope", user_id=7) is None


# --- digest rendering ---------------------------------------------------------


def test_stat_card_renders_one_line() -> None:
    card = {
        "qualified_id": "w.now",
        "title": "Weather",
        "renderer": "stat",
        "data": {"label": "Temp", "value": 12, "unit": "C", "delta": -2.0},
        "actions": [],
    }
    assert _card_lines(card) == ["Temp: 12 C (Δ -2.0)"]


def test_error_card_renders_a_warning_line() -> None:
    card = {
        "qualified_id": "w.now",
        "title": "Weather",
        "renderer": "stat",
        "error": "boom",
        "actions": [],
    }
    assert _card_lines(card) == ["⚠️ Weather — boom"]


def test_packing_never_separates_a_button_from_its_line() -> None:
    """Long sections split across messages; every button must land in the
    message that also contains the item text it acts on."""
    entries = [(f"• item {n}", {"tool": "t", "args": {}, "label": "Go"}) for n in range(60)]
    messages = _pack(entries, "Today")
    assert len(messages) > 1
    for text, actions in messages:
        assert len(actions) <= _MAX_ACTION_BUTTONS
        assert text.count("• item ") == len(actions)
        assert len(text) <= _MAX_MESSAGE_LEN


# --- /home and its buttons ------------------------------------------------------


async def _home(interface: TelegramInterface, user_id: int = 111) -> None:
    message = make_message("/home", from_user=make_user(id=user_id))
    await interface._on_home(make_update(message=message), None)


def _button_data(bot: FakeBot, label: str) -> str:
    for sent in bot.sent:
        keyboard = sent["reply_markup"]
        if keyboard is None:
            continue
        for row in keyboard.inline_keyboard:
            for button in row:
                if label in button.text:  # button labels are numbered
                    return button.callback_data
    raise AssertionError(f"no button labelled {label!r} was sent")


def _keyboard_messages(bot: FakeBot) -> list[tuple[str, list[Any]]]:
    """Every sent message that carries a keyboard, as (text, buttons)."""
    out = []
    for sent in bot.sent:
        keyboard = sent["reply_markup"]
        if keyboard is not None:
            out.append((sent["text"], [b for row in keyboard.inline_keyboard for b in row]))
    return out


def _first_button_data(bot: FakeBot) -> str:
    return _button_data(bot, "Done")


def _tap(data: str, *, user_id: int = 111, query_id: str = "cbq1") -> Update:
    query = CallbackQuery(
        id=query_id,
        from_user=make_user(id=user_id),
        chat_instance="ci1",
        message=make_message(None, chat=make_chat(), from_user=make_user(id=user_id)),
        data=data,
    )
    return make_update(callback_query=query, update_id=2)


async def test_home_sends_the_digest_to_an_allowlisted_user(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    await _home(interface)
    bot: FakeBot = interface._app.bot
    assert any("Buy milk" in sent["text"] for sent in bot.sent)
    await rt.stop()


async def test_home_from_a_stranger_sends_nothing(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[999]))
    await _home(interface)
    assert interface._app.bot.sent == []
    await rt.stop()


async def test_tapping_an_action_runs_it_once_and_answers(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    calls = spy_run_action(rt)
    await _home(interface)
    bot: FakeBot = interface._app.bot
    data = _first_button_data(bot)

    await interface._on_callback(_tap(data), None)

    assert calls == [{"tool": "complete", "args": {"id": 4}, "actor": "111", "source": "telegram"}]
    assert bot.answered[-1]["id"] == "cbq1"
    await rt.stop()


async def test_a_second_tap_does_nothing_and_reports_expiry(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    calls = spy_run_action(rt)
    await _home(interface)
    bot: FakeBot = interface._app.bot
    data = _first_button_data(bot)

    await interface._on_callback(_tap(data), None)
    await interface._on_callback(_tap(data, query_id="cbq2"), None)

    assert len(calls) == 1  # the second tap never reached the tool
    assert "expired" in bot.answered[-1]["text"]
    await rt.stop()


async def test_concurrent_taps_on_one_token_execute_once(tmp_path: Path, token_env: str) -> None:
    """The token is consumed before the tool runs, so two callbacks racing on
    one event loop cannot both get past the lookup."""
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    calls = spy_run_action(rt)
    await _home(interface)
    data = _first_button_data(interface._app.bot)

    await asyncio.gather(
        interface._on_callback(_tap(data, query_id="a"), None),
        interface._on_callback(_tap(data, query_id="b"), None),
    )

    assert len(calls) == 1
    await rt.stop()


async def test_another_users_tap_neither_runs_nor_disarms(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111, 222]))
    calls = spy_run_action(rt)
    await _home(interface, user_id=111)
    data = _first_button_data(interface._app.bot)

    await interface._on_callback(_tap(data, user_id=222, query_id="other"), None)
    assert calls == []

    await interface._on_callback(_tap(data, user_id=111, query_id="owner"), None)
    assert [call["actor"] for call in calls] == ["111"]
    await rt.stop()


async def test_a_failing_action_is_reported_on_the_callback(tmp_path: Path, token_env: str) -> None:
    """A tool that raises must surface as an answered callback, not as an
    exception escaping the handler -- and the answer must be short enough for
    Telegram to accept, or the operator is told nothing about a tool that has
    already run."""
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    await _home(interface)
    bot: FakeBot = interface._app.bot
    data = _button_data(bot, "Break")

    await interface._on_callback(_tap(data, query_id="cbq9"), None)

    assert bot.answered[-1]["id"] == "cbq9"
    assert "kaboom" in bot.answered[-1]["text"]
    assert len(bot.answered[-1]["text"]) <= CallbackQuery.MAX_ANSWER_TEXT_LENGTH
    await rt.stop()


async def test_a_vanished_action_is_not_reported_as_a_failure(
    tmp_path: Path, token_env: str
) -> None:
    """A digest can outlive the app that owned its buttons. Tapping one then
    reaches no tool at all, which is a different thing from a tool that ran
    and broke."""
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    token = interface._action_tokens.issue("ghost", {}, 111)

    await interface._on_callback(_tap(f"act:{token}", query_id="cbq8"), None)

    assert "no longer available" in interface._app.bot.answered[-1]["text"]
    await rt.stop()


async def test_every_button_is_named_on_the_line_it_acts_on(tmp_path: Path, token_env: str) -> None:
    """The guarantee: in each message, every button's label appears verbatim
    in the text, exactly once, appended to the line that button acts on. Two
    items offering the same action would otherwise render two identical
    buttons distinguishable only by keyboard order -- and one tap is consent,
    so reading that order wrong completes the wrong item.
    """
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    await _home(interface)
    messages = _keyboard_messages(interface._app.bot)
    assert messages
    for text, buttons in messages:
        labels = [button.text for button in buttons]
        assert len(set(labels)) == len(labels)  # unique within the message
        for label in labels:
            assert text.count(f"[{label}]") == 1
    # and the two same-named item actions are anchored to different lines
    text, _ = messages[0]
    milk = next(line for line in text.splitlines() if "Buy milk" in line)
    dog = next(line for line in text.splitlines() if "Walk dog" in line)
    assert "Done]" in milk and "Done]" in dog and milk != dog
    assert all("Done]" not in line for line in text.splitlines() if "Call mum" in line)
    await rt.stop()


async def test_a_confirm_action_is_marked_before_it_is_tapped(
    tmp_path: Path, token_env: str
) -> None:
    """One tap is consent on Telegram, as on the web -- but the browser's
    dialog also guards against a mis-tap, and a phone screen needs that more.
    The marker is what replaces it."""
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    await _home(interface)
    text, buttons = _keyboard_messages(interface._app.bot)[0]
    marked = [button.text for button in buttons if "Nuke" in button.text]
    assert marked and all(button.startswith(("⚠️", "1", "2", "3", "4", "5")) for button in marked)
    assert all("⚠️" in button for button in marked)
    assert all("⚠️" not in button.text for button in buttons if "Nuke" not in button.text)
    assert f"[{marked[0]}]" in text  # the marker shows in the text too
    await rt.stop()


async def test_the_digest_is_sent_plain_never_markdown(tmp_path: Path, token_env: str) -> None:
    """One parser for the whole digest. Bolding card titles would leave
    app-authored item text Markdown-interpreted in the button-less messages
    and literal in the keyboard-bearing ones."""
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    await _home(interface)
    bot: FakeBot = interface._app.bot
    assert bot.sent
    assert all(sent["parse_mode"] is None for sent in bot.sent)
    assert all("*Tasks*" not in sent["text"] for sent in bot.sent)
    assert any("Tasks" in sent["text"] for sent in bot.sent)
    await rt.stop()


async def test_a_rejected_digest_degrades_without_its_buttons(
    tmp_path: Path, token_env: str
) -> None:
    """A rejected send must not abort /home and lose every later section. The
    fallback drops the keyboard: buttons whose lines were never delivered are
    exactly the referent-less buttons packing exists to prevent."""
    rt, interface = await build(
        tmp_path,
        [],
        telegram=TelegramConfig(allowed_user_ids=[111]),
        bot=ConfirmationRejectingBot(),
    )
    await _home(interface)
    bot: ConfirmationRejectingBot = interface._app.bot
    assert len(bot.sent) == 1
    assert "could not be delivered" in bot.sent[0]["text"]
    assert bot.sent[0]["reply_markup"] is None
    await rt.stop()


async def test_a_button_less_digest_message_is_still_plain(tmp_path: Path, token_env: str) -> None:
    rt, interface = await build(tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]))
    await interface._send_digest(555, 111, "Home\nnothing to do", [])
    assert interface._app.bot.sent == [
        {"chat_id": 555, "text": "Home\nnothing to do", "parse_mode": None, "reply_markup": None}
    ]
    await rt.stop()


def test_packing_splits_on_length_with_buttons_attached() -> None:
    """The length-driven split, with action-bearing and plain lines
    interleaved: the message limit has to be enforced while a keyboard is
    being carried, and every button must still name a line of its own message.

    The entries are handed over the way `_on_home` hands them over -- text
    only, no anchor -- because `_pack` is what writes the anchor on.
    """
    entries: list[tuple[str, dict[str, Any] | None]] = []
    for n in range(40):
        entries.append((f"note {n} " + "x" * 240, None))
        entries.append(
            (f"• item {n} " + "y" * 240, {"tool": "t", "args": {}, "label": f"{n} · Go"})
        )
    messages = _pack(entries, "Today")
    assert len(messages) > 1
    assert any(len(actions) < _MAX_ACTION_BUTTONS for _, actions in messages)  # split on length
    for text, actions in messages:
        assert len(text) <= _MAX_MESSAGE_LEN
        for action in actions:
            assert text.count(f"[{action['label']}]") == 1


def test_the_section_header_is_capped_like_any_other_line() -> None:
    """`HomeSection.title` is operator-supplied and unconstrained, and the
    header repeats on every message of the section, so an uncapped one puts
    every message of that section over the limit at once."""
    entries: list[tuple[str, dict[str, Any] | None]] = [(f"• item {n}", None) for n in range(5)]
    messages = _pack(entries, "T" * 5000)
    assert messages
    assert all(len(text) <= _MAX_MESSAGE_LEN for text, _ in messages)


def test_a_section_header_cannot_forge_an_anchor() -> None:
    """A header repeats on every message of its section, so a bracketed anchor
    written into one lands beside every button that section produces. It comes
    from the operator's own config rather than from an app, but the module
    already length-caps it, so treating it as trusted for the rest of the
    class would be arbitrary."""
    action = {"tool": "t", "args": {}, "label": "1 · Done"}
    (text, _), *rest = _pack([("• Buy milk", action)], "Morning [1 · Done]")
    assert not rest
    assert text.count("[1 · Done]") == 1
    assert text.startswith("Morning (1 · Done)")


def test_a_section_header_cannot_add_a_line() -> None:
    (text, _), *rest = _pack([("• Buy milk", None)], "A\nB")
    assert not rest
    assert text.split("\n") == ["AB", "• Buy milk"]


async def test_every_button_is_anchored_whatever_the_app_writes(
    tmp_path: Path, token_env: str
) -> None:
    """The property, against text as hostile as the contract allows: every
    button's label is present, intact, exactly once, on the line it acts on --
    for any app-supplied title, subtitle, url or label, of any length or
    content. Driven through `_on_home` against a real app, so the lines have
    the shape the code actually composes rather than one a fixture invented.
    """
    rt, interface = await build(
        tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]), orc=make_hostile_orc()
    )
    await _home(interface)
    messages = _keyboard_messages(interface._app.bot)
    assert messages
    for text, buttons in messages:
        labels = [button.text for button in buttons]
        assert len(set(labels)) == len(labels)
        for label in labels:
            assert "\n" not in label
            assert text.count(f"[{label}]") == 1
            anchored = [line for line in text.splitlines() if line.endswith(f"[{label}]")]
            assert len(anchored) == 1
        assert len(text) <= _MAX_MESSAGE_LEN
    await rt.stop()


async def test_no_app_text_can_forge_or_break_an_anchor(tmp_path: Path, token_env: str) -> None:
    """The three ways app text reaches the anchor syntax: brackets in a title
    (a decoy row that reads as actionable), brackets in a label (which would
    split its own anchor), and a Unicode line separator, which the url check
    does not catch and a client renders as a line break."""
    rt, interface = await build(
        tmp_path, [], telegram=TelegramConfig(allowed_user_ids=[111]), orc=make_hostile_orc()
    )
    await _home(interface)
    text = "\n".join(sent["text"] for sent in interface._app.bot.sent)

    decoy = next(line for line in text.splitlines() if "Decoy" in line)
    assert "[" not in decoy and "(1 · Done)" in decoy  # folded, so it cannot pose as an anchor
    assert "\u2028" not in text and "\u2029" not in text
    assert not any(ord(ch) < 32 for line in text.splitlines() for ch in line)
    # the long title is truncated, but its own anchor survives the truncation
    long_line = next(line for line in text.splitlines() if line.startswith("• LLL"))
    assert long_line.endswith("]") and " · Done]" in long_line
    await rt.stop()


def test_a_confirm_action_button_label_is_marked() -> None:
    assert _button_label({"label": "Nuke", "confirm": True}, 2) == "2 · ⚠️ Nuke"
    assert _button_label({"label": "Done", "confirm": False}, 1) == "1 · Done"
