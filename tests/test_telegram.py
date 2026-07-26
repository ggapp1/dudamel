"""Telegram interface tests. No network: handlers are called directly with
hand-built telegram.Update/Message/User/CallbackQuery objects, and
`TelegramInterface._app.bot` is swapped for a stub that records calls
instead of hitting the Bot API.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from telegram import CallbackQuery, Chat, InlineKeyboardMarkup, Message, Update, User
from telegram.error import BadRequest

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TelegramConfig, TierConfig
from dudamel.interfaces.telegram import TelegramInterface, resolve_token
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
    normally — exercises `_send_confirmation`'s BadRequest-degrade retry
    (fix wave item 3), independent of message length."""

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
    BadRequest-degrade retry (fix wave item 3)."""

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
) -> tuple[Runtime, TelegramInterface]:
    orc = make_orc()
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


# --- confirmation hardening (fix wave item 3) -------------------------------------


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
    assert any(
        "telegram handler error" in r.message and "boom" in r.message for r in caplog.records
    )
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
