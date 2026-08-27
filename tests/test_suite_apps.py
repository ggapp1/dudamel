"""Cross-app properties of the shipped suite.

These grow for free with every app added in 6b-2 and 6c, which is why they are
parametrised over `SUITE_APPS` rather than written per app.
"""

import importlib

import pytest

from dudamel.apps import SUITE_APPS

ENTRIES = sorted(SUITE_APPS.values(), key=lambda entry: entry.name)
# An empty registry would SKIP every parametrised test below rather than fail
# them: pytest's default empty_parameter_set_mark is "skip".
assert ENTRIES, "SUITE_APPS is empty; every parametrised test below would skip"

# The action-labelled tools each app is expected to expose, and therefore the
# exact set its widgets may reference. Written out per app rather than derived,
# because a property computed from the same source it checks proves nothing.
EXPECTED_ACTIONS = {
    "tasks": {"complete_task"},
    "notes": set(),
    "habits": {"tick_habit", "untick_habit"},
}


def test_the_suite_ships_the_core_three():
    assert {entry.name for entry in ENTRIES} >= {"tasks", "notes", "habits"}


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_entry_module_imports_and_names_match(entry):
    assert importlib.import_module(entry.module).app.name == entry.name


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
async def test_action_labels_and_widget_references_agree(entry, seeded_app):
    """Both directions.

    A tool carrying `action=` that no widget ever offers is dead config; a
    widget naming a tool that is not action-labelled fails to resolve at render
    time (`widgets._resolve`). Subset-checking would catch only one direction,
    and running the widgets against an EMPTY database would catch neither --
    every empty state is actionless by design, so the referenced set would be
    empty and the assertion vacuous. Hence `seeded_app`.

    `notes` legitimately has both sets empty: it is the read-only archive card.
    """
    from dudamel.widgets import run_widget

    app = seeded_app
    labelled = {name for name, tool in app.tools.items() if tool.action is not None}
    referenced = set()
    for widget in app.widgets.values():
        card = await run_widget(widget, {n: app.tools[n] for n in labelled})
        assert "data" in card, card.get("error")
        referenced |= {
            item["action"]["tool"] for item in card["data"] if item.get("action") is not None
        }
        referenced |= {action["tool"] for action in card["actions"]}

    assert referenced == EXPECTED_ACTIONS[entry.name]
    assert labelled == EXPECTED_ACTIONS[entry.name]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_destructive_tools_are_never_reachable_from_the_button_plane(entry):
    """`POST /api/action/{tool}` 404s any tool without an `action` label, so an
    unlabelled destructive tool is structurally unreachable from one tap."""
    app = importlib.import_module(entry.module).app
    for name, tool in app.tools.items():
        if name.startswith("delete_"):
            assert tool.action is None, f"{entry.name}.{name} is one tap from deletion"
            assert tool.confirm is True, f"{entry.name}.{name} is not behind a confirm"


# --- web / Telegram parity ---------------------------------------------------

HOME = [
    {"title": "Today", "widgets": ["tasks.today", "habits.today"]},
    {"title": "Archive", "widgets": ["notes.recent"]},
]


def _migrate_everything(db_url: str) -> None:
    """Apply the core lane and every suite lane, as `dudamel db migrate` does.

    `Runtime.start()` runs no migrations; production applies them separately.
    Using the real lanes rather than `metadata.create_all` keeps this test
    faithful to what a user's database actually looks like.
    """
    from dudamel.apps import SUITE_APPS, suite_versions_dir
    from dudamel.migrate import upgrade_core, upgrade_suite_app

    upgrade_core(db_url)
    for name, entry in SUITE_APPS.items():
        upgrade_suite_app(db_url, name, suite_versions_dir(entry))


def _order_of(haystack: str, needles: list[str]) -> list[str]:
    """`needles` sorted by where each first appears in `haystack`.

    Compares the two surfaces by rendered ORDER rather than by parsing two
    different markup languages into a common shape. A surface that reorders,
    drops a card, or renders `render_widgets()` instead of `render_home()`
    produces a different sequence here.
    """
    positions = {n: haystack.index(n) for n in needles if n in haystack}
    return sorted(positions, key=positions.get)


async def test_the_web_and_telegram_render_the_same_home_in_the_same_order(tmp_path, monkeypatch):
    """Acceptance criterion 5, driven through BOTH surfaces.

    `compose_home` is shared, but sharing it is not what this pins -- asserting
    on `compose_home` alone would only re-test `tests/test_home.py` and would
    compose the very card list it checks. Both surfaces are rendered here from
    one Runtime, and their section and card order compared.
    """
    import test_telegram
    import test_web_ui

    from dudamel.apps.habits import app as habits_app
    from dudamel.apps.notes import app as notes_app
    from dudamel.apps.tasks import app as tasks_app
    from dudamel.config import HomeConfig, TelegramConfig
    from dudamel.orchestrator import Orchestrator

    monkeypatch.setenv("DUDAMEL_WEB_TOKEN", test_web_ui.TOKEN)
    monkeypatch.setenv("DUDAMEL_TELEGRAM_TOKEN", test_telegram.TOKEN)
    home = HomeConfig(section=HOME)
    orc = Orchestrator(apps=[tasks_app, notes_app, habits_app])
    # `Runtime.start()` does NOT bind app settings -- `resolve_apps` does that
    # during config resolution, and this test constructs the Orchestrator
    # directly. Without this, every widget raises RuntimeNotBoundError and both
    # surfaces render error cards, which would compare equal and pass.
    for suite_app in (tasks_app, notes_app, habits_app):
        suite_app.bind_settings({})

    # --- web -----------------------------------------------------------------
    (tmp_path / "web").mkdir(exist_ok=True)
    settings = test_web_ui.make_settings(tmp_path / "web").model_copy(update={"home": home})
    _migrate_everything(settings.database_url)
    web_rt = test_web_ui.Runtime(
        orc, settings, providers={"standard": test_web_ui.FakeProvider([])}
    )
    await web_rt.start()
    api = test_web_ui.create_api(web_rt, settings)
    test_web_ui.add_ui(api, web_rt, settings)
    transport = test_web_ui.httpx.ASGITransport(app=api)
    async with test_web_ui.client(transport) as http:
        await test_web_ui.login(http, test_web_ui.TOKEN)
        page = (await http.get("/")).text
    await web_rt.stop()

    # --- telegram ------------------------------------------------------------
    bot = test_telegram.FakeBot()
    (tmp_path / "tg").mkdir(exist_ok=True)
    tg_settings = test_telegram.make_settings(
        tmp_path / "tg", telegram=TelegramConfig(allowed_user_ids=[111])
    ).model_copy(update={"home": home})
    _migrate_everything(tg_settings.database_url)
    tg_rt = test_telegram.Runtime(
        orc, tg_settings, providers={"standard": test_telegram.FakeProvider([])}
    )
    await tg_rt.start()
    interface = test_telegram.TelegramInterface(tg_rt, tg_settings)
    interface._app.bot = bot
    await test_telegram._home(interface)
    digest = "\n".join(str(sent) for sent in bot.sent)
    await tg_rt.stop()

    # --- compare -------------------------------------------------------------
    landmarks = ["Today", "Habits", "Archive", "Recent notes"]
    web_order = _order_of(page, landmarks)
    digest_order = _order_of(digest, landmarks)

    assert web_order, "the dashboard rendered none of the expected sections or cards"
    assert digest_order, "the digest rendered none of the expected sections or cards"
    assert web_order == digest_order
