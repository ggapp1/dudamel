from __future__ import annotations

from dudamel.config import HomeSection
from dudamel.home import compose_home


def _card(qualified_id: str) -> dict[str, object]:
    app, wid = qualified_id.split(".")
    return {
        "id": wid,
        "qualified_id": qualified_id,
        "title": wid,
        "renderer": "markdown",
        "data": "x",
        "actions": [],
    }


def test_no_sections_yields_one_untitled_section_in_registration_order() -> None:
    cards = [_card("tasks.today"), _card("notes.recent")]
    composed = compose_home(cards, [])
    assert len(composed) == 1
    assert composed[0].title is None
    assert [c["qualified_id"] for c in composed[0].cards] == ["tasks.today", "notes.recent"]


def test_sections_order_cards_and_sections() -> None:
    cards = [_card("notes.recent"), _card("tasks.today"), _card("weather.now")]
    sections = [
        HomeSection(title="Today", widgets=["tasks.today", "weather.now"]),
        HomeSection(title="Archive", widgets=["notes.recent"]),
    ]
    composed = compose_home(cards, sections)
    assert [s.title for s in composed] == ["Today", "Archive"]
    assert [c["qualified_id"] for c in composed[0].cards] == ["tasks.today", "weather.now"]


def test_an_unregistered_widget_id_is_ignored_not_fatal() -> None:
    composed = compose_home(
        [_card("tasks.today")], [HomeSection(title="Today", widgets=["tasks.today", "gone.away"])]
    )
    assert [c["qualified_id"] for c in composed[0].cards] == ["tasks.today"]


def test_an_unnamed_widget_lands_in_a_trailing_untitled_section() -> None:
    composed = compose_home(
        [_card("tasks.today"), _card("notes.recent")],
        [HomeSection(title="Today", widgets=["tasks.today"])],
    )
    assert [s.title for s in composed] == ["Today", None]
    assert [c["qualified_id"] for c in composed[1].cards] == ["notes.recent"]


def test_a_widget_named_twice_renders_in_the_first_section_only() -> None:
    composed = compose_home(
        [_card("tasks.today")],
        [
            HomeSection(title="Today", widgets=["tasks.today"]),
            HomeSection(title="Also", widgets=["tasks.today"]),
        ],
    )
    assert [s.title for s in composed] == ["Today"]


def test_a_widget_named_twice_in_one_section_renders_once() -> None:
    """A duplicated card is two live buttons for the same mutation, so a repeat
    within a single section collapses exactly as a cross-section one does."""
    composed = compose_home(
        [_card("tasks.today"), _card("notes.recent")],
        [HomeSection(title="Today", widgets=["tasks.today", "notes.recent", "tasks.today"])],
    )
    assert len(composed) == 1
    assert [c["qualified_id"] for c in composed[0].cards] == ["tasks.today", "notes.recent"]


def test_a_section_that_resolves_to_nothing_is_omitted() -> None:
    composed = compose_home([_card("tasks.today")], [HomeSection(title="Empty", widgets=["x.y"])])
    assert [s.title for s in composed] == [None]


def test_an_error_card_is_still_placed_in_its_section() -> None:
    card = {
        "id": "today",
        "qualified_id": "tasks.today",
        "title": "Today",
        "renderer": "markdown",
        "error": "boom",
        "actions": [],
    }
    composed = compose_home([card], [HomeSection(title="Today", widgets=["tasks.today"])])
    assert composed[0].cards == [card]
