"""Homescreen layout: grouping and ordering rendered cards into sections.

One pure function over data, shared by the web dashboard and the Telegram
digest so the two surfaces cannot drift into different orderings. It knows
nothing about Runtime, HTTP or Telegram, and never runs a widget.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from dudamel.config import HomeSection


class ComposedSection(NamedTuple):
    title: str | None  # None = the trailing default section
    cards: list[dict[str, Any]]


def compose_home(cards: list[dict[str, Any]], sections: list[HomeSection]) -> list[ComposedSection]:
    """Group `cards` (as returned by `widgets.run_widget`) into `sections`.

    Three rules, each chosen so that a configuration mistake degrades rather
    than breaks:

    - A configured widget id that is not registered is skipped. It must not
      be fatal: a widget legitimately disappears when its app is disabled,
      and app activation is exactly the thing operators toggle.
    - A registered widget named in no section falls into a trailing untitled
      section, so enabling an app can never make its cards invisible.
    - No sections at all yields one untitled section in registration order --
      precisely the behaviour that predates layout config.

    A widget named in two sections renders in the first. A section that
    resolves to no cards is omitted rather than rendered empty.
    """
    by_id = {card["qualified_id"]: card for card in cards}
    composed: list[ComposedSection] = []
    placed: set[str] = set()
    for section in sections:
        chosen = [by_id[wid] for wid in section.widgets if wid in by_id and wid not in placed]
        placed.update(card["qualified_id"] for card in chosen)
        if chosen:
            composed.append(ComposedSection(title=section.title, cards=chosen))
    remaining = [card for card in cards if card["qualified_id"] not in placed]
    if remaining:
        composed.append(ComposedSection(title=None, cards=remaining))
    return composed
