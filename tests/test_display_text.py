"""App-authored display text is sanitized at the contract, on every surface.

`ItemAction.label` has long been cleaned; the text sitting next to the
button was not. Telegram stripped it (`telegram._plain`) and the web did not
(`dashboard.html` renders `{{ item.title }}` in the same `<li>` as the button),
which is the same surface disagreement already fixed one field to the left.
"""

import pytest

from dudamel.contract.renderers import (
    DISPLAY_TEXT_MAX,
    clean_action_label,
    clean_display_text,
    validate_widget_payload,
)


def _one(title: str = "t", subtitle: str | None = None):
    """One validated ListItem, straight through the renderer contract."""
    return validate_widget_payload("list", [{"title": title, "subtitle": subtitle}])[0]


def test_a_bidi_override_is_stripped_from_a_title():
    """U+202E reorders everything after it, so a title reading "Archive" can sit
    on a button wired to delete -- and the same string reaches window.confirm."""
    item = _one(title="Buy milk ‮)etucexE( sutats kcehC")

    assert "‮" not in item.title


def test_a_bidi_override_is_stripped_from_a_subtitle():
    """A subtitle sits on the same line as the title and the button."""
    item = _one(subtitle="due ‮yadretsey")

    assert "‮" not in item.subtitle


def test_a_title_that_cleans_to_nothing_is_rejected():
    """A row whose title is entirely control characters has no honest rendering,
    and an empty title on a card with a button is a button labelling nothing."""
    import pytest

    with pytest.raises(ValueError, match="title"):
        _one(title="‮‭  ")


def test_a_subtitle_that_cleans_to_nothing_becomes_none():
    """A subtitle is optional, so dropping it is the honest degrade -- unlike a
    title, losing it costs nothing and failing the whole card would cost the row."""
    assert _one(subtitle="‮  ").subtitle is None


def test_an_overlong_title_is_truncated_not_rejected():
    """The opposite rule to an action label, deliberately.

    A too-long label is an app bug -- the author typed it -- so it raises. A
    too-long title is ordinary data the user or the model supplied, and losing
    the whole card over it is worse than an ellipsis.
    """
    from dudamel.contract.renderers import DISPLAY_TEXT_MAX

    item = _one(title="x" * (DISPLAY_TEXT_MAX + 500))

    assert len(item.title) == DISPLAY_TEXT_MAX
    assert item.title.endswith("…")


def test_an_overlong_subtitle_is_truncated_too():
    from dudamel.contract.renderers import DISPLAY_TEXT_MAX

    assert len(_one(subtitle="y" * (DISPLAY_TEXT_MAX + 10)).subtitle) == DISPLAY_TEXT_MAX


def test_both_surfaces_remove_exactly_the_same_character_class():
    """A regression guard, not a driven test -- the two classes already agree.

    They were byte-identical duplicates kept in sync by a comment, and a comment
    is what let the *fields* they apply to drift apart in the first place. This
    sweeps every codepoint rather than asserting the constants are equal, so it
    still holds if either side is rewritten in another form.

    Scoped to the character class alone: telegram's `_plain` additionally strips
    whitespace and folds brackets to parentheses (its anchor syntax), which are
    surface-specific and deliberately not shared.
    """
    from dudamel.contract.renderers import _UNSAFE_LABEL_CHARS
    from dudamel.interfaces.telegram import _UNSAFE_DIGEST_CHARS

    mismatched = [
        hex(cp)
        for cp in range(0x110000)
        if bool(_UNSAFE_LABEL_CHARS.fullmatch(chr(cp)))
        != bool(_UNSAFE_DIGEST_CHARS.fullmatch(chr(cp)))
    ]

    assert mismatched == []


def test_a_stat_cards_text_fields_are_cleaned_too():
    """A stat card carries card-level buttons like any other, so its label sits
    in the same relationship to a live control that a list title does."""
    payload = validate_widget_payload(
        "stat", {"label": "Weekly ‮emulov", "value": "12 ‮gk", "unit": "‮gk"}
    )

    assert "‮" not in payload.label
    assert "‮" not in str(payload.value)
    assert "‮" not in payload.unit


def test_a_tables_headers_and_string_cells_are_cleaned():
    """Telegram already strips table cells (`_plain` over every fragment); the
    web did not. Non-string cells pass through untouched -- there is nothing to
    reorder in a number, and coercing them here would change the payload's type.
    """
    payload = validate_widget_payload(
        "table", {"columns": ["Exer‮esic"], "rows": [["Squ‮ta", 5, 3.5, None]]}
    )

    assert "‮" not in payload.columns[0]
    assert "‮" not in payload.rows[0][0]
    assert payload.rows[0][1:] == [5, 3.5, None]


# Every range in UNSAFE_DISPLAY_CHARS, as (first, last): the codepoints at and
# just inside each edge must go, the ones just outside each edge must stay.
_STRIPPED = [
    0x00,  # NUL, first of C0
    0x09,  # TAB: inside the class even though str.strip() would also take it at an edge
    0x1F,  # US, last of C0
    0x7F,  # DEL
    0x80,  # first of C1
    0x9F,  # APC, last of C1
    0x2028,  # LINE SEPARATOR
    0x2029,  # PARAGRAPH SEPARATOR
    0x202A,  # LRE, first of the embedding/override run
    0x202D,
    0x202E,  # RLO, last of the run
    0x2066,  # LRI, first of the isolates
    0x2069,  # PDI, last of the isolates
]
_SURVIVING = [
    0x20,  # SPACE: one past the end of C0
    0x7E,  # TILDE: one before DEL
    0xA0,  # NBSP: one past the end of C1
    0x2027,  # one before LINE SEPARATOR
    0x202F,  # NARROW NBSP: one past RLO
    0x2065,  # one before LRI
    0x206A,  # one past PDI
    0x200E,  # LRM -- deliberately NOT in the class (see the comment on it)
    0x200F,  # RLM -- likewise
]


@pytest.mark.parametrize("cp", _STRIPPED, ids=lambda cp: f"U+{cp:04X}")
@pytest.mark.parametrize("clean", [clean_action_label, clean_display_text])
def test_every_edge_of_the_unsafe_class_is_stripped(cp: int, clean) -> None:
    assert clean(f"Ar{chr(cp)}chive") == "Archive"


@pytest.mark.parametrize("cp", _SURVIVING, ids=lambda cp: f"U+{cp:04X}")
@pytest.mark.parametrize("clean", [clean_action_label, clean_display_text])
def test_characters_just_outside_the_unsafe_class_survive(cp: int, clean) -> None:
    """The other half of a boundary: an over-wide class would delete legitimate
    text (a space between words, a non-breaking space in a price)."""
    text = f"Ar{chr(cp)}chive"
    assert clean(text) == text


def test_overrides_are_removed_before_the_outer_whitespace_is_trimmed() -> None:
    """Order matters: stripping first would leave the space that only becomes
    outer once the override in front of it is gone."""
    assert clean_action_label("\u202e Archive") == "Archive"
    assert clean_display_text("\u202e Archive \u202e") == "Archive"


def test_display_text_is_truncated_only_past_the_maximum() -> None:
    exact = "x" * DISPLAY_TEXT_MAX
    assert clean_display_text(exact) == exact

    over = clean_display_text("x" * (DISPLAY_TEXT_MAX + 1))
    assert over == "x" * (DISPLAY_TEXT_MAX - 1) + "\u2026"
    assert len(over) == DISPLAY_TEXT_MAX


# --- markdown bodies, widget titles and errors -----------------------------------


def test_a_markdown_body_loses_its_bidi_controls_but_keeps_its_layout():
    """A markdown card can carry a card-level button (`actions=[...]`), so its
    body sits beside a live control like any list title does. Newlines and tabs
    ARE the document and must survive; a bare CR must not become a line break."""
    body = validate_widget_payload("markdown", "# Wipe ‮etelp\n\n- a\tb\r\n- c\rd\x00e f")

    assert body == "# Wipe etelp\n\n- a\tb\n- cde" + "f"


def test_the_markdown_class_is_the_display_class_minus_layout_whitespace():
    """Held in lockstep with UNSAFE_DISPLAY_CHARS by sweeping every codepoint,
    so widening the display class cannot silently leave markdown behind."""
    from dudamel.contract.renderers import UNSAFE_DISPLAY_CHARS, UNSAFE_MARKDOWN_CHARS

    layout = {"\n", "\t"}
    mismatched = [
        hex(cp)
        for cp in range(0x110000)
        if bool(UNSAFE_MARKDOWN_CHARS.fullmatch(chr(cp)))
        != (bool(UNSAFE_DISPLAY_CHARS.fullmatch(chr(cp))) and chr(cp) not in layout)
    ]

    assert mismatched == []


async def test_widget_title_and_error_are_cleaned_for_both_surfaces():
    """The card the web API serves and the one Telegram renders are the same
    dict, so cleaning at the producer is the one place that covers both."""
    from dudamel.contract.types import Widget
    from dudamel.widgets import run_widget

    async def boom() -> str:
        raise RuntimeError("bad ‮noitelet")

    async def fine() -> str:
        return "ok"

    for fn, key in ((boom, "error"), (fine, "data")):
        widget = Widget(
            id="w", app_name="a", title="Ar‮hcive", renderer="markdown", fn=fn, timeout=5.0
        )
        card = await run_widget(widget)
        assert key in card
        assert "‮" not in card["title"] and card["title"] == "Arhcive"
        if key == "error":
            assert "‮" not in card["error"]


async def test_web_and_telegram_drop_the_same_characters_from_a_markdown_card():
    from dudamel.contract.types import Widget
    from dudamel.interfaces.telegram import _card_lines
    from dudamel.widgets import run_widget

    hostile = "a‮b⁦c\x07d e"

    async def fn() -> str:
        return f"line1\n{hostile}"

    card = await run_widget(
        Widget(id="w", app_name="a", title="T", renderer="markdown", fn=fn, timeout=5.0)
    )
    web_text = card["data"]
    telegram_text = "".join(_card_lines(card))

    for ch in "‮⁦\x07 ":
        assert ch not in web_text and ch not in telegram_text
    assert "\n" in web_text
    assert telegram_text == web_text.replace("\n", "")
