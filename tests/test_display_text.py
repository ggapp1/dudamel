"""App-authored display text is sanitized at the contract, on every surface.

`ItemAction.label` has long been cleaned; the text sitting next to the
button was not. Telegram stripped it (`telegram._plain`) and the web did not
(`dashboard.html` renders `{{ item.title }}` in the same `<li>` as the button),
which is the same surface disagreement already fixed one field to the left.
"""

from dudamel.contract.renderers import validate_widget_payload


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
