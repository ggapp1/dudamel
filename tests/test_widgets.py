import pytest

from dudamel import App
from dudamel.contract.renderers import validate_widget_payload
from dudamel.exceptions import RegistryError


def test_widget_registration():
    app = App("workouts", description="d")

    @app.widget(title="This week", renderer="stat")
    async def week_volume() -> dict:
        return {"label": "Weekly volume", "value": 1200, "unit": "kg"}

    w = app.widgets["week_volume"]
    assert w.title == "This week" and w.renderer == "stat" and w.app_name == "workouts"


def test_unknown_renderer_rejected():
    app = App("workouts", description="d")
    with pytest.raises(RegistryError, match="renderer"):

        @app.widget(title="X", renderer="pie3d")
        async def bad() -> dict:
            return {}


def test_stat_payload_valid():
    out = validate_widget_payload("stat", {"label": "Vol", "value": 12.5, "unit": "kg"})
    assert out.label == "Vol"


def test_stat_payload_invalid():
    with pytest.raises(ValueError):
        validate_widget_payload("stat", {"value_only": 1})


def test_table_list_markdown_payloads():
    validate_widget_payload("table", {"columns": ["a"], "rows": [[1]]})
    validate_widget_payload("list", [{"title": "t", "url": "http://x"}])
    assert validate_widget_payload("markdown", "# hi") == "# hi"
    with pytest.raises(ValueError):
        validate_widget_payload("markdown", {"not": "str"})


def test_sync_widget_rejected():
    app = App("workouts", description="d")
    with pytest.raises(RegistryError, match="must be async"):

        @app.widget(title="X", renderer="stat")
        def sync_widget() -> dict:
            return {"label": "x", "value": 1}


def test_widget_with_required_arg_rejected():
    app = App("workouts", description="d")
    with pytest.raises(RegistryError, match="must take no arguments"):

        @app.widget(title="X", renderer="stat")
        async def needs_arg(period: str) -> dict:
            return {"label": period, "value": 1}


def test_widget_with_defaulted_arg_allowed():
    """A parameter with a default is not "required" -- widgets just never
    receive one in practice, but the registration check should only reject
    parameters the caller would actually have to supply."""
    app = App("workouts", description="d")

    @app.widget(title="X", renderer="stat")
    async def has_default(period: str = "week") -> dict:
        return {"label": period, "value": 1}

    assert "has_default" in app.widgets


def test_widget_default_timeout():
    app = App("workouts", description="d")

    @app.widget(title="X", renderer="stat")
    async def w() -> dict:
        return {"label": "x", "value": 1}

    assert app.widgets["w"].timeout == 15.0


def test_widget_custom_timeout():
    app = App("workouts", description="d")

    @app.widget(title="X", renderer="stat", timeout=2.5)
    async def w() -> dict:
        return {"label": "x", "value": 1}

    assert app.widgets["w"].timeout == 2.5


def test_duplicate_widget_id_rejected():
    app = App("workouts", description="d")

    @app.widget(title="A", renderer="stat")
    async def w() -> dict:
        return {"label": "Test", "value": 1, "unit": "kg"}

    # Attempt to register another function with the same __name__
    async def fn2() -> dict:
        return {"label": "Test2", "value": 2, "unit": "kg"}

    fn2.__name__ = "w"
    with pytest.raises(RegistryError, match="already registered"):
        app.widget(title="B", renderer="stat")(fn2)


def test_list_item_accepts_an_action() -> None:
    items = validate_widget_payload(
        "list",
        [{"title": "Buy milk", "action": {"tool": "complete", "args": {"id": 4}}}],
    )
    assert items[0].action is not None
    assert items[0].action.tool == "complete"
    assert items[0].action.args == {"id": 4}
    assert items[0].action.label is None


def test_list_item_action_is_optional() -> None:
    items = validate_widget_payload("list", [{"title": "Buy milk"}])
    assert items[0].action is None


@pytest.mark.parametrize("url", ["http://x.test/a", "https://x.test/a", "mailto:a@x.test"])
def test_list_item_allows_safe_url_schemes(url: str) -> None:
    items = validate_widget_payload("list", [{"title": "t", "url": url}])
    assert items[0].url == url


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        "  javascript:alert(1)",
        "java\tscript:alert(1)",
        "data:text/html;base64,PHN2Zz4=",
        "vbscript:msgbox(1)",
        "/relative/path",
    ],
)
def test_list_item_rejects_unsafe_url_schemes(url: str) -> None:
    """Browsers strip ASCII control characters from a URL before parsing its
    scheme, so `java\\tscript:` is a live bypass of any validator that does not
    strip them first. Relative URLs are rejected too: a widget that links is
    linking off-page, so allowing one would only widen the surface."""
    with pytest.raises(ValueError, match="url"):
        validate_widget_payload("list", [{"title": "t", "url": url}])
