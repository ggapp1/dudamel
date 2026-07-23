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
