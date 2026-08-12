import pytest
from pydantic import BaseModel, Field

from dudamel import App
from dudamel.exceptions import AppSettingsError, RuntimeNotBoundError


class Weather(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    units: str = "metric"


def test_settings_available_after_binding() -> None:
    app = App("weather", description="d", settings=Weather)
    app.bind_settings({"latitude": 52.52})
    assert app.settings.latitude == 52.52
    assert app.settings.units == "metric"


def test_settings_before_binding_raises() -> None:
    app = App("weather", description="d", settings=Weather)
    with pytest.raises(RuntimeNotBoundError, match="settings accessed before load"):
        _ = app.settings


def test_unknown_key_rejected_even_without_extra_forbid() -> None:
    """Weather does NOT set model_config extra='forbid'. The framework, not the
    app author, is what makes a typo an error instead of a silent no-op."""
    app = App("weather", description="d", settings=Weather)
    with pytest.raises(AppSettingsError, match="latitide"):
        app.bind_settings({"latitude": 1.0, "latitide": 2.0})


def test_validation_failure_names_app_and_field() -> None:
    app = App("weather", description="d", settings=Weather)
    with pytest.raises(AppSettingsError) as excinfo:
        app.bind_settings({"latitude": 999.0})
    message = str(excinfo.value)
    assert "weather" in message and "latitude" in message


def test_app_without_model_rejects_any_key() -> None:
    app = App("tasks", description="d")
    with pytest.raises(AppSettingsError, match="takes no settings"):
        app.bind_settings({"foo": 1})


def test_app_without_model_accepts_empty_block() -> None:
    app = App("tasks", description="d")
    app.bind_settings({})
    assert app.settings_model is None


def test_alias_accepted_and_unknown_key_still_rejected() -> None:
    class Aliased(BaseModel):
        api_key: str = Field(alias="key")

    app = App("svc", description="d", settings=Aliased)
    app.bind_settings({"key": "abc"})
    assert app.settings.api_key == "abc"
    with pytest.raises(AppSettingsError, match="nope"):
        App("svc", description="d", settings=Aliased).bind_settings({"key": "a", "nope": 1})
