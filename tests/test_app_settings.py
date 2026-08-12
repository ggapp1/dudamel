import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

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
    with pytest.raises(AppSettingsError) as excinfo:
        app.bind_settings({"latitude": 1.0, "latitide": 2.0})
    message = str(excinfo.value)
    assert "latitide" in message
    # the hint half of the message is what lets an operator fix the typo
    assert "known: latitude, units" in message


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


def test_plain_alias_replaces_the_field_name() -> None:
    """With `alias=` and no population by name, pydantic accepts only the alias.
    The framework must reject the field name here rather than advertise it and
    then let model_validate fail on a key the user never wrote."""

    class Aliased(BaseModel):
        api_key: str = Field(alias="key")

    app = App("svc", description="d", settings=Aliased)
    with pytest.raises(AppSettingsError) as excinfo:
        app.bind_settings({"api_key": "abc"})
    message = str(excinfo.value)
    assert "unknown setting(s) api_key" in message
    assert "known: key" in message
    assert "api_key" not in message.split("known:")[1]


def test_population_by_name_accepts_both_spellings() -> None:
    class Aliased(BaseModel):
        model_config = ConfigDict(populate_by_name=True)

        api_key: str = Field(alias="key")

    App("svc", description="d", settings=Aliased).bind_settings({"key": "a"})
    app = App("svc", description="d", settings=Aliased)
    app.bind_settings({"api_key": "b"})
    assert app.settings.api_key == "b"


def test_validation_alias_accepted() -> None:
    class Aliased(BaseModel):
        api_key: str = Field(validation_alias="key")

    app = App("svc", description="d", settings=Aliased)
    app.bind_settings({"key": "abc"})
    assert app.settings.api_key == "abc"


def test_alias_choices_each_accepted() -> None:
    class Choosy(BaseModel):
        api_key: str = Field(validation_alias=AliasChoices("key", "token"))

    for spelling in ("key", "token"):
        app = App("svc", description="d", settings=Choosy)
        app.bind_settings({spelling: "abc"})
        assert app.settings.api_key == "abc"
    with pytest.raises(AppSettingsError, match="api_key"):
        App("svc", description="d", settings=Choosy).bind_settings({"api_key": "abc"})


def test_path_alias_accepts_its_top_level_key_and_not_the_field_name() -> None:
    """A path alias is looked up under its first element. The field name is not
    accepted, so the framework must not advertise it as a fallback."""

    class Pathed(BaseModel):
        api_key: str = Field(validation_alias=AliasPath("creds", 0))

    app = App("svc", description="d", settings=Pathed)
    app.bind_settings({"creds": ["abc"]})
    assert app.settings.api_key == "abc"

    with pytest.raises(AppSettingsError) as excinfo:
        App("svc", description="d", settings=Pathed).bind_settings({"api_key": "abc"})
    message = str(excinfo.value)
    assert "unknown setting(s) api_key" in message
    assert "known: creds" in message


def test_alias_choices_mixing_a_path_accepts_both_forms() -> None:
    class Mixed(BaseModel):
        api_key: str = Field(validation_alias=AliasChoices("key", AliasPath("creds", 0)))

    plain = App("svc", description="d", settings=Mixed)
    plain.bind_settings({"key": "abc"})
    assert plain.settings.api_key == "abc"

    pathed = App("svc", description="d", settings=Mixed)
    pathed.bind_settings({"creds": ["xyz"]})
    assert pathed.settings.api_key == "xyz"

    with pytest.raises(AppSettingsError, match="unknown setting\\(s\\) api_key"):
        App("svc", description="d", settings=Mixed).bind_settings({"api_key": "abc"})
