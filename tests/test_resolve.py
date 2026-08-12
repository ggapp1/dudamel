import importlib
import itertools
import sys
import textwrap
from pathlib import Path

import pytest

from dudamel import App, Orchestrator
from dudamel.apps import SuiteApp
from dudamel.config import Settings
from dudamel.exceptions import AppResolutionError
from dudamel.resolve import resolve_apps

# Each fake suite app gets its own package. A package is cached in sys.modules
# under its own name, and the resolver deliberately purges only the app module
# (purging a parent would mean re-importing `dudamel` itself for a real suite
# app) -- so a shared package name would resolve every test's app module
# through the FIRST test's directory.
_suite_packages = itertools.count()


@pytest.fixture(autouse=True)
def _drop_fake_suite_modules():
    yield
    for name in [n for n in sys.modules if n.split(".", 1)[0].startswith("fake_suite")]:
        del sys.modules[name]


def write_suite_app(tmp_path: Path, monkeypatch, name: str, body: str) -> SuiteApp:
    """Write a fake suite app module, make it importable, and return its entry."""
    pkg_name = f"fake_suite{next(_suite_packages)}"
    pkg = tmp_path / pkg_name
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").touch()
    (pkg / f"{name}.py").write_text(textwrap.dedent(body))
    if str(tmp_path) not in sys.path:
        monkeypatch.syspath_prepend(str(tmp_path))
    # Unconditionally, not just on the first call: the import system caches a
    # directory listing per sys.path entry, so a package written into a
    # directory that is ALREADY on the path is invisible without this.
    # `syspath_prepend` does it for us on the first call and not after.
    importlib.invalidate_caches()
    versions = tmp_path / f"{name}_versions"
    versions.mkdir(exist_ok=True)
    return SuiteApp(
        name=name,
        module=f"{pkg_name}.{name}",
        summary=f"{name} summary",
        versions_dir=versions,
    )


def register(monkeypatch, *entries: SuiteApp) -> None:
    monkeypatch.setattr("dudamel.apps.SUITE_APPS", {e.name: e for e in entries}, raising=True)


DEMO_TEMPLATE = '''
from pydantic import BaseModel
from dudamel import App


class DemoSettings(BaseModel):
    city: str = "here"


app = App("{name}", description="{name} summary", settings=DemoSettings)


@app.tool
async def ping() -> str:
    """Ping."""
    return "pong"
'''

DEMO = DEMO_TEMPLATE.format(name="demo")


def settings_for(tmp_path: Path, toml: str) -> Settings:
    (tmp_path / "dudamel.toml").write_text(toml)
    return Settings.load(tmp_path)


def test_enabled_suite_app_is_resolved(tmp_path, monkeypatch) -> None:
    entry = write_suite_app(tmp_path, monkeypatch, "demo", DEMO)
    register(monkeypatch, entry)
    settings = settings_for(tmp_path, "[apps.demo]\nenabled = true\ncity = 'lisbon'\n")
    resolution = resolve_apps(Orchestrator(apps=[]), settings, strict=True)
    assert [a.name for a in resolution.apps] == ["demo"]
    assert resolution.apps[0].settings.city == "lisbon"
    assert resolution.suite_lanes == [("demo", entry.versions_dir)]
    assert resolution.local_apps == []


def test_enabled_defaults_true_when_omitted(tmp_path, monkeypatch) -> None:
    register(monkeypatch, write_suite_app(tmp_path, monkeypatch, "demo", DEMO))
    settings = settings_for(tmp_path, "[apps.demo]\ncity = 'porto'\n")
    assert [a.name for a in resolve_apps(Orchestrator(), settings, strict=True).apps] == ["demo"]


def test_suite_app_without_a_block_is_off(tmp_path, monkeypatch) -> None:
    """Presence means enabled: no block at all, no app."""
    entry = write_suite_app(tmp_path, monkeypatch, "demo", DEMO)
    register(monkeypatch, entry)
    resolution = resolve_apps(Orchestrator(), settings_for(tmp_path, ""), strict=True)
    assert resolution.apps == []
    assert resolution.suite_lanes == []
    assert entry.module not in sys.modules


def test_disabled_suite_app_is_not_imported(tmp_path, monkeypatch) -> None:
    entry = write_suite_app(tmp_path, monkeypatch, "demo", DEMO)
    register(monkeypatch, entry)
    settings = settings_for(tmp_path, "[apps.demo]\nenabled = false\ncity = 12345\n")
    resolution = resolve_apps(Orchestrator(), settings, strict=True)
    assert resolution.apps == []
    # The garbage `city` value is never validated, because the module is never
    # imported -- the honest cost of lazy loading, pinned so it cannot drift.
    assert resolution.errors == []
    assert entry.module not in sys.modules


def test_unknown_app_name_strict_raises(tmp_path, monkeypatch) -> None:
    register(monkeypatch)
    settings = settings_for(tmp_path, "[apps.note]\nenabled = true\n")
    with pytest.raises(AppResolutionError, match="note"):
        resolve_apps(Orchestrator(), settings, strict=True)


def test_unknown_app_name_diagnostic_collects(tmp_path, monkeypatch) -> None:
    register(monkeypatch)
    settings = settings_for(tmp_path, "[apps.note]\nenabled = true\n")
    resolution = resolve_apps(Orchestrator(), settings, strict=False)
    assert [e.app for e in resolution.errors] == ["note"]
    assert resolution.errors[0].stage == 1


def test_missing_dependency_reports_extra_without_importing(tmp_path, monkeypatch) -> None:
    entry = write_suite_app(tmp_path, monkeypatch, "demo", "raise AssertionError('imported!')")
    entry = SuiteApp(
        name=entry.name,
        module=entry.module,
        summary=entry.summary,
        extra="demo",
        requires=("dudamel_absent_dep",),
        versions_dir=entry.versions_dir,
    )
    register(monkeypatch, entry)
    settings = settings_for(tmp_path, "[apps.demo]\nenabled = true\n")
    resolution = resolve_apps(Orchestrator(), settings, strict=False)
    assert "pip install dudamel[demo]" in resolution.errors[0].message
    assert resolution.errors[0].stage == 1
    assert entry.module not in sys.modules


def test_undeclared_import_error_keeps_its_own_message(tmp_path, monkeypatch) -> None:
    register(
        monkeypatch,
        write_suite_app(tmp_path, monkeypatch, "demo", "import dudamel_typo_module"),
    )
    settings = settings_for(tmp_path, "[apps.demo]\nenabled = true\n")
    resolution = resolve_apps(Orchestrator(), settings, strict=False)
    assert resolution.errors[0].stage == 2
    assert "dudamel_typo_module" in resolution.errors[0].message
    assert "pip install" not in resolution.errors[0].message


def test_import_failure_strict_raises(tmp_path, monkeypatch) -> None:
    register(
        monkeypatch,
        write_suite_app(tmp_path, monkeypatch, "demo", "import dudamel_typo_module"),
    )
    settings = settings_for(tmp_path, "[apps.demo]\nenabled = true\n")
    with pytest.raises(AppResolutionError, match="dudamel_typo_module"):
        resolve_apps(Orchestrator(), settings, strict=True)


def test_import_calling_sys_exit_is_still_collected(tmp_path, monkeypatch) -> None:
    """A module that kills the interpreter at import must not take `doctor`
    with it: diagnostic mode never raises."""
    register(
        monkeypatch,
        write_suite_app(tmp_path, monkeypatch, "demo", "import sys\nsys.exit(3)\n"),
    )
    settings = settings_for(tmp_path, "[apps.demo]\nenabled = true\n")
    resolution = resolve_apps(Orchestrator(), settings, strict=False)
    assert [(e.app, e.stage) for e in resolution.errors] == [("demo", 2)]
    assert resolution.apps == []


def test_suite_module_without_an_app_object_is_stage_two(tmp_path, monkeypatch) -> None:
    register(monkeypatch, write_suite_app(tmp_path, monkeypatch, "demo", "app = 'not an App'"))
    settings = settings_for(tmp_path, "[apps.demo]\nenabled = true\n")
    resolution = resolve_apps(Orchestrator(), settings, strict=False)
    assert resolution.errors[0].stage == 2
    assert "module-level `app`" in resolution.errors[0].message
    assert resolution.apps == []


def test_settings_failure_is_stage_three(tmp_path, monkeypatch) -> None:
    register(monkeypatch, write_suite_app(tmp_path, monkeypatch, "demo", DEMO))
    settings = settings_for(tmp_path, "[apps.demo]\nenabled = true\nnope = 1\n")
    resolution = resolve_apps(Orchestrator(), settings, strict=False)
    assert resolution.errors[0].stage == 3
    assert resolution.apps == []


def test_settings_failure_strict_raises(tmp_path, monkeypatch) -> None:
    register(monkeypatch, write_suite_app(tmp_path, monkeypatch, "demo", DEMO))
    settings = settings_for(tmp_path, "[apps.demo]\nenabled = true\nnope = 1\n")
    with pytest.raises(AppResolutionError, match="nope"):
        resolve_apps(Orchestrator(), settings, strict=True)


def test_local_app_runs_without_a_config_block(tmp_path, monkeypatch) -> None:
    register(monkeypatch)
    local = App("mine", description="d")
    settings = settings_for(tmp_path, "")
    resolution = resolve_apps(Orchestrator(apps=[local]), settings, strict=True)
    assert [a.name for a in resolution.apps] == ["mine"]
    assert [a.name for a in resolution.local_apps] == ["mine"]


def test_local_app_disabled_by_config(tmp_path, monkeypatch) -> None:
    register(monkeypatch)
    local = App("mine", description="d")
    settings = settings_for(tmp_path, "[apps.mine]\nenabled = false\n")
    resolution = resolve_apps(Orchestrator(apps=[local]), settings, strict=True)
    assert resolution.apps == []


@pytest.mark.parametrize(
    "block", ["", "[apps.demo]\nenabled = false\n", "[apps.demo]\nenabled = true\n"]
)
def test_local_app_may_not_take_a_reserved_name(tmp_path, monkeypatch, block) -> None:
    """Unconditional: the rule must not depend on the suite entry's state."""
    register(monkeypatch, write_suite_app(tmp_path, monkeypatch, "demo", DEMO))
    local = App("demo", description="mine")
    settings = settings_for(tmp_path, block)
    with pytest.raises(AppResolutionError, match="rename"):
        resolve_apps(Orchestrator(apps=[local]), settings, strict=True)


def test_second_resolution_does_not_mutate_the_first(tmp_path, monkeypatch) -> None:
    """A suite module's `app` is a module global, so two resolutions in one
    process must not end up sharing (and reconfiguring) the same object."""
    register(monkeypatch, write_suite_app(tmp_path, monkeypatch, "demo", DEMO))
    first = resolve_apps(
        Orchestrator(), settings_for(tmp_path, "[apps.demo]\ncity = 'lisbon'\n"), strict=True
    )
    second = resolve_apps(
        Orchestrator(), settings_for(tmp_path, "[apps.demo]\ncity = 'porto'\n"), strict=True
    )
    assert first.apps[0].settings.city == "lisbon"
    assert second.apps[0].settings.city == "porto"
    assert first.apps[0] is not second.apps[0]


def test_local_app_settings_failure_is_stage_three(tmp_path, monkeypatch) -> None:
    register(monkeypatch)
    local = App("mine", description="d")
    settings = settings_for(tmp_path, "[apps.mine]\nnope = 1\n")
    resolution = resolve_apps(Orchestrator(apps=[local]), settings, strict=False)
    assert [(e.app, e.stage) for e in resolution.errors] == [("mine", 3)]
    assert resolution.apps == []
    assert resolution.local_apps == []


def test_local_app_settings_are_rebound_on_a_second_resolution(tmp_path, monkeypatch) -> None:
    """A local app object is owned by the caller's Orchestrator and outlives a
    resolution, so re-resolving it must re-bind rather than refuse."""
    from pydantic import BaseModel

    register(monkeypatch)

    class LocalSettings(BaseModel):
        city: str = "here"

    local = App("mine", description="d", settings=LocalSettings)
    orchestrator = Orchestrator(apps=[local])
    resolve_apps(
        orchestrator, settings_for(tmp_path, "[apps.mine]\ncity = 'lisbon'\n"), strict=True
    )
    resolve_apps(orchestrator, settings_for(tmp_path, "[apps.mine]\ncity = 'porto'\n"), strict=True)
    assert local.settings.city == "porto"


def test_diagnostic_mode_collects_several_failures(tmp_path, monkeypatch) -> None:
    register(monkeypatch, write_suite_app(tmp_path, monkeypatch, "demo", DEMO))
    settings = settings_for(tmp_path, "[apps.demo]\nnope = 1\n\n[apps.ghost]\nenabled = true\n")
    resolution = resolve_apps(Orchestrator(), settings, strict=False)
    assert {e.app for e in resolution.errors} == {"demo", "ghost"}


def test_a_healthy_app_survives_its_broken_neighbours(tmp_path, monkeypatch) -> None:
    """The point of diagnostic mode: one failure per app, and the apps that are
    fine still resolve completely."""
    good = write_suite_app(tmp_path, monkeypatch, "good", DEMO_TEMPLATE.format(name="good"))
    broken = write_suite_app(tmp_path, monkeypatch, "broken", "import dudamel_typo_module")
    misconfigured = write_suite_app(
        tmp_path, monkeypatch, "badcfg", DEMO_TEMPLATE.format(name="badcfg")
    )
    register(monkeypatch, good, broken, misconfigured)
    settings = settings_for(
        tmp_path,
        "[apps.good]\ncity = 'lisbon'\n\n[apps.broken]\n\n[apps.badcfg]\nnope = 1\n"
        "\n[apps.ghost]\n",
    )
    resolution = resolve_apps(Orchestrator(), settings, strict=False)
    assert [a.name for a in resolution.apps] == ["good"]
    assert resolution.apps[0].settings.city == "lisbon"
    assert resolution.suite_lanes == [("good", good.versions_dir)]
    assert {(e.app, e.stage) for e in resolution.errors} == {
        ("broken", 2),
        ("badcfg", 3),
        ("ghost", 1),
    }
