from pathlib import Path

import pytest
from sqlalchemy import MetaData, create_engine, inspect

from dudamel import App, Orchestrator
from dudamel.exceptions import DestructiveMigrationError
from dudamel.migrate import (
    ensure_app_migrations,
    generate_app_migration,
    sync_url,
    upgrade_apps,
    upgrade_core,
)


class _FakeRegistry:
    """Duck-typed stand-in for `Registry` exposing only what
    `generate_app_migration` reads (`.apps` for prefixes, `.metadatas` for
    the target metadata union) -- lets tests construct an "orchestrator"
    carrying an app name that the real `Registry` would now reject outright
    (see test_registry.py), so migrate.py's own defense can be exercised in
    isolation."""

    def __init__(self, apps: dict, metadatas: dict) -> None:
        self.apps = apps
        self.metadatas = metadatas


class _FakeOrchestrator:
    def __init__(self, registry: _FakeRegistry) -> None:
        self.registry = registry


def make_orc(with_extra_column: bool = True) -> Orchestrator:
    app = App("blog", description="d")
    if with_extra_column:

        class Post(app.Model):
            title: str
            body: str

    else:

        class Post(app.Model):  # body removed -> destructive diff
            title: str

    return Orchestrator(apps=[app])


@pytest.fixture
def project(tmp_path: Path) -> tuple[str, Path]:
    url = f"sqlite+aiosqlite:///{tmp_path}/app.db"
    upgrade_core(url)
    ensure_app_migrations(tmp_path)
    return url, tmp_path


def test_generate_and_apply_creates_prefixed_table(project):
    url, pdir = project
    script = generate_app_migration(make_orc(), url, "add posts", pdir)
    assert script is not None and "blog_post" in script.read_text()
    upgrade_apps(url, pdir)
    insp = inspect(create_engine(sync_url(url)))
    assert "blog_post" in insp.get_table_names()
    assert "alembic_version_apps" in insp.get_table_names()


def test_noop_when_no_changes(project):
    url, pdir = project
    generate_app_migration(make_orc(), url, "add posts", pdir)
    upgrade_apps(url, pdir)
    assert generate_app_migration(make_orc(), url, "again", pdir) is None


def test_destructive_diff_gated(project):
    url, pdir = project
    generate_app_migration(make_orc(), url, "add posts", pdir)
    upgrade_apps(url, pdir)
    with pytest.raises(DestructiveMigrationError, match="body"):
        generate_app_migration(make_orc(with_extra_column=False), url, "drop body", pdir)
    # explicit override works
    script = generate_app_migration(
        make_orc(with_extra_column=False), url, "drop body", pdir, allow_destructive=True
    )
    assert script is not None


def test_unregistered_app_tables_never_dropped(project):
    url, pdir = project
    generate_app_migration(make_orc(), url, "add posts", pdir)
    upgrade_apps(url, pdir)
    # new orchestrator WITHOUT the blog app: its tables must be invisible, not dropped
    empty = Orchestrator(apps=[App("other", description="d")])
    assert generate_app_migration(empty, url, "nothing", pdir) is None


@pytest.mark.parametrize("fake_app_name", ["job", "alembic", "pending"])
def test_core_tables_excluded_even_if_an_apps_prefix_would_match_them(project, fake_app_name):
    """Registry now refuses App("job")/App("alembic")/App("pending") outright
    (their table prefix would shadow the core table "job_runs"/the alembic
    version-table namespace/"pending_confirmations" -- see test_registry.py),
    so this can no longer be mounted through the public API. Prove
    migrate.py's *own* defense holds independently of that by hand-building
    an orchestrator-shaped object carrying that (now-illegal) app name with
    zero tables of its own -- exactly the shape the old prefix-only
    `include_object` allowlist would have let a *reflected* physical core
    table slip through as an "extra" (droppable) table."""
    url, pdir = project
    fake_orc = _FakeOrchestrator(
        _FakeRegistry(apps={fake_app_name: None}, metadatas={fake_app_name: MetaData()})
    )
    # No diff at all -- not even a destructive one -- because the core
    # tables/alembic version tables are excluded before the prefix allowlist
    # (which would otherwise treat "job_runs" as belonging to app "job") ever
    # sees them. Without the fix, this raises DestructiveMigrationError (or,
    # with allow_destructive=True, silently generates a script that drops the
    # core table).
    assert generate_app_migration(fake_orc, url, "test", pdir) is None
    assert generate_app_migration(fake_orc, url, "test", pdir, allow_destructive=True) is None


def test_message_sanitized_prevents_path_traversal_and_docstring_injection(project):
    """A migration `message` is attacker/user-controlled free text embedded
    both in a filename (mig_dir / "versions" / f"..._{message}.py" -- pathlib
    treats an embedded "/" as a path separator) and inside a generated
    triple-quoted Python docstring (an embedded triple-quote breaks out of
    it). Both must be neutralized."""
    url, pdir = project
    evil_message = '../../etc/evil"""; import os  #'
    script = generate_app_migration(make_orc(), url, evil_message, pdir)
    assert script is not None
    # stays inside migrations/versions/ -- no "../" path-traversal
    assert script.parent.resolve() == (pdir / "migrations" / "versions").resolve()
    assert "/" not in script.stem
    assert '"' not in script.stem
    text = script.read_text()
    # a stray '"""' from the raw message would raise SyntaxError here
    compile(text, str(script), "exec")
    assert "etc_evil" in text
