from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

from dudamel import App, Orchestrator
from dudamel.exceptions import DestructiveMigrationError
from dudamel.migrate import (
    ensure_app_migrations,
    generate_app_migration,
    sync_url,
    upgrade_apps,
    upgrade_core,
)


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
