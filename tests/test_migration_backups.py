"""Pre-migration backups: taken only when something is about to be applied,
never overwritten by a later one, one per change set, pruned to a bounded
count. The old fixed `<db>.bak` was replaced by every no-op run, so a
crash-restart after a bad partial migration destroyed the only good copy."""

import sqlite3
from pathlib import Path

import pytest
from test_suite_migrations import (
    CREATE_NOTES,
    CREATE_TASKS,
    db_url_for,
    make_lane,
    table_names,
)

from dudamel import migrate
from dudamel.exceptions import DudamelError
from dudamel.migrate import upgrade_all, upgrade_core, upgrade_database


def backups(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.glob("x.db.bak-*"))


def seed(tmp_path: Path) -> str:
    url = db_url_for(tmp_path)
    upgrade_core(url)
    conn = sqlite3.connect(tmp_path / "x.db")
    conn.execute("CREATE TABLE marker (v INTEGER)")
    conn.execute("INSERT INTO marker VALUES (42)")
    conn.commit()
    conn.close()
    return url


def test_noop_upgrade_takes_no_backup_and_leaves_existing_ones_alone(tmp_path) -> None:
    url = seed(tmp_path)
    lane = make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1")
    upgrade_all(url, tmp_path, [("notes", lane)])
    (first,) = backups(tmp_path)
    before = first.read_bytes()

    for _ in range(3):  # nothing pending: every flavour is a no-op
        upgrade_all(url, tmp_path, [("notes", lane)])
        upgrade_core(url)
        upgrade_database(url, tmp_path, [("notes", lane)])

    assert backups(tmp_path) == [first]
    assert first.read_bytes() == before


def test_real_migration_takes_exactly_one_backup_with_the_pre_migration_state(tmp_path) -> None:
    url = seed(tmp_path)
    lane = make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1")

    upgrade_all(url, tmp_path, [("notes", lane)])

    (backup,) = backups(tmp_path)
    conn = sqlite3.connect(backup)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        assert conn.execute("SELECT v FROM marker").fetchone() == (42,)
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        conn.close()
    assert "notes_note" not in tables  # the state BEFORE the lane was applied


def test_second_real_migration_keeps_the_first_backup(tmp_path) -> None:
    url = seed(tmp_path)
    notes = make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1")
    upgrade_all(url, tmp_path, [("notes", notes)])
    (first,) = backups(tmp_path)
    before = first.read_bytes()

    tasks = make_lane(tmp_path, "tasks", body=CREATE_TASKS, rev="t1")
    upgrade_all(url, tmp_path, [("notes", notes), ("tasks", tasks)])

    after = backups(tmp_path)
    assert len(after) == 2
    assert first in after and first.read_bytes() == before


def test_one_start_with_core_and_lanes_pending_makes_one_backup(tmp_path) -> None:
    url = db_url_for(tmp_path)
    # An existing but empty db file: core AND the lane are both pending.
    conn = sqlite3.connect(tmp_path / "x.db")
    conn.execute("CREATE TABLE marker (v INTEGER)")
    conn.commit()
    conn.close()
    lane = make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1")
    assert migrate.core_pending(url) and migrate.suite_lane_pending(url, "notes", lane)

    taken = upgrade_database(url, tmp_path, [("notes", lane)])

    assert taken is not None and backups(tmp_path) == [taken]


def test_upgrade_database_backs_up_once_for_lanes(tmp_path) -> None:
    url = seed(tmp_path)
    lane = make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1")

    taken = upgrade_database(url, tmp_path, [("notes", lane)])

    assert backups(tmp_path) == [taken]
    # backup=False: a caller already holding this run's snapshot skips another.
    tasks = make_lane(tmp_path, "tasks", body=CREATE_TASKS, rev="t1")
    assert (
        upgrade_database(url, tmp_path, [("notes", lane), ("tasks", tasks)], backup=False) is None
    )
    assert backups(tmp_path) == [taken]


def test_retention_prunes_oldest_first_and_never_touches_other_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(migrate, "BACKUPS_KEPT", 3)
    url = seed(tmp_path)
    # Neighbours that look similar but are not this database's backups.
    bystanders = [
        tmp_path / "x.db.bak",  # the legacy fixed name
        tmp_path / "y.db.bak-20200101T000000000000Z",  # another db's backup
        tmp_path / "x.db.bak-notes.txt",
        tmp_path / "x.db.bak-20200101T000000000000Z.keep",
    ]
    for b in bystanders:
        b.write_text("keep me")
    # Old backups of this db that must be pruned first.
    old = [tmp_path / f"x.db.bak-2020010{i}T000000000000Z" for i in (1, 2, 3)]
    for o in old:
        o.write_text("old")

    lanes: list[tuple[str, Path]] = []
    for i in range(3):
        lanes.append(
            (
                f"app{i}",
                make_lane(
                    tmp_path,
                    f"app{i}",
                    body=CREATE_NOTES.replace("notes_note", f"t{i}"),
                    rev=f"r{i}",
                ),
            )
        )
        upgrade_all(url, tmp_path, list(lanes))

    kept = backups(tmp_path)
    kept = [p for p in kept if migrate._backup_name_re("x.db").fullmatch(p.name)]
    assert len(kept) == 3
    assert not any(o.exists() for o in old)  # oldest went first
    assert all(b.exists() and b.read_text() == "keep me" for b in bystanders)


def test_a_migration_that_fails_on_every_restart_keeps_the_clean_backup(tmp_path) -> None:
    """The shipped unit restarts every 5s forever, so a migration that always
    fails re-runs the upgrade far more than BACKUPS_KEPT times. Each run is
    still pending; if each took a backup, the prune would delete the only
    pre-migration copy within a few restarts. Nothing has changed between
    runs, so none after the first may take one."""
    url = seed(tmp_path)
    lane = make_lane(tmp_path, "notes", body="op.execute('NOT VALID SQL')", rev="n1")

    for _ in range(migrate.BACKUPS_KEPT + 3):
        with pytest.raises(DudamelError):
            upgrade_all(url, tmp_path, [("notes", lane)])

    (clean,) = backups(tmp_path)
    conn = sqlite3.connect(clean)
    try:
        assert conn.execute("SELECT v FROM marker").fetchone() == (42,)
    finally:
        conn.close()


def test_a_partial_failure_adds_one_half_migrated_backup_beside_the_clean_one(tmp_path) -> None:
    """SQLite applies a lane's first revision even when its second fails. The
    database has then changed, so one more backup is right -- but beside the
    clean one, not over it, and only once however many restarts follow."""
    url = seed(tmp_path)
    lane = make_lane(tmp_path, "notes", body=CREATE_NOTES, rev="n1")
    make_lane(tmp_path, "notes", body="op.execute('NOT VALID SQL')", rev="n2", down="n1")

    for _ in range(migrate.BACKUPS_KEPT + 3):
        with pytest.raises(DudamelError):
            upgrade_all(url, tmp_path, [("notes", lane)])

    clean, half = backups(tmp_path)
    assert "notes_note" not in table_names(f"sqlite+aiosqlite:///{clean}")
    assert "notes_note" in table_names(f"sqlite+aiosqlite:///{half}")
