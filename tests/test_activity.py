import json
from datetime import date, datetime
from enum import Enum

import pytest
from sqlalchemy import select

from dudamel.activity import json_safe, log_activity
from dudamel.db import Database
from dudamel.migrate import upgrade_core
from dudamel.models_core import Activity


class Color(Enum):
    RED = "red"


class Holiday(Enum):
    XMAS = date(2026, 12, 25)


def test_json_safe_handles_the_validate_output_types() -> None:
    out = json_safe(
        {
            "color": Color.RED,
            "when": date(2026, 7, 24),
            "at": datetime(2026, 7, 24, 12, 0),
            "tags": {
                "a",
            },
            "nested": [{"c": Color.RED}],
            "weird": object(),
        }
    )
    json.dumps(out)  # must not raise
    assert out["color"] == "red"
    assert out["when"] == "2026-07-24"
    assert out["nested"][0]["c"] == "red"
    assert isinstance(out["tags"], list)


@pytest.fixture
async def db(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    upgrade_core(url)
    d = Database(url)
    yield d
    await d.dispose()


async def test_log_activity_writes_row_with_enum_args(db: Database) -> None:
    await log_activity(
        db,
        tool="paint",
        args={"color": Color.RED},
        status="ok",
        result_preview="x" * 900,
    )
    async with db.session() as s:
        row = (await s.execute(select(Activity))).scalar_one()
    assert row.args == {"color": "red"}
    assert row.status == "ok" and len(row.result_preview) == 500


def test_enum_with_nonprimitive_value_recurses() -> None:
    out = json_safe({"h": Holiday.XMAS})
    json.dumps(out)  # must not raise
    assert out["h"] == "2026-12-25"


async def test_an_unattributed_row_claims_no_actor_and_no_surface(db: Database) -> None:
    """Omitting the attribution must record "unknown", never a plausible
    default -- a guessed surface would be indistinguishable from a real one."""
    await log_activity(db, tool="paint", args={}, status="ok")
    async with db.session() as s:
        row = (await s.execute(select(Activity))).scalar_one()
    assert (row.actor, row.source) == (None, None)


async def test_actor_and_source_round_trip(db: Database) -> None:
    await log_activity(db, tool="paint", args={}, status="ok", actor="web", source="web")
    async with db.session() as s:
        row = (await s.execute(select(Activity))).scalar_one()
    assert (row.actor, row.source) == ("web", "web")
