from datetime import date

import pytest

from dudamel.contract.schema import ToolSchema
from dudamel.exceptions import ToolValidationError


async def log_workout(exercise: str, sets: int, reps: int, weight_kg: float) -> str:
    """Record one exercise."""
    return exercise


async def search(query: str, since: date | None = None) -> str:
    """Search."""
    return query


def test_string_numbers_coerced():
    schema = ToolSchema(log_workout)
    out = schema.validate({"exercise": "bench", "sets": "3", "reps": "5", "weight_kg": "100"})
    assert out == {"exercise": "bench", "sets": 3, "reps": 5, "weight_kg": 100.0}
    assert isinstance(out["weight_kg"], float)


def test_iso_date_string_coerced():
    out = ToolSchema(search).validate({"query": "papers", "since": "2026-01-01"})
    assert out["since"] == date(2026, 1, 1)


def test_default_applied_when_omitted():
    out = ToolSchema(search).validate({"query": "papers"})
    assert out["since"] is None


def test_invalid_args_raise_model_readable_error():
    with pytest.raises(ToolValidationError, match="weight_kg"):
        ToolSchema(log_workout).validate(
            {"exercise": "bench", "sets": 3, "reps": 5, "weight_kg": "heavy"}
        )


def test_unknown_arg_rejected():
    with pytest.raises(ToolValidationError, match="extra"):
        ToolSchema(search).validate({"query": "x", "hallucinated": True})
