import pytest

from dudamel import App
from dudamel.exceptions import RegistryError


def test_cron_job_registration():
    app = App("workouts", description="d")

    @app.job(cron="0 20 * * *")
    async def evening() -> None:
        pass

    j = app.jobs["workouts.evening"]
    assert j.cron == "0 20 * * *" and j.interval_seconds is None and j.timeout == 300.0


def test_interval_job_registration():
    app = App("workouts", description="d")

    @app.job(interval_seconds=600)
    async def poll() -> None:
        pass

    assert app.jobs["workouts.poll"].interval_seconds == 600


def test_invalid_cron_fails_at_registration():
    app = App("workouts", description="d")
    with pytest.raises(RegistryError, match="cron"):

        @app.job(cron="99 99 * * *")
        async def bad() -> None:
            pass


def test_sync_job_rejected():
    app = App("workouts", description="d")
    with pytest.raises(RegistryError, match="must be async"):

        @app.job(interval_seconds=60)
        def sync_job() -> None:
            pass


def test_exactly_one_trigger_required():
    app = App("workouts", description="d")
    with pytest.raises(RegistryError, match="exactly one"):

        @app.job(cron="0 0 * * *", interval_seconds=60)
        async def both() -> None:
            pass

    with pytest.raises(RegistryError, match="exactly one"):

        @app.job()
        async def neither() -> None:
            pass
