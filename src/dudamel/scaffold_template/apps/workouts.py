from datetime import datetime

from dudamel import App

app = App("workouts", description="Log and review gym workouts")


class WorkoutSet(app.Model, table="sets"):
    exercise: str
    sets: int
    reps: int
    weight_kg: float
    logged_at: datetime = app.now()


@app.tool
async def log_workout(exercise: str, sets: int, reps: int, weight_kg: float) -> str:
    """Record one exercise from today's session."""
    async with app.db() as db:
        db.add(WorkoutSet(exercise=exercise, sets=sets, reps=reps, weight_kg=weight_kg))
    return f"Logged: {exercise} {sets}x{reps} @ {weight_kg}kg"


@app.widget(title="This week", renderer="stat")
async def week_volume() -> dict:
    async with app.db() as db:
        from sqlalchemy import func, select

        total = (await db.execute(select(func.sum(WorkoutSet.weight_kg)))).scalar() or 0
    return {"label": "Weekly volume", "value": total, "unit": "kg"}


@app.job(cron="0 20 * * *")
async def evening_summary() -> None:
    text = await app.llm("Summarize today's training", tier="fast")
    await app.notify(text)
