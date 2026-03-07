from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dateutil import parser as date_parser


WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class WorkoutLogEntry:
    entry_date: date
    workout_type: str
    duration_min: int
    sets: Optional[int] = None
    notes: str = ""
    calories_kcal: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entry_date"] = self.entry_date.isoformat()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorkoutLogEntry":
        return WorkoutLogEntry(
            entry_date=date.fromisoformat(d["entry_date"]),
            workout_type=str(d.get("workout_type") or "workout"),
            duration_min=int(d.get("duration_min") or 0),
            sets=d.get("sets"),
            notes=str(d.get("notes") or ""),
            calories_kcal=d.get("calories_kcal"),
        )


def normalize_workout_type(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "workout"

    mapping = {
        "cardio": ["cardio", "run", "running", "jog", "jogging", "bike", "cycling", "swim", "rowing", "elliptical"],
        "strength": ["strength", "lift", "lifting", "weights", "weight", "resistance", "upper", "lower", "push", "pull"],
        "flexibility": ["stretch", "stretching", "mobility", "flexibility"],
        "yoga": ["yoga", "vinyasa", "hatha"],
        "hiit": ["hiit", "interval"],
    }
    for key, words in mapping.items():
        if any(w in t for w in words):
            return key

    # Common body-part splits
    if any(w in t for w in ["chest", "back", "legs", "glutes", "shoulders", "arms", "biceps", "triceps", "core", "abs"]):
        return "strength"

    return t.split()[0]


_DURATION_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>h|hr|hrs|hour|hours|min|mins|minute|minutes)\b",
    flags=re.IGNORECASE,
)


def parse_duration_minutes(text: str) -> Optional[int]:
    if not text:
        return None

    m = _DURATION_RE.search(text)
    if not m:
        return None

    value = float(m.group("value"))
    unit = m.group("unit").lower()

    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return int(round(value * 60))
    return int(round(value))


_SETS_RE = re.compile(r"\+?\s*(?P<sets>\d+)\s*sets?\b", flags=re.IGNORECASE)


def parse_sets(text: str) -> Optional[int]:
    if not text:
        return None
    m = _SETS_RE.search(text)
    if not m:
        return None
    return int(m.group("sets"))


def estimate_calories_kcal(
    *,
    duration_min: int,
    workout_type: str,
    weight_kg: Optional[float],
) -> Optional[int]:
    if not weight_kg or duration_min <= 0:
        return None

    met_table = {
        "cardio": 7.0,
        "strength": 6.0,
        "hiit": 9.0,
        "yoga": 3.0,
        "flexibility": 2.5,
        "workout": 5.0,
    }
    met = met_table.get(workout_type, met_table["workout"])
    kcal = met * 3.5 * float(weight_kg) / 200.0 * float(duration_min)
    return int(round(kcal))


def parse_quick_log_message(
    text: str,
    *,
    today: date,
    default_duration_min: int,
    weight_kg: Optional[float],
) -> Optional[WorkoutLogEntry]:
    """Parse messages like "+Chest +3 sets today" or "+30 min cardio".

    Returns None if the message doesn't look like a log.
    """
    raw = (text or "").strip()
    if not raw.startswith("+"):
        return None

    # Strip leading '+' markers but keep internal '+' for sets patterns
    content = raw.lstrip("+").strip()
    if not content:
        return None

    duration = parse_duration_minutes(content)
    sets = parse_sets(content)

    workout_type = normalize_workout_type(content)
    duration_min = duration if duration is not None else int(default_duration_min)

    calories_kcal = estimate_calories_kcal(
        duration_min=duration_min,
        workout_type=workout_type,
        weight_kg=weight_kg,
    )

    return WorkoutLogEntry(
        entry_date=today,
        workout_type=workout_type,
        duration_min=duration_min,
        sets=sets,
        notes=content,
        calories_kcal=calories_kcal,
    )


_PR_RE = re.compile(
    r"^\+\s*pr\s+(?P<lift>[a-zA-Z][a-zA-Z0-9 _\-/]{1,40}?)\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>kg|kgs|lb|lbs)?\s*$",
    flags=re.IGNORECASE,
)


def parse_pr_message(text: str, *, today: date) -> Optional[Dict[str, Any]]:
    """Parse messages like "+PR bench 100kg" into a PR record dict."""
    raw = (text or "").strip()
    m = _PR_RE.match(raw)
    if not m:
        return None

    lift = " ".join(m.group("lift").strip().split())
    value = float(m.group("value"))
    unit = (m.group("unit") or "").lower()
    if unit in {"kgs", "kg"}:
        unit = "kg"
    elif unit in {"lb", "lbs"}:
        unit = "lb"
    else:
        unit = "kg"

    return {
        "date": today.isoformat(),
        "lift": lift,
        "value": value,
        "unit": unit,
    }


def coerce_logs(log_dicts: Sequence[Dict[str, Any]]) -> List[WorkoutLogEntry]:
    logs: List[WorkoutLogEntry] = []
    for d in log_dicts:
        try:
            logs.append(WorkoutLogEntry.from_dict(d))
        except Exception:
            continue
    return sorted(logs, key=lambda e: e.entry_date)


def totals(logs: Sequence[WorkoutLogEntry]) -> Dict[str, Any]:
    total_workouts = len(logs)
    total_minutes = sum(max(0, int(e.duration_min)) for e in logs)
    total_hours = total_minutes / 60.0
    total_calories = sum(int(e.calories_kcal or 0) for e in logs) or None
    return {
        "total_workouts": total_workouts,
        "total_minutes": total_minutes,
        "total_hours": total_hours,
        "total_calories": total_calories,
    }


def start_of_week(d: date) -> date:
    # Monday as week start
    return d - timedelta(days=d.weekday())


def weekly_summary(logs: Sequence[WorkoutLogEntry], *, today: date) -> Dict[str, Any]:
    week_start = start_of_week(today)
    week_end = week_start + timedelta(days=7)
    week_logs = [e for e in logs if week_start <= e.entry_date < week_end]
    t = totals(week_logs)
    return {
        "week_start": week_start,
        "week_end": week_end,
        **t,
    }


def _iter_planned_workout_dates(
    *,
    start: date,
    end_inclusive: date,
    workout_weekdays: Sequence[int],
) -> Iterable[date]:
    workout_set = set(int(x) for x in workout_weekdays)
    d = start
    while d <= end_inclusive:
        if d.weekday() in workout_set:
            yield d
        d += timedelta(days=1)


def projected_total_workouts_by_date(
    *,
    logs: Sequence[WorkoutLogEntry],
    start_date: date,
    workout_weekdays: Sequence[int],
    target_date: date,
) -> int:
    completed = len([e for e in logs if e.entry_date <= target_date])

    # Project future planned workouts between tomorrow and target_date
    tomorrow = date.today() + timedelta(days=1)
    projection_start = max(tomorrow, start_date)

    if target_date < projection_start:
        return completed

    planned_dates = list(
        _iter_planned_workout_dates(
            start=projection_start,
            end_inclusive=target_date,
            workout_weekdays=workout_weekdays,
        )
    )

    # Avoid double-counting dates already logged
    logged_dates = {e.entry_date for e in logs}
    projected_additional = sum(1 for d in planned_dates if d not in logged_dates)
    return completed + projected_additional


def estimate_completion_date_for_target_workouts(
    *,
    logs: Sequence[WorkoutLogEntry],
    start_date: date,
    workout_weekdays: Sequence[int],
    target_total_workouts: int,
    today: date,
) -> Optional[date]:
    if target_total_workouts <= 0:
        return None

    completed = len([e for e in logs if e.entry_date <= today])
    if completed >= target_total_workouts:
        return today

    # Simulate forward on planned workout days
    logged_dates = {e.entry_date for e in logs}
    remaining = target_total_workouts - completed

    d = today + timedelta(days=1)
    workout_set = set(int(x) for x in workout_weekdays)
    safety_end = today + timedelta(days=365 * 3)

    while d <= safety_end:
        if d.weekday() in workout_set and d not in logged_dates:
            remaining -= 1
            if remaining <= 0:
                return d
        d += timedelta(days=1)

    return None


def try_extract_date_from_question(text: str, *, today: date) -> Optional[date]:
    """Best-effort extraction of a date for projection questions.

    Handles strings like "by March 31" / "by 2026-03-31".
    """
    if not text:
        return None

    lower = text.lower()
    if " by " not in lower and not lower.startswith("by ") and " until " not in lower:
        return None

    # Prefer substring after 'by' or 'until'
    anchor = None
    if " by " in lower:
        anchor = lower.split(" by ", 1)[1]
    elif lower.startswith("by "):
        anchor = lower[3:]
    elif " until " in lower:
        anchor = lower.split(" until ", 1)[1]

    if not anchor:
        return None

    anchor = anchor.strip().strip("?.! ")
    if not anchor:
        return None

    try:
        dt = date_parser.parse(anchor, default=datetime(today.year, today.month, today.day))
        return dt.date()
    except Exception:
        return None
