from __future__ import annotations

from datetime import date
import html
import hashlib
import os
import re
from typing import Any, Dict, List, Optional

import streamlit as st

from src.fitness_tracker import (
    WEEKDAY_LABELS,
    coerce_logs,
    estimate_completion_date_for_target_workouts,
    parse_quick_log_message,
    parse_pr_message,
    projected_total_workouts_by_date,
    totals,
    try_extract_date_from_question,
    weekly_summary,
)


APP_TITLE = "SmartFit"


SAFETY_NOTE = (
    "This chatbot provides general fitness and nutrition guidance and progress tracking. "
    "It is not medical advice and does not replace a doctor or certified personal trainer. "
    "If you have injuries, pain, or health concerns, consult a qualified professional."
)


SYSTEM_PROMPT = """You are a friendly, motivating, and knowledgeable gym instructor chatbot. You provide safe, effective exercise guidance, nutrition tips, and progress tracking for users of all fitness levels.

ON START:
Ask the user for:
- Fitness goal (strength, weight loss, endurance, flexibility, general health)
- Current fitness level (beginner, intermediate, advanced)
- Workout frequency (days per week)
- Any injuries or limitations

TRACKING RULES:
- Remember user preferences (goals, fitness level, workout frequency) for the current session.
- Log completed workouts, exercises, and reps for progress tracking.
- Suggest modifications if exercises are too difficult or unsafe.

OUTPUT FORMAT:
Always respond with:
- A motivational greeting
- Clear, step-by-step exercise instructions (or next actions)
- Optional tips for form, intensity, or variations
- Progress summary if relevant

WEEKLY SCHEDULE TABLE:
- Include a concise weekly schedule as a Markdown table in your responses.
- The table should cover Mon–Sun and include at least these columns: Day | Workout | Meals | Time.
- Use the user's stated goal, level, frequency, and limitations to keep it realistic and safe.
- If the user didn't specify times, choose sensible default times and label them as suggestions.
- On non-workout days, schedule rest/recovery and light activity (walk/mobility) instead of intense training.

SAFETY RULES:
- Never provide medical advice.
- Recommend exercises only suitable for the user’s stated fitness level.
- Advise proper warm-up and cool-down routines.
- Suggest consulting a doctor if the user has injuries or health concerns.
- Never guarantee results or make exaggerated claims.

TONE:
Be friendly, concise, and motivating. Use clear formatting.
"""


def _normalize_day_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z]", "", s)
    # Support both short and long day names.
    mapping = {
        "mon": "mon",
        "monday": "mon",
        "tue": "tue",
        "tues": "tue",
        "tuesday": "tue",
        "wed": "wed",
        "weds": "wed",
        "wednesday": "wed",
        "thu": "thu",
        "thurs": "thu",
        "thursday": "thu",
        "fri": "fri",
        "friday": "fri",
        "sat": "sat",
        "saturday": "sat",
        "sun": "sun",
        "sunday": "sun",
    }
    return mapping.get(s, s)


def _today_short_label(today: date) -> str:
    return ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][today.weekday()]


def _try_parse_week_schedule_table(text: str) -> Optional[List[Dict[str, str]]]:
    """Extract rows from a Markdown table containing Day|Workout|Meals|Time."""
    if not text or "|" not in text:
        return None

    lines = [ln.rstrip() for ln in text.splitlines()]
    for i in range(len(lines) - 2):
        header = lines[i]
        sep = lines[i + 1]
        if "|" not in header or "|" not in sep:
            continue

        header_cells = [c.strip().lower() for c in header.strip().strip("|").split("|")]
        if not header_cells:
            continue

        if "day" not in header_cells or "workout" not in header_cells:
            continue

        # Basic separator line check: | --- | --- |
        if not re.search(r"\|\s*:?[-]{3,}:?\s*\|", sep):
            continue

        col_index = {name: idx for idx, name in enumerate(header_cells)}
        rows: List[Dict[str, str]] = []
        for j in range(i + 2, len(lines)):
            ln = lines[j]
            if "|" not in ln:
                break
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            # Skip separator-ish rows or empties.
            if not any(cells):
                continue

            def get(name: str) -> str:
                idx = col_index.get(name)
                return cells[idx] if idx is not None and idx < len(cells) else ""

            rows.append(
                {
                    "Day": get("day"),
                    "Workout": get("workout"),
                    "Meals": get("meals"),
                    "Time": get("time"),
                }
            )
        return rows or None

    return None


def _infer_workout_type(workout_text: str) -> str:
    w = (workout_text or "").lower()
    if any(k in w for k in ["rest", "recovery", "off day", "day off"]):
        return "workout"
    if any(k in w for k in ["hiit", "interval", "tabata"]):
        return "hiit"
    if any(k in w for k in ["yoga"]):
        return "yoga"
    if any(k in w for k in ["mobility", "stretch", "flexibility"]):
        return "flexibility"
    if any(k in w for k in ["run", "cardio", "bike", "cycle", "swim", "row", "walk", "jog"]):
        return "cardio"
    if any(
        k in w
        for k in [
            "strength",
            "lift",
            "weights",
            "bench",
            "squat",
            "deadlift",
            "press",
            "upper",
            "lower",
            "push",
            "pull",
            "full body",
        ]
    ):
        return "strength"
    return "workout"


def _extract_duration_minutes(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(\d{1,3})\s*(?:min|mins|minute|minutes)\b", text, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{1,3})\s*m\b", text, flags=re.I)
    if m:
        return int(m.group(1))
    return None


def _extract_sets_count(text: str) -> Optional[int]:
    if not text:
        return None

    # Prefer total sets from common 3x10 patterns.
    reps_patterns = re.findall(r"\b(\d{1,2})\s*x\s*\d{1,3}\b", text, flags=re.I)
    if reps_patterns:
        total = sum(int(n) for n in reps_patterns)
        return max(0, min(total, 200))

    m = re.search(r"\b(\d{1,3})\s*sets\b", text, flags=re.I)
    if m:
        return int(m.group(1))
    return None


def _extract_latest_suggested_workout(messages: List[Dict[str, Any]], today: date) -> Optional[Dict[str, Any]]:
    """Return the latest suggested workout for today from the most recent assistant message."""
    if not messages:
        return None

    today_key = _today_short_label(today)
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")

        rows = _try_parse_week_schedule_table(content)
        if rows:
            for row in rows:
                day_label = _normalize_day_label(row.get("Day", ""))
                if day_label == today_key:
                    workout = (row.get("Workout") or "").strip()
                    if not workout:
                        continue
                    # Skip auto-fill on rest days.
                    if any(k in workout.lower() for k in ["rest", "recovery", "off"]):
                        return None

                    meals = (row.get("Meals") or "").strip()
                    time = (row.get("Time") or "").strip()
                    notes_bits = [workout]
                    if time:
                        notes_bits.append(f"Suggested time: {time}")
                    if meals:
                        notes_bits.append(f"Meals: {meals}")

                    sig = hashlib.sha256((content + today.isoformat()).encode("utf-8")).hexdigest()[:12]
                    return {
                        "type": _infer_workout_type(workout),
                        "duration_min": _extract_duration_minutes(workout),
                        "sets": _extract_sets_count(workout),
                        "notes": "\n".join(notes_bits).strip(),
                        "date": today,
                        "sig": sig,
                    }

        # Fallback: look for a “Today” workout line in free text.
        m = re.search(r"(?im)^\s*(?:today(?:'s)?\s*)?workout\s*:\s*(.+)$", content)
        if m:
            workout = m.group(1).strip()
            if workout and not any(k in workout.lower() for k in ["rest", "recovery", "off"]):
                sig = hashlib.sha256((content + today.isoformat()).encode("utf-8")).hexdigest()[:12]
                return {
                    "type": _infer_workout_type(workout),
                    "duration_min": _extract_duration_minutes(workout),
                    "sets": _extract_sets_count(workout),
                    "notes": workout,
                    "date": today,
                    "sig": sig,
                }

    return None


def _apply_structured_log_autofill(*, profile: Dict[str, Any]) -> None:
    """If the assistant suggested a workout, prefill the structured log widgets."""
    suggestion = _extract_latest_suggested_workout(st.session_state.messages, date.today())
    if not suggestion:
        return

    # Only overwrite when the suggestion changes (keeps user edits).
    prev_sig = st.session_state.get("structured_autofill_sig")
    if prev_sig == suggestion.get("sig"):
        return
    st.session_state["structured_autofill_sig"] = suggestion.get("sig")

    # Prefill widget states used by the sidebar form.
    st.session_state["structured_w_type"] = suggestion.get("type") or "workout"
    st.session_state["structured_duration"] = int(
        suggestion.get("duration_min")
        or int(profile.get("session_minutes") or 60)
    )
    st.session_state["structured_sets"] = int(suggestion.get("sets") or 0)
    st.session_state["structured_notes"] = str(suggestion.get("notes") or "")
    st.session_state["structured_date"] = suggestion.get("date") or date.today()


def _init_state() -> None:
    st.session_state.setdefault("profile", None)
    st.session_state.setdefault("workout_logs", [])  # list[dict]
    st.session_state.setdefault("prs", [])  # list[dict]
    st.session_state.setdefault("messages", [])  # list[dict] with {role, content}
    st.session_state.setdefault("last_weekly_summary_week", None)
    st.session_state.setdefault("page", "setup")
    st.session_state.setdefault("pending_user_text", None)


def _get_gemini_api_key() -> Optional[str]:
    # Prefer Streamlit secrets, then environment variable.
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[assignment]
    except Exception:
        key = None
    key = (key or os.getenv("GEMINI_API_KEY") or "").strip()
    return key or None


def _gemini_setup_hint() -> str:
    has_secrets = os.path.exists(os.path.join(".streamlit", "secrets.toml"))
    has_example = os.path.exists(os.path.join(".streamlit", "secrets.toml.example"))

    if not has_secrets and has_example:
        return (
            "Gemini API key not found. It looks like you only edited "
            "`.streamlit/secrets.toml.example`.\n\n"
            "Fix:\n"
            "1) Copy/rename `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml`\n"
            "2) Put your key like: `GEMINI_API_KEY = \"...\"`\n"
            "3) Restart: `streamlit run app.py`"
        )

    return (
        "Gemini API key not found.\n\n"
        "Set it internally using one of these options:\n"
        "- Create `.streamlit/secrets.toml` with `GEMINI_API_KEY = \"...\"` (recommended)\n"
        "- Or set env var `GEMINI_API_KEY`\n\n"
        "Then restart: `streamlit run app.py`"
    )


def _default_workout_days(days_per_week: int) -> List[int]:
    days_per_week = max(1, min(int(days_per_week), 7))
    patterns = {
        1: [2],
        2: [1, 4],
        3: [0, 2, 4],
        4: [0, 2, 4, 6],
        5: [0, 1, 3, 4, 6],
        6: [0, 1, 2, 3, 4, 6],
        7: [0, 1, 2, 3, 4, 5, 6],
    }
    return patterns.get(days_per_week, [0, 2, 4])


@st.dialog("Quick Setup")
def _dialog_quick_setup() -> None:
    """Modal dialog for fitness profile setup."""
    existing: Optional[Dict[str, Any]] = st.session_state.profile
    is_edit = existing is not None

    st.caption("Fill this once to personalize the chatbot." if not is_edit else "Edit your fitness profile.")
    with st.form("quick_setup_dialog"):
            goal_options = ["Strength", "Weight loss", "Endurance", "Flexibility", "General health"]
            level_options = ["Beginner", "Intermediate", "Advanced"]

            goal = st.selectbox(
                "Fitness goal",
                goal_options,
                index=goal_options.index((existing or {}).get("goal", "Strength")),
            )
            level = st.selectbox(
                "Current fitness level",
                level_options,
                index=level_options.index((existing or {}).get("level", "Beginner")),
            )
            days_per_week = st.number_input(
                "Workout frequency (days/week)",
                min_value=1,
                max_value=7,
                value=int((existing or {}).get("days_per_week") or 4),
            )
            session_minutes = st.number_input(
                "Typical session length (minutes)",
                min_value=10,
                max_value=180,
                value=int((existing or {}).get("session_minutes") or 60),
                step=5,
            )
            limitations = st.text_area(
                "Any injuries or limitations? (optional)",
                value=str((existing or {}).get("limitations") or ""),
            )

            default_days = (existing or {}).get("workout_weekdays") or _default_workout_days(int(days_per_week))
            planned_days = st.multiselect(
                "Planned workout days",
                options=list(range(7)),
                format_func=lambda i: WEEKDAY_LABELS[i],
                default=list(default_days),
                help="Used for rest-day-aware projections.",
            )

            start_date = st.date_input("Start date", value=(existing or {}).get("start_date") or date.today())
            target_weeks = st.number_input(
                "Program length (weeks)",
                min_value=1,
                max_value=104,
                value=int((existing or {}).get("target_weeks") or 12),
            )
            weight_kg = st.number_input(
                "Body weight (kg) (optional, for calorie estimates)",
                min_value=0.0,
                max_value=400.0,
                value=float((existing or {}).get("weight_kg") or 0.0),
                step=0.5,
            )

            submitted = st.form_submit_button("Save" if not is_edit else "Save changes")

    if submitted:
        workout_weekdays = sorted(planned_days) if planned_days else _default_workout_days(int(days_per_week))
        st.session_state.profile = {
            "goal": goal,
            "level": level,
            "days_per_week": int(days_per_week),
            "session_minutes": int(session_minutes),
            "limitations": str(limitations or "").strip(),
            "workout_weekdays": workout_weekdays,
            "start_date": start_date,
            "target_weeks": int(target_weeks),
            "target_total_workouts": int(target_weeks) * max(1, len(workout_weekdays)),
            "weight_kg": float(weight_kg) if float(weight_kg) > 0 else None,
        }
        st.session_state.page = "chat"
        st.rerun()


def _render_quick_setup_sidebar() -> None:
    if st.button("Edit Setup", use_container_width=True, key="open_setup_dialog"):
        _dialog_quick_setup()


def _render_quick_setup_main() -> None:
    """First-screen setup shown in the main page (not sidebar)."""
    existing: Optional[Dict[str, Any]] = st.session_state.profile

    st.subheader("Quick setup")
    st.caption("Fill this once to personalize the chatbot. You can edit later in the sidebar.")

    with st.form("quick_setup_main"):
        goal_options = ["Strength", "Weight loss", "Endurance", "Flexibility", "General health"]
        level_options = ["Beginner", "Intermediate", "Advanced"]

        goal = st.selectbox(
            "Fitness goal",
            goal_options,
            index=goal_options.index((existing or {}).get("goal", "Strength")),
        )
        level = st.selectbox(
            "Current fitness level",
            level_options,
            index=level_options.index((existing or {}).get("level", "Beginner")),
        )
        days_per_week = st.number_input(
            "Workout frequency (days/week)",
            min_value=1,
            max_value=7,
            value=int((existing or {}).get("days_per_week") or 4),
        )
        session_minutes = st.number_input(
            "Typical session length (minutes)",
            min_value=10,
            max_value=180,
            value=int((existing or {}).get("session_minutes") or 60),
            step=5,
        )
        limitations = st.text_area(
            "Any injuries or limitations? (optional)",
            value=str((existing or {}).get("limitations") or ""),
        )

        default_days = (existing or {}).get("workout_weekdays") or _default_workout_days(int(days_per_week))
        planned_days = st.multiselect(
            "Planned workout days",
            options=list(range(7)),
            format_func=lambda i: WEEKDAY_LABELS[i],
            default=list(default_days),
            help="Used for rest-day-aware projections.",
        )

        start_date = st.date_input("Start date", value=(existing or {}).get("start_date") or date.today())
        target_weeks = st.number_input(
            "Program length (weeks)",
            min_value=1,
            max_value=104,
            value=int((existing or {}).get("target_weeks") or 12),
        )
        weight_kg = st.number_input(
            "Body weight (kg) (optional, for calorie estimates)",
            min_value=0.0,
            max_value=400.0,
            value=float((existing or {}).get("weight_kg") or 0.0),
            step=0.5,
        )

        submitted = st.form_submit_button("Save and continue")

    if submitted:
        workout_weekdays = sorted(planned_days) if planned_days else _default_workout_days(int(days_per_week))
        st.session_state.profile = {
            "goal": goal,
            "level": level,
            "days_per_week": int(days_per_week),
            "session_minutes": int(session_minutes),
            "limitations": str(limitations or "").strip(),
            "workout_weekdays": workout_weekdays,
            "start_date": start_date,
            "target_weeks": int(target_weeks),
            "target_total_workouts": int(target_weeks) * max(1, len(workout_weekdays)),
            "weight_kg": float(weight_kg) if float(weight_kg) > 0 else None,
        }
        st.session_state.page = "chat"
        st.rerun()


def _compute_targets(profile: Dict[str, Any]) -> Dict[str, Any]:
    start_date = profile["start_date"]
    workout_weekdays = profile["workout_weekdays"]
    target_weeks = int(profile.get("target_weeks") or 12)
    target_total_workouts = profile.get("target_total_workouts")

    if not target_total_workouts:
        target_total_workouts = int(target_weeks * max(1, len(workout_weekdays)))

    return {
        "start_date": start_date,
        "workout_weekdays": workout_weekdays,
        "target_weeks": target_weeks,
        "target_total_workouts": int(target_total_workouts),
    }


def _progress_block(profile: Dict[str, Any]) -> None:
    logs = coerce_logs(st.session_state.workout_logs)
    today = date.today()
    t = totals(logs)
    targets = _compute_targets(profile)

    percent = 0.0
    if targets["target_total_workouts"] > 0:
        percent = min(1.0, t["total_workouts"] / targets["target_total_workouts"])

    est_done = estimate_completion_date_for_target_workouts(
        logs=logs,
        start_date=targets["start_date"],
        workout_weekdays=targets["workout_weekdays"],
        target_total_workouts=targets["target_total_workouts"],
        today=today,
    )

    est_done_compact = est_done.strftime("%b %d") if est_done else "—"
    est_done_full = est_done.isoformat() if est_done else "No estimate yet"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Workouts", f"{t['total_workouts']}")
    c2.metric("Hours", f"{t['total_hours']:.1f}")
    c3.metric("Goal", f"{percent * 100:.0f}%")
    c4.metric("Est. completion", est_done_compact, help=f"Full date: {est_done_full}")


def _show_weekly_summary_if_sunday(profile: Dict[str, Any]) -> None:
    today = date.today()
    # Python weekday: Monday=0 ... Sunday=6
    if today.weekday() != 6:
        return

    current_week = int(today.strftime("%G%V"))  # ISO year+week
    if st.session_state.last_weekly_summary_week == current_week:
        return

    logs = coerce_logs(st.session_state.workout_logs)
    s = weekly_summary(logs, today=today)

    targets = _compute_targets(profile)
    total = totals(logs)
    pct = 0.0
    if targets["target_total_workouts"] > 0:
        pct = min(1.0, total["total_workouts"] / targets["target_total_workouts"])

    st.info(
        "Weekly summary (this week): "
        f"{s['total_workouts']} workouts, {s['total_hours']:.1f} hours"
        + (f", ~{s['total_calories']} kcal" if s.get("total_calories") else "")
        + f". Overall goal: {pct * 100:.0f}% complete."
    )

    st.session_state.last_weekly_summary_week = current_week


def _gemini_generate(api_key: str, prompt: str) -> str:
    """Generate text with Gemini.

    Uses `google-genai` and tries multiple model IDs.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Missing Gemini API key")

    from google import genai  # type: ignore

    client = genai.Client(api_key=api_key)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    model_candidates = [
        "models/gemini-flash-latest",
        "models/gemini-flash-lite-latest",
        "models/gemini-pro-latest",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.0-flash",
    ]

    errors: List[str] = []
    for model_name in model_candidates:
        try:
            resp = client.models.generate_content(model=model_name, contents=full_prompt)
            text = getattr(resp, "text", None) or ""
            if text.strip():
                return text.strip()
            errors.append(f"{model_name}: empty response")
        except Exception as e:
            errors.append(f"{model_name}: {e}")
            continue

    raise RuntimeError("Gemini SDK error: no model succeeded. " + " | ".join(errors[:3]) + (" | ..." if len(errors) > 3 else ""))


def _assistant_reply(content: str) -> None:
    st.session_state.messages.append({"role": "assistant", "content": content})


def _user_msg(content: str) -> None:
    st.session_state.messages.append({"role": "user", "content": content})


def _render_chat() -> None:
    for msg in st.session_state.messages:
        role = str(msg.get("role") or "assistant")
        content = str(msg.get("content") or "")

        if role == "user":
            content_html = html.escape(content).replace("\n", "<br>")
            st.markdown(
                (
                    "<div style='width:100%; border:1px solid rgba(255,255,255,0.10); "
                        "border-radius:12px; padding:10px 12px; background:rgba(255,255,255,0.01); "
                    "box-sizing:border-box; text-align:right;'>"
                    f"{content_html}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(content)

        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)


def _build_chat_context(
    *,
    profile: Dict[str, Any],
    logs: List[Any],
    user_text: str,
    action_notes: List[str],
    projection_note: Optional[str],
) -> str:
    t = totals(logs)
    targets = _compute_targets(profile)
    est_done = estimate_completion_date_for_target_workouts(
        logs=logs,
        start_date=targets["start_date"],
        workout_weekdays=targets["workout_weekdays"],
        target_total_workouts=targets["target_total_workouts"],
        today=date.today(),
    )

    action_block = "\n".join(f"- {n}" for n in action_notes) if action_notes else "- None"
    projection_block = f"- {projection_note}" if projection_note else "- None"

    return (
        "USER PROFILE\n"
        f"- Fitness goal: {profile.get('goal')}\n"
        f"- Fitness level: {profile.get('level')}\n"
        f"- Workout frequency (days/week): {profile.get('days_per_week')}\n"
        f"- Planned workout days: {[WEEKDAY_LABELS[d] for d in profile.get('workout_weekdays', [])]}\n"
        f"- Typical session minutes: {profile.get('session_minutes')}\n"
        + (f"- Injuries/limitations: {profile.get('limitations')}\n" if profile.get("limitations") else "")
        + "\n"
        "PROGRESS\n"
        f"- Workouts logged: {t['total_workouts']}\n"
        f"- Hours logged: {t['total_hours']:.1f}\n"
        + (f"- Estimated completion date: {est_done.isoformat()}\n" if est_done else "")
        + "\n"
        "TRACKING UPDATE (what happened this message)\n"
        f"{action_block}\n\n"
        "PROJECTION (if requested)\n"
        f"{projection_block}\n\n"
        "USER MESSAGE\n"
        f"{user_text}\n"
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    _init_state()

    # Sidebar action buttons: left-aligned, borderless by default, visible on hover.
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            justify-content: flex-start !important;
            text-align: left !important;
            border: 1px solid transparent;
            background: transparent;
            box-shadow: none;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button > div,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button p,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button span {
            justify-content: flex-start !important;
            text-align: left !important;
            width: 100% !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            border-color: rgba(250, 250, 250, 0.22);
            background: rgba(255, 255, 255, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(APP_TITLE)
    st.caption(SAFETY_NOTE)

    # Sidebar: safety note and quick actions
    with st.sidebar:
        st.caption("What it will NOT do: medical advice, risky exercise prescriptions, sharing personal data without consent, or guaranteed results.")
        st.divider()

        # Only show editable setup in the sidebar after initial setup is complete.
        if st.session_state.profile and st.session_state.page == "chat":
            _render_quick_setup_sidebar()
            if st.button("Log Workout", use_container_width=True, key="open_log_workout"):
                _dialog_structured_log()
        else:
            st.caption("Complete setup first to enable logging.")

    # Phase 1: Setup (main page)
    if st.session_state.page == "setup" or not st.session_state.profile:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Get Started", use_container_width=True, key="initial_setup_btn"):
                _dialog_quick_setup()
        return

    profile: Dict[str, Any] = st.session_state.profile

    _show_weekly_summary_if_sunday(profile)
    _progress_block(profile)

    st.divider()
    if st.session_state.prs:
        with st.expander("Personal records"):
            # Show most recent first
            for pr in reversed(st.session_state.prs[-20:]):
                st.write(f"{pr.get('date')} — {pr.get('lift')}: {pr.get('value')} {pr.get('unit')}")

    st.divider()
    st.subheader("Chat")
    _render_chat()

    user_text = st.chat_input("Build me a workout plan…")
    if user_text:
        _user_msg(user_text)
        st.session_state.pending_user_text = user_text
        st.rerun()

    pending_user_text = st.session_state.get("pending_user_text")
    if pending_user_text:
        user_text = str(pending_user_text)
        today = date.today()

        logs = coerce_logs(st.session_state.workout_logs)

        action_notes: List[str] = []
        projection_note: Optional[str] = None

        # 0) Personal record logging
        pr = parse_pr_message(user_text, today=today)
        if pr:
            st.session_state.prs.append(pr)
            action_notes.append(f"PR logged: {pr.get('lift')} {pr.get('value')} {pr.get('unit')} ({pr.get('date')})")

        # 1) Quick logging
        entry = parse_quick_log_message(
            user_text,
            today=today,
            default_duration_min=int(profile.get("session_minutes") or 60),
            weight_kg=profile.get("weight_kg"),
        )
        if entry:
            st.session_state.workout_logs.append(entry.to_dict())
            logs = coerce_logs(st.session_state.workout_logs)
            action_notes.append(
                f"Workout logged: type={entry.workout_type}, duration_min={entry.duration_min}, sets={entry.sets or 'n/a'}, date={entry.entry_date.isoformat()}"
            )

        # 2) Projection questions
        target_date = try_extract_date_from_question(user_text, today=today)
        if target_date:
            projected = projected_total_workouts_by_date(
                logs=logs,
                start_date=profile["start_date"],
                workout_weekdays=profile["workout_weekdays"],
                target_date=target_date,
            )
            projection_note = f"By {target_date.isoformat()}, projected total workouts ≈ {projected}."

        # 3) Otherwise: Gemini Q&A
        key = _get_gemini_api_key()
        if not key:
            st.session_state.pending_user_text = None
            _assistant_reply(_gemini_setup_hint())
            st.rerun()

        context = _build_chat_context(
            profile=profile,
            logs=logs,
            user_text=user_text,
            action_notes=action_notes,
            projection_note=projection_note,
        )

        try:
            with st.spinner("Thinking..."):
                answer = _gemini_generate(key, context)
        except Exception as e:
            st.session_state.pending_user_text = None
            _assistant_reply(f"Error: {e}")
            st.rerun()

        st.session_state.pending_user_text = None
        _assistant_reply(answer)
        st.rerun()


@st.dialog("Structured Workout Log")
def _dialog_structured_log() -> None:
    """Modal dialog for logging structured workouts."""
    if st.session_state.profile:
        profile = st.session_state.profile
        _apply_structured_log_autofill(profile=profile)

        # Initialize defaults only when keys don't exist.
        if "structured_w_type" not in st.session_state:
            st.session_state["structured_w_type"] = "workout"
        if "structured_duration" not in st.session_state:
            st.session_state["structured_duration"] = int(profile.get("session_minutes") or 60)
        if "structured_sets" not in st.session_state:
            st.session_state["structured_sets"] = 0
        if "structured_notes" not in st.session_state:
            st.session_state["structured_notes"] = ""
        if "structured_date" not in st.session_state:
            st.session_state["structured_date"] = date.today()

        with st.form("structured_log_dialog"):
            w_type = st.selectbox(
                "Workout type",
                ["strength", "cardio", "hiit", "yoga", "flexibility", "workout"],
                key="structured_w_type",
            )
            duration = st.number_input(
                "Duration (minutes)",
                min_value=5,
                max_value=240,
                step=5,
                key="structured_duration",
            )
            sets = st.number_input(
                "Sets (optional)",
                min_value=0,
                max_value=200,
                step=1,
                key="structured_sets",
            )
            notes = st.text_area(
                "Notes (optional)",
                key="structured_notes",
            )
            entry_date = st.date_input(
                "Date",
                key="structured_date",
            )
            ok = st.form_submit_button("Add log")

        if ok:
            from src.fitness_tracker import WorkoutLogEntry, estimate_calories_kcal

            calories = estimate_calories_kcal(
                duration_min=int(duration),
                workout_type=str(w_type),
                weight_kg=profile.get("weight_kg"),
            )
            entry = WorkoutLogEntry(
                entry_date=entry_date,
                workout_type=str(w_type),
                duration_min=int(duration),
                sets=int(sets) if int(sets) > 0 else None,
                notes=notes.strip(),
                calories_kcal=calories,
            )
            st.session_state.workout_logs.append(entry.to_dict())
            st.success("Logged!")
            st.rerun()
    else:
        st.caption("Complete setup first to enable logging.")

if __name__ == "__main__":
    main()
