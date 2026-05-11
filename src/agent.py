"""
SmartFit Agentic AI module
==========================

**What is Agentic AI?**

Traditional chatbots follow a fixed pipeline:
    User message → (optional RAG lookup) → LLM → text response

An *agentic* AI adds a *reasoning-and-acting loop*.  The LLM is given a set of
**tools** (Python functions) and decides *on its own* which tools to call, in
what order, and how many times before producing a final answer.  This mimics
how a knowledgeable human would tackle a request:

    User: "I just did a 45-min chest workout — how am I doing on my goal?"

    Agent loop, iteration 1:
        Reason:  I should log this workout AND check progress at the same time.
        Action:  call log_workout(workout_type="strength", duration_min=45)
        Observe: "Workout logged: strength, 45 minutes on 2026-05-11."

    Agent loop, iteration 2:
        Action:  call get_progress_summary()
        Observe: "Progress: 18 workouts, 14.2 hours, 60% toward goal."

    Agent loop, iteration 3:
        No more tool calls needed — generate final motivational response.

The five tools exposed to the LLM are:

    search_exercises       — RAG: query the ChromaDB exercise knowledge base
    log_workout            — write a workout entry to session state
    log_personal_record    — write a PR/PB entry to session state
    get_progress_summary   — read aggregate progress stats
    calculate_projection   — project total workouts by a future date

Under the hood this uses **Gemini function calling** (google-genai SDK).  Each
iteration the model either returns a text answer (loop ends) or one-or-more
FunctionCall parts (loop continues: execute tools, feed results back, repeat).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from src.fitness_tracker import (
    WorkoutLogEntry,
    coerce_logs,
    estimate_calories_kcal,
    normalize_workout_type,
    projected_total_workouts_by_date,
    totals,
)
from src.knowledge_base import retrieve_relevant_exercises

# Maximum reasoning iterations before returning whatever text the agent produced.
_MAX_AGENT_ITERATIONS = 5

# ──────────────────────────────────────────────────────────────────────────────
# Tool schemas  (plain-dict format; converted to types.Schema at call time)
# ──────────────────────────────────────────────────────────────────────────────

_FUNCTION_DECLARATIONS: List[Dict[str, Any]] = [
    {
        "name": "search_exercises",
        "description": (
            "Search the fitness exercise knowledge base for relevant exercises "
            "by muscle group, difficulty, or exercise type.  Use this whenever "
            "the user asks about specific exercises, workout routines, or form tips."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": (
                        "Search query describing the exercises needed, "
                        "e.g. 'beginner chest exercises' or 'core strengthening'."
                    ),
                },
                "n_results": {
                    "type": "INTEGER",
                    "description": "Number of results to return (1–5, default 3).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "log_workout",
        "description": (
            "Log a completed workout session for the user.  Call this when the "
            "user says they completed, finished, or did a workout or physical activity."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "workout_type": {
                    "type": "STRING",
                    "description": (
                        "Type of workout: strength, cardio, hiit, yoga, "
                        "flexibility, or workout."
                    ),
                },
                "duration_min": {
                    "type": "INTEGER",
                    "description": "Duration in minutes.",
                },
                "sets": {
                    "type": "INTEGER",
                    "description": "Total sets performed (optional).",
                },
                "notes": {
                    "type": "STRING",
                    "description": "Brief notes about the workout (optional).",
                },
                "entry_date": {
                    "type": "STRING",
                    "description": (
                        "Date of the workout in YYYY-MM-DD format.  "
                        "Defaults to today if omitted."
                    ),
                },
            },
            "required": ["workout_type", "duration_min"],
        },
    },
    {
        "name": "get_progress_summary",
        "description": (
            "Retrieve the user's current fitness progress: total workouts logged, "
            "total hours trained, and goal completion percentage.  Use when the "
            "user asks about their progress, stats, or how they are doing."
        ),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "log_personal_record",
        "description": (
            "Log a personal record (PR) or personal best for a lift or exercise.  "
            "Call this when the user explicitly mentions hitting a new PR or PB."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "lift": {
                    "type": "STRING",
                    "description": (
                        "Name of the lift, e.g. 'bench press', 'squat', 'deadlift'."
                    ),
                },
                "value": {
                    "type": "NUMBER",
                    "description": "The record value (weight, reps, time, etc.).",
                },
                "unit": {
                    "type": "STRING",
                    "description": "Unit: kg, lbs, reps, seconds, or minutes.",
                },
                "entry_date": {
                    "type": "STRING",
                    "description": (
                        "Date in YYYY-MM-DD format.  Defaults to today if omitted."
                    ),
                },
            },
            "required": ["lift", "value", "unit"],
        },
    },
    {
        "name": "calculate_projection",
        "description": (
            "Calculate the projected number of workouts the user will complete "
            "by a target date, based on current history and planned schedule.  "
            "Use when the user asks 'how many workouts by X date' or similar."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "target_date": {
                    "type": "STRING",
                    "description": "Target date in YYYY-MM-DD format.",
                }
            },
            "required": ["target_date"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Schema conversion helper
# ──────────────────────────────────────────────────────────────────────────────

def _dict_to_schema(param_dict: Dict[str, Any]) -> Any:
    """Convert a plain-dict parameter schema to a ``google.genai types.Schema``."""
    from google.genai import types  # type: ignore

    type_map = {
        "STRING": types.Type.STRING,
        "INTEGER": types.Type.INTEGER,
        "NUMBER": types.Type.NUMBER,
        "BOOLEAN": types.Type.BOOLEAN,
        "OBJECT": types.Type.OBJECT,
        "ARRAY": types.Type.ARRAY,
    }
    t = type_map.get((param_dict.get("type") or "STRING").upper(), types.Type.STRING)

    properties: Dict[str, Any] = {}
    for prop_name, prop_schema in (param_dict.get("properties") or {}).items():
        properties[prop_name] = _dict_to_schema(prop_schema)

    return types.Schema(
        type=t,
        properties=properties if properties else None,
        required=param_dict.get("required") or None,
        description=param_dict.get("description") or None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_agent(
    *,
    api_key: str,
    system_prompt: str,
    sandwich_reminder: str,
    context: str,
    workout_logs: List[Dict[str, Any]],
    prs: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> Tuple[str, List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the SmartFit agentic AI loop using Gemini function calling.

    The agent autonomously decides which tools to invoke (RAG search, workout
    logging, progress lookup, projection) before crafting its final response.

    Parameters
    ----------
    api_key:          Gemini API key.
    system_prompt:    High-trust system prompt (passed via system_instruction).
    sandwich_reminder: Appended after user content as defence-in-depth.
    context:          Full structured context string built by _build_chat_context.
    workout_logs:     Current session workout log entries (list of dicts).
    prs:              Current personal records (list of dicts).
    profile:          User fitness profile dict.

    Returns
    -------
    (response_text, action_notes, new_log_entries, new_prs)
        response_text   — final assistant text to display.
        action_notes    — human-readable descriptions of tool actions taken.
        new_log_entries — workout entries created by the agent this turn.
        new_prs         — personal records created by the agent this turn.
    """
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    client = genai.Client(api_key=(api_key or "").strip())

    # ── Mutable state accumulated inside the tool executor ───────────────────
    new_log_entries: List[Dict[str, Any]] = []
    new_prs_list: List[Dict[str, Any]] = []
    action_notes: List[str] = []
    # Working copy of all logs so tools can read up-to-date totals mid-loop.
    all_logs: List[Dict[str, Any]] = list(workout_logs)

    # ── Tool executor ────────────────────────────────────────────────────────

    def _execute_tool(name: str, args: Dict[str, Any]) -> str:
        """Dispatch a function call to the appropriate tool and return its result."""
        nonlocal all_logs

        # ── search_exercises ─────────────────────────────────────────────────
        if name == "search_exercises":
            query = str(args.get("query") or "")
            n = min(int(args.get("n_results") or 3), 5)
            result = retrieve_relevant_exercises(query, n_results=n)
            return result or "No relevant exercises found."

        # ── log_workout ──────────────────────────────────────────────────────
        if name == "log_workout":
            wtype = normalize_workout_type(str(args.get("workout_type") or "workout"))
            duration = max(1, int(args.get("duration_min") or 60))
            sets_val: Optional[int] = (
                int(args["sets"]) if args.get("sets") else None
            )
            notes_val = str(args.get("notes") or "")
            date_str = str(args.get("entry_date") or date.today().isoformat())
            try:
                entry_date = date.fromisoformat(date_str)
            except ValueError:
                entry_date = date.today()

            calories: Optional[int] = None
            if profile.get("weight_kg"):
                calories = estimate_calories_kcal(
                    duration_min=duration,
                    workout_type=wtype,
                    weight_kg=float(profile["weight_kg"]),
                )

            entry = WorkoutLogEntry(
                entry_date=entry_date,
                workout_type=wtype,
                duration_min=duration,
                sets=sets_val,
                notes=notes_val,
                calories_kcal=calories,
            )
            entry_dict = entry.to_dict()
            new_log_entries.append(entry_dict)
            all_logs.append(entry_dict)
            action_notes.append(
                f"Workout logged: type={wtype}, duration_min={duration}, "
                f"sets={sets_val or 'n/a'}, date={entry_date.isoformat()}"
            )
            return (
                f"Workout logged successfully: {wtype}, {duration} minutes "
                f"on {entry_date.isoformat()}."
            )

        # ── get_progress_summary ─────────────────────────────────────────────
        if name == "get_progress_summary":
            logs = coerce_logs(all_logs)
            t = totals(logs)
            target = int(profile.get("target_total_workouts") or 1)
            pct = (
                min(100.0, t["total_workouts"] / target * 100)
                if target > 0
                else 0.0
            )
            cal_part = (
                f", ~{t['total_calories']} kcal burned" if t.get("total_calories") else ""
            )
            return (
                f"Progress: {t['total_workouts']} workouts logged, "
                f"{t['total_hours']:.1f} hours trained"
                f"{cal_part}, "
                f"{pct:.0f}% toward the goal of {target} workouts."
            )

        # ── log_personal_record ──────────────────────────────────────────────
        if name == "log_personal_record":
            lift = str(args.get("lift") or "")
            value = float(args.get("value") or 0)
            unit = str(args.get("unit") or "kg")
            date_str = str(args.get("entry_date") or date.today().isoformat())
            try:
                pr_date = date.fromisoformat(date_str)
            except ValueError:
                pr_date = date.today()

            pr = {
                "lift": lift,
                "value": value,
                "unit": unit,
                "date": pr_date.isoformat(),
            }
            new_prs_list.append(pr)
            action_notes.append(
                f"PR logged: {lift} {value} {unit} ({pr_date.isoformat()})"
            )
            return (
                f"Personal record logged: {lift} — {value} {unit} "
                f"on {pr_date.isoformat()}."
            )

        # ── calculate_projection ─────────────────────────────────────────────
        if name == "calculate_projection":
            date_str = str(args.get("target_date") or "")
            try:
                target_date = date.fromisoformat(date_str)
            except ValueError:
                return "Invalid date format. Please provide a date in YYYY-MM-DD format."

            logs = coerce_logs(all_logs)
            projected = projected_total_workouts_by_date(
                logs=logs,
                start_date=profile["start_date"],
                workout_weekdays=profile["workout_weekdays"],
                target_date=target_date,
            )
            action_notes.append(
                f"Projection: by {target_date.isoformat()}, "
                f"projected total workouts ≈ {projected}"
            )
            return (
                f"By {target_date.isoformat()}, you are projected to complete "
                f"approximately {projected} workouts."
            )

        return f"Unknown tool '{name}'."

    # ── Build Gemini tool declarations ───────────────────────────────────────

    gemini_tools = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=fd["name"],
                description=fd["description"],
                parameters=_dict_to_schema(fd["parameters"]),
            )
            for fd in _FUNCTION_DECLARATIONS
        ]
    )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[gemini_tools],
    )

    # Initial user message: structured context + sandwich defence reminder.
    initial_text = f"{context}\n\n{sandwich_reminder}"
    contents: List[Any] = [
        types.Content(role="user", parts=[types.Part(text=initial_text)])
    ]

    # Fastest-first model fallback list (mirrors _gemini_generate in app.py).
    model_candidates = [
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
    ]

    # ── Agentic reasoning loop ───────────────────────────────────────────────

    last_text = ""
    call_errors: List[str] = []

    for _iteration in range(_MAX_AGENT_ITERATIONS):
        # Try each model candidate until one succeeds.
        response = None
        for model_name in model_candidates:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                # Reset error list on success so only the latest failures are kept.
                call_errors = []
                break
            except Exception as exc:
                call_errors.append(f"{model_name}: {exc}")
                continue

        if response is None:
            raise RuntimeError(
                "Gemini agent error — no model succeeded: "
                + " | ".join(call_errors[:3])
                + (" | ..." if len(call_errors) > 3 else "")
            )

        # ── Inspect the response for tool calls and/or text ──────────────────
        candidate = response.candidates[0]
        function_calls_found: List[Any] = []
        text_parts: List[str] = []

        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                function_calls_found.append(part.function_call)
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        last_text = "".join(text_parts).strip()

        if not function_calls_found:
            # No tool calls → agent is done; return the final text response.
            return last_text, action_notes, new_log_entries, new_prs_list

        # ── Execute tool calls; feed results back into the conversation ──────
        # Add the model's function-call turn to contents.
        contents.append(candidate.content)

        # Execute each function call and collect response Parts.
        function_response_parts: List[Any] = []
        for fc in function_calls_found:
            tool_result = _execute_tool(
                fc.name, dict(fc.args) if fc.args else {}
            )
            function_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": tool_result},
                )
            )

        # Add function responses as the next user turn.
        contents.append(
            types.Content(role="user", parts=function_response_parts)
        )
        # Continue the loop so the agent can reason about tool results.

    # Max iterations reached — return whatever text was last generated.
    return (
        last_text or "I've processed your request.",
        action_notes,
        new_log_entries,
        new_prs_list,
    )
