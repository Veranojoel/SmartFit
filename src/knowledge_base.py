from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore


_EXERCISES: List[Dict[str, str]] = [
    {
        "id": "1",
        "name": "Push-ups",
        "difficulty": "beginner",
        "muscles": "chest, shoulders, triceps",
        "instructions": "Place hands shoulder-width apart, lower body until chest nearly touches floor, push back up",
    },
    {
        "id": "2",
        "name": "Squats",
        "difficulty": "beginner",
        "muscles": "legs, glutes, quads",
        "instructions": "Feet shoulder-width apart, lower hips back and down, keep chest up, return to standing",
    },
    {
        "id": "3",
        "name": "Deadlifts",
        "difficulty": "intermediate",
        "muscles": "back, glutes, hamstrings",
        "instructions": "Feet hip-width apart, grip bar, drive through heels to stand up, lower with control",
    },
    {
        "id": "4",
        "name": "Bench Press",
        "difficulty": "intermediate",
        "muscles": "chest, shoulders, triceps",
        "instructions": "Lie flat, grip bar shoulder-width, lower to chest, press back up",
    },
    {
        "id": "5",
        "name": "Pull-ups",
        "difficulty": "intermediate",
        "muscles": "back, biceps, shoulders",
        "instructions": "Grip bar with hands shoulder-width apart, pull body up until chin above bar, lower with control",
    },
    {
        "id": "6",
        "name": "Planks",
        "difficulty": "beginner",
        "muscles": "core, shoulders",
        "instructions": "Forearms on ground, body in straight line, hold position without sagging",
    },
    {
        "id": "7",
        "name": "Running",
        "difficulty": "beginner",
        "muscles": "legs, cardio",
        "instructions": "Maintain steady pace, keep posture upright, breathe regularly",
    },
    {
        "id": "8",
        "name": "Burpees",
        "difficulty": "advanced",
        "muscles": "full body, cardio",
        "instructions": "Squat, place hands down, jump feet back, do push-up, jump feet forward, jump up",
    },
]


_collection: Any = None
_user_collection: Any = None
_user_docs: List[Dict[str, str]] = []
_user_docs_enabled = True
_last_exercise_hits: List[str] = []
_last_user_hits: List[str] = []

# Chunking constants for user PDF text
_USER_DOC_CHUNK_SIZE = 800
_USER_DOC_CHUNK_OVERLAP = 120


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _chunk_text(text: str) -> List[str]:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return []

    chunks: List[str] = []
    step = max(1, _USER_DOC_CHUNK_SIZE - _USER_DOC_CHUNK_OVERLAP)
    for i in range(0, len(clean), step):
        chunk = clean[i:i + _USER_DOC_CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _simple_retrieve(query: str, n_results: int) -> str:
    tokens = set(_tokenize(query))
    if not tokens:
        return "No relevant exercises found"

    scored: List[tuple[int, Dict[str, str]]] = []
    q_lower = (query or "").lower()
    for ex in _EXERCISES:
        haystack = f"{ex.get('name','')} {ex.get('difficulty','')} {ex.get('muscles','')} {ex.get('instructions','')}".lower()
        score = sum(1 for t in tokens if t in haystack)
        if ex.get("name", "").lower() in q_lower:
            score += 5
        if score > 0:
            scored.append((score, ex))

    if not scored:
        return "No relevant exercises found"

    scored.sort(key=lambda x: (-x[0], x[1].get("name", "")))
    top = [ex for _, ex in scored[: max(1, min(int(n_results), 5))]]

    parts: List[str] = []
    for ex in top:
        parts.append(
            f"{ex['name']} ({ex['difficulty']})\n"
            f"Muscles: {ex['muscles']}\n"
            f"Instructions: {ex['instructions']}"
        )
    return "\n\n".join(parts)


def _get_collection() -> Optional[Any]:
    global _collection

    if _collection is not None:
        return _collection

    if chromadb is None:
        _collection = None
        return None

    try:
        client = chromadb.Client()
        if hasattr(client, "get_or_create_collection"):
            collection = client.get_or_create_collection(name="fitness_exercises")
        else:
            try:
                collection = client.get_collection(name="fitness_exercises")
            except Exception:
                collection = client.create_collection(name="fitness_exercises")

        ids = [ex["id"] for ex in _EXERCISES]
        documents = [f"{ex['name']}: {ex['instructions']}" for ex in _EXERCISES]
        metadatas = [{"difficulty": ex["difficulty"], "muscles": ex["muscles"]} for ex in _EXERCISES]

        if hasattr(collection, "upsert"):
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        else:
            try:
                collection.add(ids=ids, documents=documents, metadatas=metadatas)
            except Exception:
                # Some Chroma versions error on duplicate IDs; safe to ignore.
                pass

        _collection = collection
        return collection
    except Exception:
        _collection = None
        return None


def _get_user_collection() -> Optional[Any]:
    global _user_collection

    if _user_collection is not None:
        return _user_collection

    if chromadb is None:
        _user_collection = None
        return None

    try:
        client = chromadb.Client()
        if hasattr(client, "get_or_create_collection"):
            collection = client.get_or_create_collection(name="user_docs")
        else:
            try:
                collection = client.get_collection(name="user_docs")
            except Exception:
                collection = client.create_collection(name="user_docs")

        _user_collection = collection
        return collection
    except Exception:
        _user_collection = None
        return None


def retrieve_relevant_exercises(query: str, n_results: int = 3) -> str:
    """Search for relevant exercises based on a user query.

    If ChromaDB is installed, use vector search; otherwise fall back to a simple
    keyword search over the built-in exercise list.
    """
    n = max(1, min(int(n_results or 3), 5))

    global _last_exercise_hits
    global _last_user_hits

    collection = _get_collection()
    if collection is None:
        simple = _simple_retrieve(query, n)
        _last_exercise_hits = [p for p in simple.split("\n\n") if p]
        _last_user_hits = []
        return simple

    try:
        results = collection.query(query_texts=[query], n_results=n)
        docs = (results or {}).get("documents") or []
        exercise_list = [str(d) for d in (docs[0] if docs and docs[0] else [])]
        _last_exercise_hits = exercise_list

        user_hits = retrieve_relevant_user_docs(query, n_results=n)
        _last_user_hits = [p for p in user_hits.split("\n\n") if p] if user_hits else []

        exercise_hits = "\n".join(exercise_list) if exercise_list else ""
        combined = "\n\n".join(p for p in [exercise_hits, user_hits] if p)
        return combined or "No relevant exercises found"
    except Exception:
        simple = _simple_retrieve(query, n)
        _last_exercise_hits = [p for p in simple.split("\n\n") if p]
        _last_user_hits = []
        return simple


def add_user_documents(*, name: str, text: str, doc_id: str) -> int:
    """Add user-provided PDF text to the knowledge base.

    Returns number of chunks indexed.
    """
    chunks = _chunk_text(text)
    if not chunks:
        return 0

    if not _user_docs_enabled:
        return 0

    collection = _get_user_collection()
    if collection is None:
        # Fallback: store in memory for keyword search.
        for idx, chunk in enumerate(chunks):
            _user_docs.append(
                {
                    "id": f"{doc_id}-{idx}",
                    "name": name,
                    "text": chunk,
                }
            )
        return len(chunks)

    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    documents = chunks
    metadatas = [{"name": name, "doc_id": doc_id} for _ in chunks]

    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    else:
        try:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
        except Exception:
            pass

    return len(chunks)


def clear_user_documents() -> None:
    global _user_docs
    collection = _get_user_collection()
    if collection is not None:
        try:
            collection.delete(where={})
        except Exception:
            pass
    _user_docs = []


def set_user_docs_enabled(enabled: bool) -> None:
    global _user_docs_enabled
    _user_docs_enabled = bool(enabled)


def retrieve_relevant_user_docs(query: str, n_results: int = 3) -> str:
    if not _user_docs_enabled:
        return ""
    n = max(1, min(int(n_results or 3), 5))

    collection = _get_user_collection()
    if collection is None:
        # Fallback keyword search over in-memory chunks.
        tokens = set(_tokenize(query))
        scored: List[tuple[int, Dict[str, str]]] = []
        for doc in _user_docs:
            haystack = (doc.get("text") or "").lower()
            score = sum(1 for t in tokens if t in haystack)
            if score > 0:
                scored.append((score, doc))
        if not scored:
            return ""
        scored.sort(key=lambda x: (-x[0], x[1].get("id", "")))
        top = [d for _, d in scored[:n]]
        return "\n\n".join(d.get("text", "") for d in top if d.get("text"))

    try:
        results = collection.query(query_texts=[query], n_results=n)
        docs = (results or {}).get("documents") or []
        if docs and docs[0]:
            return "\n".join(str(d) for d in docs[0])
        return ""
    except Exception:
        return ""


def get_last_retrievals() -> Dict[str, List[str]]:
    return {
        "exercise": list(_last_exercise_hits),
        "user": list(_last_user_hits),
    }
