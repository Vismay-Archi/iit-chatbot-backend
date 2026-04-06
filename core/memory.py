import json
import os
import time
from typing import Optional
from pathlib import Path

# Where session memory is stored — change path if needed
MEMORY_FILE = "session_memory.json"
MAX_HISTORY_PER_SESSION = 20      # Max Q&A pairs to keep per session
SESSION_TTL_HOURS = 24            # Sessions expire after 24 hours of inactivity


def _load_store() -> dict:
    """Load memory store from disk."""
    if Path(MEMORY_FILE).exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_store(store: dict):
    """Save memory store to disk."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
    except Exception as e:
        print(f"⚠️  Memory save failed: {e}")


def _purge_expired(store: dict) -> dict:
    """Remove sessions that haven't been active within TTL."""
    cutoff = time.time() - (SESSION_TTL_HOURS * 3600)
    return {
        sid: data for sid, data in store.items()
        if data.get("last_active", 0) > cutoff
    }


# ── Context extraction ─────────────────────────────────────────────────────────

def extract_user_context(question: str, answer: str) -> dict:
    """
    Extract useful facts about the user from a Q&A pair.
    Returns a dict of context fields to update — empty dict if nothing found.
    """
    context = {}
    q = question.lower()
    a = answer.lower()
    combined = q + " " + a

    # Student level
    if any(w in combined for w in ["i am a graduate", "i'm a graduate", "as a graduate",
                                    "i am a grad", "i'm a grad", "graduate student",
                                    "master's student", "phd student", "doctoral student"]):
        context["student_level"] = "graduate"
    elif any(w in combined for w in ["i am an undergraduate", "i'm an undergraduate",
                                      "as an undergraduate", "i am an undergrad",
                                      "i'm an undergrad", "undergraduate student",
                                      "bachelor's student"]):
        context["student_level"] = "undergraduate"

    # F-1 / international status
    if any(w in combined for w in ["f-1", "f1 visa", "international student",
                                    "i am international", "i'm international"]):
        context["visa_status"] = "F-1"

    # Program
    if "coterminal" in combined or "co-terminal" in combined:
        context["program"] = "coterminal"
    elif "mba" in combined:
        context["program"] = "MBA"
    elif "law" in combined or "chicago-kent" in combined:
        context["program"] = "law"

    # Campus
    if "mies campus" in combined or "main campus" in combined:
        context["campus"] = "Mies"
    elif "downtown campus" in combined or "rice campus" in combined:
        context["campus"] = "Downtown"

    # Graduation term
    for term in ["spring 2026", "summer 2026", "fall 2026", "spring 2027"]:
        if f"graduating in {term}" in combined or f"graduate in {term}" in combined:
            context["graduation_term"] = term.title()

    return context


# ── Public API ─────────────────────────────────────────────────────────────────

def get_session(session_id: str) -> dict:
    """
    Get or create a session. Returns:
    {
        "history": [{"question": ..., "answer": ...}, ...],
        "user_context": {"student_level": ..., "visa_status": ..., ...},
        "last_active": timestamp
    }
    """
    store = _load_store()
    store = _purge_expired(store)

    if session_id not in store:
        store[session_id] = {
            "history": [],
            "user_context": {},
            "last_active": time.time()
        }
        _save_store(store)

    return store[session_id]


def save_turn(session_id: str, question: str, answer: str):
    """
    Save a Q&A turn to session memory.
    Also extracts and updates user context from the conversation.
    """
    store = _load_store()
    store = _purge_expired(store)

    if session_id not in store:
        store[session_id] = {"history": [], "user_context": {}, "last_active": time.time()}

    session = store[session_id]

    # Add to history
    session["history"].append({
        "question": question,
        "answer": answer,
        "timestamp": time.time()
    })

    # Keep only recent history
    if len(session["history"]) > MAX_HISTORY_PER_SESSION:
        session["history"] = session["history"][-MAX_HISTORY_PER_SESSION:]

    # Extract and merge user context
    new_context = extract_user_context(question, answer)
    session["user_context"].update(new_context)

    session["last_active"] = time.time()
    store[session_id] = session
    _save_store(store)


def get_context_string(session_id: str) -> str:
    """
    Build a context string to inject into the LLM prompt.
    Includes recent conversation history and known user facts.
    """
    session = get_session(session_id)
    parts = []

    # User context facts
    user_ctx = session.get("user_context", {})
    if user_ctx:
        facts = []
        if "student_level" in user_ctx:
            facts.append(f"Student level: {user_ctx['student_level']}")
        if "visa_status" in user_ctx:
            facts.append(f"Visa status: {user_ctx['visa_status']}")
        if "program" in user_ctx:
            facts.append(f"Program: {user_ctx['program']}")
        if "campus" in user_ctx:
            facts.append(f"Campus: {user_ctx['campus']}")
        if "graduation_term" in user_ctx:
            facts.append(f"Graduating: {user_ctx['graduation_term']}")
        if facts:
            parts.append("Known about this student: " + "; ".join(facts))

    # Recent conversation history (last 3 turns)
    history = session.get("history", [])
    recent = history[-3:] if len(history) > 3 else history
    if recent:
        history_lines = []
        for turn in recent:
            history_lines.append(f"Student: {turn['question']}")
            # Truncate long answers in history
            ans = turn["answer"][:200] + "..." if len(turn["answer"]) > 200 else turn["answer"]
            history_lines.append(f"Assistant: {ans}")
        parts.append("Recent conversation:\n" + "\n".join(history_lines))

    return "\n\n".join(parts)


def clear_session(session_id: str):
    """Clear a session's memory."""
    store = _load_store()
    if session_id in store:
        del store[session_id]
        _save_store(store)


def get_user_context(session_id: str) -> dict:
    """Get the known user context facts for a session."""
    session = get_session(session_id)
    return session.get("user_context", {})
