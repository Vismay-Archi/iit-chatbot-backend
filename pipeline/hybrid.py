"""
Hybrid pipeline — conversational + stress-test hardened.

Layer 0 — Scope guard: refuses out-of-scope / hallucination traps
Layer 1 — Clarify first: asks before retrieving on ambiguous/incomplete questions
Layer 1b— Clarification MERGE: if last bot turn was a clarification Q, merge with prior Q
Layer 2 — Parallel run: Self-Query + Traffic Cop run simultaneously
Layer 3 — Score + pick: identical scorer picks the better answer
Layer 4 — Memory: winning answer saved ONCE (sub-pipelines told not to save)
"""

import asyncio
import re
import time
from typing import Optional, Dict, Any

from pipeline.self_query  import run_self_query
from pipeline.traffic_cop import run_traffic_cop, needs_followup
from core.memory import save_turn, get_session
from core.llm import call_llm

# ── Scope / hallucination guards ───────────────────────────────────────────────

_OOS_PATTERNS = [
    r"\bpresident\b.{0,30}\biit\b|\biit\b.{0,30}\bpresident\b",
    r"\bacceptance rate\b",
    r"\bhow many students\b|\bstudent (enrollment|population|count)\b",
    r"\bwhen did\s+ww[i1]",
    r"\bwhat courses will\b|\bcourses?.{0,20}next year\b",
    r"\b(mars|moon|london|paris|tokyo)\s+campus\b",
]

_TRAP_PATTERNS = [
    (r"\b20[3-9]\d\b",
     "My data covers the 2025–2026 academic year only. I don't have information for future academic years."),
    (r"\bwinter (semester|term|session)\b",
     "Illinois Tech does not have a Winter semester. The available terms are Spring, Fall, Summer 1, and Summer 2."),
]


def _check_oos(q: str) -> Optional[str]:
    ql = q.lower()
    for p in _OOS_PATTERNS:
        if re.search(p, ql):
            return ("I don't have that information in my knowledge base. "
                    "For questions about IIT admissions statistics, leadership, or general "
                    "knowledge, please visit iit.edu or contact the appropriate office directly.")
    return None


def _check_trap(q: str) -> Optional[str]:
    ql = q.lower()
    for p, msg in _TRAP_PATTERNS:
        if re.search(p, ql):
            return msg
    return None


# ── Clarification helpers ──────────────────────────────────────────────────────

def _has_term(q: str) -> bool:
    return any(t in q.lower() for t in ["spring", "fall", "summer", "2025", "2026", "2027"])

def _has_level(q: str) -> bool:
    return any(t in q.lower() for t in [
        "undergrad", "undergraduate", "grad", "graduate",
        "master", "phd", "doctoral", "ug", "bachelor",
    ])

# Regex → clarification message pairs (exact bare questions)
_EXACT_CLARIFIERS = [
    (re.compile(r"^(when does? (the |this )?(semester|term|classes?|courses?|school)\s*(start|begin|end|finish|close)\.?\??)$", re.I),
     "Which semester are you asking about — Spring, Fall, Summer 1, or Summer 2? And which year?"),
    (re.compile(r"^(when is (graduation|commencement|the graduation ceremony|the ceremony)\.?\??)$", re.I),
     "Which graduation ceremony — Spring 2026, Summer 2026, or another term?"),
    (re.compile(r"^(how much (is |does )?(tuition|it cost|the tuition)\.?\??)$", re.I),
     "Are you asking about **undergraduate** or **graduate** tuition? And which semester?"),
    (re.compile(r"^(when do i (pay|make a payment|pay tuition)\.?\??)$", re.I),
     "Which semester's payment deadline — Spring, Fall, Summer 1, or Summer 2?"),
    (re.compile(r"^(what happens if i withdraw\.?\??)$", re.I),
     "Do you mean **withdrawing from a course** (W grade on transcript) or **withdrawing from the university** entirely?"),
    (re.compile(r"^(how many credits (do i need|are required|should i take)\.?\??)$", re.I),
     "Are you asking about **undergraduate** or **graduate** credit requirements?"),
]

# Contextual clarifiers for longer incomplete questions
_CTX_CLARIFIERS = [
    (lambda ql: (any(x in ql for x in ["semester start", "term start", "classes begin",
                                        "semester end", "term end", "when does the semester",
                                        "when does the term", "when do classes start"])
                 and not _has_term(ql)),
     "Which semester are you asking about — Spring, Fall, Summer 1, or Summer 2?"),

    (lambda ql: (any(x in ql for x in ["how much is tuition", "tuition cost", "tuition rate",
                                        "cost per credit", "cost of tuition"])
                 and not _has_level(ql)),
     "Are you asking about **undergraduate** or **graduate** tuition?"),

    (lambda ql: (any(x in ql for x in ["when is payment due", "when is tuition due",
                                        "payment deadline"])
                 and not _has_term(ql)),
     "Which semester's payment deadline — Spring, Fall, or Summer?"),

    (lambda ql: (any(x in ql for x in ["when is graduation", "when is commencement"])
                 and not _has_term(ql)),
     "Which graduation ceremony? (e.g. Spring 2026, Summer 2026)"),
]


def _needs_clarification(question: str) -> Optional[str]:
    """Returns a clarifying question if the question is ambiguous, else None."""
    q, ql = question.strip(), question.strip().lower()
    for regex, msg in _EXACT_CLARIFIERS:
        if regex.match(q):
            return msg
    for check_fn, msg in _CTX_CLARIFIERS:
        if check_fn(ql):
            return msg
    return needs_followup(question)   # delegate to traffic_cop's clarifier


# ── Clarification merge ────────────────────────────────────────────────────────

def _get_last_clarification(session_id: str) -> Optional[str]:
    """
    If the bot's last turn was a clarification question (not a real answer),
    return the original user question that triggered it so we can merge.
    """
    if not session_id:
        return None
    session = get_session(session_id)
    history = session.get("history", [])
    if len(history) < 1:
        return None
    last = history[-1]
    last_answer = last.get("answer", "")
    # Clarification answers are short and end with "?"
    if last_answer.strip().endswith("?") and len(last_answer.split()) < 30:
        return last.get("question", "")
    return None


def _merge_clarification(original_q: str, followup: str) -> str:
    """
    Merge a clarification answer back into the original question.
    e.g. original="When does the semester start?" + followup="spring 2026"
         → "When does the Spring 2026 semester start?"

    If the follow-up is short (<= 6 words), merge. Otherwise use as-is.
    """
    if len(followup.split()) <= 6:
        # Build a natural merged question
        orig_lower = original_q.lower().rstrip("?").strip()
        fu_lower   = followup.lower().strip()

        # semester/term questions
        if any(x in orig_lower for x in ["semester start", "term start", "when does the semester",
                                          "when do classes", "classes begin", "courses begin"]):
            return f"When does the {followup.strip().title()} semester start?"

        if any(x in orig_lower for x in ["semester end", "term end", "last day of classes"]):
            return f"When does the {followup.strip().title()} semester end?"

        if any(x in orig_lower for x in ["graduation", "commencement"]):
            return f"When is {followup.strip().title()} commencement?"

        if any(x in orig_lower for x in ["tuition", "cost", "how much"]):
            return f"What is the {followup.strip()} tuition?"

        if any(x in orig_lower for x in ["pay", "payment", "payment due"]):
            return f"When is {followup.strip().title()} tuition payment due?"

        if any(x in orig_lower for x in ["withdraw"]):
            return f"What happens if I {followup.strip()}?"

        if any(x in orig_lower for x in ["credits", "credit hours"]):
            return f"How many credits does a {followup.strip()} student need?"

        # Generic merge — append follow-up to original
        return f"{original_q.rstrip('?').strip()} for {followup.strip()}?"

    # Long follow-up — use as the full question
    return followup


# ── Scorer ─────────────────────────────────────────────────────────────────────

_YYYY_MM_DD = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_GOOD_DATE  = re.compile(
    r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
    r"(?:st|nd|rd|th)?,?(?:\s+\d{4})?\b)|(\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b)", re.I)
_NUMBERS    = re.compile(r"\$[\d,]+|\b\d+\s*credit|\b\d+\s*hour")
_URL        = re.compile(r"https?://\S+")
_HEDGE      = re.compile(
    r"\b(likely|probably|may vary|please verify|check the calendar|"
    r"i recommend checking|it is recommended)\b", re.I)
_META       = re.compile(r"based on the provided context|according to the context", re.I)
_FT_ERROR   = re.compile(
    r"(9\s*(credits?|hours?).{0,60}(full.?time)|full.?time.{0,60}9\s*(credits?|hours?))"
    r".{0,80}(undergrad|ug|bachelor)", re.I)
# Penalise answers that invent a numbered list when no list was asked for
_FAKE_LIST  = re.compile(r"^\s*\d+\.\s+what is", re.I | re.MULTILINE)


def _score(answer: str, question: str) -> tuple:
    if not answer or len(answer.strip()) < 10:
        return 0, ["empty answer"]
    s, r = 0, []

    if _YYYY_MM_DD.search(answer):   s -= 4; r.append("-4 YYYY-MM-DD format")
    if _GOOD_DATE.search(answer):    s += 3; r.append("+3 explicit date")
    if _NUMBERS.search(answer):      s += 2; r.append("+2 specific numbers")
    if _URL.search(answer):          s += 2; r.append("+2 URL cited")
    if _HEDGE.search(answer):        s -= 3; r.append("-3 hedging")
    if _META.search(answer):         s -= 2; r.append("-2 meta-phrase")
    if _FT_ERROR.search(answer):     s -= 5; r.append("-5 full-time error")
    if _FAKE_LIST.search(answer):    s -= 6; r.append("-6 invented question list")

    ql = question.lower()
    date_q = any(x in ql for x in ["when", "what date", "deadline", "last day"])
    if date_q and not _GOOD_DATE.search(answer) and not _YYYY_MM_DD.search(answer):
        s -= 3; r.append("-3 date expected but missing")

    words = len(answer.split())
    cap = min(words // 10, 5)
    if cap: s += cap; r.append(f"+{cap} completeness")
    if len(answer.split(".")) >= 3 and len(ql.split()) > 8:
        s += 2; r.append("+2 multi-sentence")
    if words < 20 and len(ql.split()) > 8:
        s -= 3; r.append("-3 too short")

    return s, r


async def judge_answers(question: str, answer_a: str, answer_b: str) -> str:
    prompt = f"""
You are an evaluator.

Choose which answer better answers the question.

Criteria:
- factual correctness
- completeness
- directness

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Output ONLY: A or B
"""
    res = await call_llm(prompt)
    return res.strip()


# ── Main entry point ───────────────────────────────────────────────────────────

async def run_hybrid(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Conversational hybrid pipeline.

    Step 0 — Refuse out-of-scope and hallucination traps
    Step 1 — Check if this is a clarification reply; if so, merge with prior question
    Step 2 — Check if current question needs clarification; if so, ask and stop
    Step 3 — Run SQ + TC in parallel (pass session_id to TC only; SQ has no memory yet)
              NOTE: tell TC not to save to memory (we save once here, in step 5)
    Step 4 — Score both answers; pick winner
    Step 5 — Save ONE turn to memory
    """
    start = time.time()

    # ── Step 0: Scope guards ───────────────────────────────────────────────────
    oos = _check_oos(question)
    if oos:
        if session_id:
            save_turn(session_id, question, oos)
        return {"pipeline": "hybrid", "answer": oos, "winner": "scope_guard",
                "refused": True, "source_urls": [], "chunks_used": 0,
                "response_time_ms": round((time.time() - start) * 1000)}

    trap = _check_trap(question)
    if trap:
        if session_id:
            save_turn(session_id, question, trap)
        return {"pipeline": "hybrid", "answer": trap, "winner": "hallucination_guard",
                "refused": True, "source_urls": [], "chunks_used": 0,
                "response_time_ms": round((time.time() - start) * 1000)}

    # ── Step 1: Clarification merge ────────────────────────────────────────────
    # If the bot's last response was a clarification question, this reply is the answer to it.
    # Merge them into a proper complete question before sending to retrieval.
    effective_question = question
    original_q = _get_last_clarification(session_id)
    if original_q:
        effective_question = _merge_clarification(original_q, question)

    # ── Step 2: Clarification check (only if NOT already a clarification reply) ─
    if not original_q:
        clarification = _needs_clarification(question)
        if clarification:
            if session_id:
                save_turn(session_id, question, clarification)
            return {"pipeline": "hybrid", "answer": clarification,
                    "clarification_suggestion": clarification,
                    "winner": "clarification", "source_urls": [], "chunks_used": 0,
                    "response_time_ms": round((time.time() - start) * 1000)}

    # ── Step 3: Run SQ + TC in parallel ───────────────────────────────────────
    # Pass session_id=None to TC here — we will save the turn ourselves in step 5
    # to avoid double-saving (TC's run_traffic_cop also calls save_turn internally).
    sq_res, tc_res = await asyncio.gather(
        run_self_query(effective_question),
        run_traffic_cop(effective_question, session_id=None),  # no save inside TC
        return_exceptions=True,
    )

    sq_ok = not isinstance(sq_res, Exception) and bool(sq_res.get("answer"))
    tc_ok = not isinstance(tc_res, Exception) and bool(tc_res.get("answer"))

    if not sq_ok and not tc_ok:
        err_msg = "Unable to retrieve an answer at this time. Please try again."
        if session_id:
            save_turn(session_id, question, err_msg)
        return {"pipeline": "hybrid", "answer": err_msg,
                "source_urls": [], "chunks_used": 0,
                "response_time_ms": round((time.time() - start) * 1000)}

    # ── Step 4: Score and pick ─────────────────────────────────────────────────
    scores: Dict[str, Any] = {}
    if not sq_ok:
        winner, winning = "traffic_cop", tc_res
    elif not tc_ok:
        winner, winning = "self_query", sq_res
    else:
        if sq_res["answer"] == tc_res["answer"]:
            winner, winning = "self_query", sq_res
        else:
            decision = await judge_answers(
                effective_question,
                sq_res["answer"],
                tc_res["answer"]
            )

            if decision not in ["A", "B"]:
                if len(sq_res["answer"]) >= len(tc_res["answer"]):
                    winner, winning = "self_query", sq_res
                else:
                    winner, winning = "traffic_cop", tc_res
            else:
                if decision == "A":
                    winner, winning = "self_query", sq_res
                else:
                    winner, winning = "traffic_cop", tc_res

    # ── Step 5: Save ONE turn (keyed to original user question, not merged) ────
    answer = winning.get("answer", "")
    if session_id and answer:
        save_turn(session_id, question, answer)

    return {
        "pipeline":    "hybrid",
        "answer":      answer,
        "winner":      winner,
        "scores":      scores,
        "source_urls": winning.get("source_urls", []),
        "chunks_used": winning.get("chunks_used", 0),
        "clarification_suggestion": None,
        "response_time_ms": round((time.time() - start) * 1000),
    }