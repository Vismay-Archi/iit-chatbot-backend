# traffic_cop_faiss.py
# Run:
#   source .venv/bin/activate
#   in terminal run export ON_DEMAND_API_ACCESS_TOKEN="token"
#   python traffic_cop_faiss.py

from __future__ import annotations

import os
#  macOS stability knobs 
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEBUG_THETA = False
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import faiss
import re
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# project paths
PKL_PATH = "output/faiss_store.pkl"
FAISS_PATH = "output/faiss_store.faiss"

# Theta EdgeCloud endpoint
THETA_URL = "https://ondemand.thetaedgecloud.com/infer_request/llama_3_8b/completions"


# Data containers
@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]


@dataclass
class Store:
    index: faiss.Index
    chunks: List[Chunk]
    model_name: str


# Store loading
def _coerce_chunk(obj: Any) -> Chunk:
    """
    Coerce whatever the pkl stored into a {text, meta} pair.
    Supports common shapes:
      - LangChain Document-like: obj.page_content + obj.metadata
      - dict: {text/content/page_content/chunk, meta/metadata}
      - tuple/list: (text, meta)
    """
    # LangChain Document-like
    if hasattr(obj, "page_content") and hasattr(obj, "metadata"):
        return Chunk(text=str(obj.page_content), meta=dict(obj.metadata or {}))

    # dict-based
    if isinstance(obj, dict):
        text = obj.get("text") or obj.get("page_content") or obj.get("content") or obj.get("chunk") or ""
        meta = obj.get("meta") or obj.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {"meta": meta}
        return Chunk(text=str(text), meta=meta)

    # tuple/list (text, meta)
    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        text = obj[0]
        meta = obj[1] if len(obj) > 1 else {}
        if not isinstance(meta, dict):
            meta = {"meta": meta}
        return Chunk(text=str(text), meta=meta)

    # fallback
    return Chunk(text=str(obj), meta={})


def load_store(pkl_path: str = PKL_PATH, faiss_path: str = FAISS_PATH) -> Store:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing: {pkl_path}")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"Missing: {faiss_path}")

    with open(pkl_path, "rb") as f:
        store_obj = pickle.load(f)

    if not isinstance(store_obj, dict):
        raise ValueError("faiss_store.pkl must be a dict.")
    if "chunks" not in store_obj:
        raise ValueError("faiss_store.pkl missing key: 'chunks'")

    model_name = store_obj.get("model", "all-mpnet-base-v2")

    raw_chunks = store_obj["chunks"]
    if not isinstance(raw_chunks, list):
        raise ValueError("store['chunks'] must be a list")

    chunks = [_coerce_chunk(c) for c in raw_chunks]
    index = faiss.read_index(faiss_path)

    # Warn only (not fatal)
    try:
        if index.ntotal != len(chunks):
            print(f"[WARN] index.ntotal={index.ntotal} but chunks={len(chunks)} (may still work).")
    except Exception:
        pass

    return Store(index=index, chunks=chunks, model_name=model_name)

# Traffic-cop clarification rules (expandable)

TERMS = ["spring", "summer 1", "summer", "summer 2", "fall", "winter"]
GRAD_WORDS = [
    "grad student", "graduate student",
    "masters", "master's", "ms", "m.s.", "m.s",
    "phd", "ph.d.", "doctoral", "doctorate",
    "postgraduate", "graduate program"]
UNDERGRAD_WORDS = ["undergrad", "undergraduate", "ug", "bachelor", "bachelors"]


def _is_calendar_chunk(meta: Dict[str, Any], text: str) -> bool:
    meta = meta or {}
    blob = " ".join([
        str(meta.get("source", "")),
        str(meta.get("title", "")),
        str(meta.get("url", "")),
        str(meta.get("file", "")),
        str(meta.get("path", "")),
        text or "",
    ]).lower()

    return any(k in blob for k in [
        "academic calendar", "registrar calendar", "calendar",
        "courses begin", "classes begin", "spring break", "fall break",
        "add/drop", "withdrawal deadline", "last day of classes",
        "final exam", "commencement", "degree conferral"
    ])


def _has_any(q: str, words: List[str]) -> bool:
    return any(w in q for w in words)

def _mentions_term(q: str) -> bool:
    return _has_any(q, TERMS)

def _mentions_level(q: str) -> bool:
    q = (q or "").lower()

    # phrase-first checks
    if any(x in q for x in [
        "graduate student", "grad student",
        "graduate", "grad",
        "undergraduate", "undergrad", "ug"
    ]):
        return True

    # token/word-boundary checks
    return bool(re.search(
        r"\b(ms|m\.s\.|masters|master's|phd|ph\.d\.|doctoral|doctorate)\b",
        q
    ))

def _mentions_summer_session(q: str) -> bool:
    return ("summer 1" in q) or ("summer i" in q) or ("summer 2" in q) or ("summer ii" in q)

def _is_international_topic(q: str) -> bool:
    return _has_any(q, ["i-20", "i20", "cpt", "opt", "visa", "f-1", "f1", "sevis"])

def _is_max_hours_topic(q: str) -> bool:
    return _has_any(q, [
        "max hours", "maximum hours", "max credits", "maximum credits",
        "max credit hours", "maximum credit hours",
        "overload", "course overload", "credit overload",
        "how many credits can i take", "how many credit hours can i take"
    ])

def _is_it_topic(q: str) -> bool:
    return _has_any(q, ["password", "reset", "log in", "login", "canvas", "blackboard", "portal", "email", "wifi", "wi-fi", "eduroam"])

def _is_location_topic(q: str) -> bool:
    return _has_any(q, ["where is", "where's", "location", "hours", "open", "close", "office", "building"])

def _is_cost_topic(q: str) -> bool:
    return _has_any(q, [
        "tuition", "cost", "fees", "price", "how much", "bill", "payment", "due", "refund",
        "financial aid", "scholarship",
        "full-time", "full time", "part-time", "part time",
    ])

def _is_registration_topic(q: str) -> bool:
    return _has_any(q, ["register", "registration", "enroll", "enrollment", "add/drop", "add drop", "drop a class", "withdraw", "withdrawal"])

def _is_expected_grad_date_topic(q: str) -> bool:
    q = q.lower()

    has_grad_timeline = _has_any(q, [
        "expected graduation", "expected grad",
        "graduate earlier", "graduate sooner",
        "finish earlier", "finish sooner",
        "time to degree", "how soon can i graduate", "how fast can i graduate",
        "move my graduation", "graduation date earlier"
    ])

    has_transfer = _has_any(q, [
        "transfer credit", "transfer credits", "transfer",
        "from another university", "community college",
        "accepted credits", "credit transfer"
    ])

    has_credit_number = bool(re.search(r"\b\d+\s*(credit|credits|credit hours|hours)\b", q))

    return (has_grad_timeline or has_transfer) and (has_credit_number or has_grad_timeline or has_transfer)

def _is_calendar_topic(q: str) -> bool:
    return _has_any(q, ["academic calendar", "calendar", "semester start", "semester begins", "when does the semester", "first day", "finals", "final exam", "exam schedule"])

def _is_calendar_query(q: str) -> bool:
    ql = (q or "").lower()
    return _is_date_event_question(ql) or _is_calendar_topic(ql) or any(x in ql for x in ["how long", "length of", "duration of"])

def _is_graduation_topic(q: str) -> bool:
    return _has_any(q, ["graduation", "graduating", "commencement", "diploma", "degree conferral", "conferral", "cap and gown"])

def _is_date_event_question(q: str) -> bool:
    q = q.lower()
    return _has_any(q, [
        "when is", "what date", "what day",
        "semester start", "classes begin", "courses begin",
        "spring break", "fall break",
        "add/drop", "withdraw", "deadline", "due date",
        "last day of classes", "final exam", "finals",
        "last day to",   # ← "last day to withdraw", "last day to add" (needs object)
    ])

def _expand_query_for_retrieval(q: str) -> str:
    ql = (q or "").lower()
    if any(x in ql for x in ["pass/fail", "pass fail", "p/f", "s/u", "satisfactory", "unsatisfactory"]):
        return q + " policy maximum limit three 3 courses"
    if any(x in ql for x in ["how long is", "how long does", "length of", "duration of"]):
        return q + " courses begin last day of classes start end date"
    if any(x in ql for x in ["walk in", "walk in the", "ceremony"]) and "spring" in ql:
        return q + " spring graduation application deadline degree conferral spring ceremony"
    return q

# Follow-up logic
def needs_followup(user_q: str) -> Optional[str]:
    """
    Traffic-cop clarification logic for ambiguous queries.
    Ask clarifying questions BEFORE retrieval/LLM.
    Preference order: term/session -> level/audience -> specific object/system.
    """
    q = user_q.lower()
    current_indicators = ["this semester", "current semester", "right now", "this fall", "this spring", "this summer"]
    user_specified_current = any(x in q for x in current_indicators)
    if re.search(r"\blast day\b", q) and not _has_any(q, [
        "withdraw", "withdrawal", "add", "drop", "add/drop",
        "classes", "class", "semester", "finals", "register", "registration",
        "apply", "application", "tuition", "payment", "pass/fail"
    ]):
        return "Last day for what? For example: last day to **withdraw**, last day of **classes**, last day to **add/drop**, or last day to **pay tuition**?"

    if _is_expected_grad_date_topic(q):
        if not _mentions_level(q):
            return "Is this for **undergraduate** or **graduate** transfer credit/time-to-degree?"
        return None

    if _is_graduation_topic(q):
        # FIX: If the question explicitly mentions two terms (e.g. "apply for Summer
        # but walk in Spring ceremony"), it is self-contained — do not ask for
        # clarification.  Count distinct term mentions to detect this case.
        term_mentions = sum(1 for t in ["spring", "summer", "fall", "winter"] if t in q)
        has_both_terms = term_mentions >= 2

        if not has_both_terms:
            if not _mentions_term(q) and not user_specified_current:
                return "Which term are you asking about for graduation (Spring, Fall, Summer 1, or Summer 2)?"
            if "summer" in q and not _mentions_summer_session(q):
                return "Do you mean **Summer 1** or **Summer 2** for graduation?"
        if _has_any(q, ["cap", "gown"]) and not _mentions_level(q):
            return "Is this for **undergraduate** or **graduate** commencement?"
        return None


    if _is_max_hours_topic(q):
        if not _mentions_level(q):
            return "Is this for **undergraduate** or **graduate** maximum credit hours/overload?"
        if not _mentions_term(q):
            return "Which term is this for (Fall, Spring, or Summer)?"
        return None

    if _is_calendar_topic(q):
        # If they said "summer" but not which session, ask first
        if "summer" in q and not _mentions_summer_session(q):
            return "Do you mean **Summer 1** or **Summer 2**?"

        if _has_any(q, ["semester start", "semester begins", "when does the semester", "first day", "start of classes", "classes start"]) and not _mentions_term(q):
            return "Which semester are you asking about (Fall, Spring, or Summer)?"

        if _has_any(q, ["finals", "final exam", "exam schedule"]) and not _mentions_term(q):
            return "Which semester is this for (Fall, Spring, or Summer)?"

        return None

    if _is_registration_topic(q):
        # Skip clarification for policy questions that apply universally
        _is_policy_q = _has_any(q, ["refund", "w grade", "gpa", "affect my", "what happens",
                                     "will it appear", "does it affect", "does dropping",
                                     "what is the difference", "transfer credit"])
        if _is_policy_q:
            return None
        # Skip clarification if question has specific credit numbers + drop/withdraw context
        import re as _re
        _has_credit_nums = bool(_re.search(r"\d+\s*(credit|credits)", q))
        if _has_credit_nums and _has_any(q, ["drop", "withdraw", "if i", "after", "still"]):
            return None

        asks_deadline = (
            _has_any(q, ["deadline", "last day", "when is", "by when", "what date"]) and
            _has_any(q, ["add/drop", "add drop", "drop a class", "withdraw", "withdrawal", "register"])
        )
        if asks_deadline:
            if "summer" in q and not _mentions_summer_session(q):
                return "Do you mean **Summer 1** or **Summer 2**?"
            if not _mentions_term(q):
                return "Which term are you asking about (Fall, Spring, or Summer)?"

        if _has_any(q, ["full-time", "full time", "part-time", "part time", "credit", "credits", "credit hours"]) \
            and _has_any(q, ["tuition", "fees", "cost", "bill", "payment"]) \
            and not _mentions_level(q):
            return "Is this for **undergraduate** or **graduate** full-time status/tuition?"

        if _has_any(q, ["registration opens", "when does registration open", "enrollment date"]) and not _mentions_level(q):
            return "Are you asking as an **undergraduate** or **graduate** student?"
        return None

    if _is_cost_topic(q):
        # Skip if question has specific credit numbers + drop context
        import re as _re2
        _has_specific_credits = bool(_re2.search(r"\d+\s*(credit|credits)", q))
        if _has_specific_credits and _has_any(q, ["drop", "withdraw", "if i", "after"]):
            return None
        if _has_any(q, ["full-time", "full time", "part-time", "part time", "credit", "credits", "credit hours"]) and not _mentions_level(q):
            return "Is this for **undergraduate** or **graduate** full-time status/tuition?"

        if _has_any(q, ["tuition", "fees", "cost", "how much"]) and not _mentions_level(q):
            return "Are you asking about **undergraduate** or **graduate** tuition/fees?"
        if _has_any(q, ["payment", "bill", "due", "deadline"]) and not _mentions_term(q):
            return "Which term is the bill/payment for (Fall, Spring, or Summer)?"
        if "refund" in q and not _has_any(q, ["tuition", "drop", "withdraw", "financial aid", "aid"]):
            return "Do you mean a **tuition refund** or a **financial aid refund**?"
        return None

    if _has_any(q, ["apply", "application", "admissions", "deadline", "requirements"]):
        if _has_any(q, ["application", "apply", "admissions"]) and _has_any(q, ["deadline", "due"]) and not _mentions_level(q):
            return "Is this for an **undergraduate** or **graduate** application?"
        if "requirements" in q and not _has_any(q, ["admission", "degree", "major", "program", "prereq", "prerequisite", "course"]):
            return "Do you mean **admission requirements** or **degree/major requirements**?"
        return None

    if _is_international_topic(q):
        if not _has_any(q, ["f-1", "f1"]):
            return "Is this for an **F-1** student?"
        if _has_any(q, ["cpt", "opt"]) and not _has_any(q, ["eligibility", "process", "timeline", "requirements"]):
            return "Are you asking about **eligibility** or the **application process/timeline**?"
        return None

    if _is_location_topic(q):
        if _has_any(q, ["hours", "open", "close", "location", "where is", "where's"]) and not _has_any(
            q, ["registrar", "bursar", "library", "financial aid", "admissions", "campus", "mies", "wishnick", "iit tower", "hermann", "sspa", "ideashop"]
        ):
            return "Which office/building are you asking about?"
        return None

    if _has_any(q, ["parking"]):
        if not _has_any(q, ["student","permit", "visitor", "daily", "rate", "garage"]):
            return "Do you mean **student parking permits**, **visitor parking**, or **daily rates**?"
        return None

    if _has_any(q, ["shuttle", "bus"]):
        if not _has_any(q, ["mies", "downtown", "route", "schedule"]):
            return "Which shuttle route/campus (Mies/Main vs Downtown), and what day/time?"
        return None

    if _is_it_topic(q):
        if _has_any(q, ["password", "reset", "log in", "login"]) and not _has_any(q, ["email", "portal", "canvas", "blackboard", "password manager"]):
            return "Is this for your **IIT email**, the **student portal**, or the **LMS (Canvas/Blackboard)**?"
        if _has_any(q, ["wifi", "wi-fi", "eduroam"]) and not _has_any(q, ["laptop", "phone", "mac", "windows"]):
            return "Are you connecting on a **laptop** or **phone**, and are you using **eduroam**?"
        return None

    if _has_any(q, ["transfer credit", "transfer credits", "transfer"]):
        # GPA/affect questions are same rule for both levels — skip clarification
        if _has_any(q, ["gpa", "affect", "grade point", "count", "included", "calculation"]):
            return None
        if not _mentions_level(q):
            return "Is this for **undergraduate** or **graduate** transfer credit?"
        return None

    if _has_any(q, ["leave of absence", "leave", "withdraw from school", "withdraw from the university"]):
        if not _mentions_level(q):
            return "Is this for an **undergraduate** or **graduate** program?"
        return None

    if _has_any(q, ["pass/fail", "pass fail", "p/f"]):
        if not _mentions_term(q):
            return "Which term is this for (Fall, Spring, or Summer)?"
        return None

    return None

# Retrieval
def _embed_query(embedder: SentenceTransformer, query: str) -> np.ndarray:
    v = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return v.astype(np.float32)


def _rerank(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    q = query.lower()

    wants_start = any(x in q for x in [
        "semester start", "when does", "start", "begin", "classes begin", "courses begin"
    ])
    wants_grad_date = any(x in q for x in [
        "expected graduation", "expected grad", "time to degree",
        "graduate earlier", "graduate sooner", "finish earlier", "finish sooner",
        "transfer credit", "transfer credits", "credit transfer"
    ])
    wants_ug = _has_any(q, UNDERGRAD_WORDS)
    wants_grad = _has_any(q, GRAD_WORDS)
    wants_tuition = any(x in q for x in ["tuition", "fees", "bill", "payment", "refund", "financial aid", "withdraw", "withdrawal"])
    wants_max = any(x in q for x in ["max", "maximum", "overload", "how many credits", "how many credit hours"])
    wants_pass_fail = any(x in q for x in ["pass/fail", "pass fail", "p/f"])
    wants_load = any(x in q for x in [
        "full-time", "full time", "part-time", "part time",
        "credits", "credit hours", "credit"
    ])

    def boost(r: Dict[str, Any]) -> int:
        meta = r.get("meta") or {}
        text = (r.get("text") or "").lower()
        term = str(meta.get("term", "")).lower()
        session = str(meta.get("session", "")).lower()
        b = 0

        # search-relevant blob = text + metadata fields
        blob = " ".join([
            text,
            str(meta.get("title", "")).lower(),
            str(meta.get("url", "")).lower(),
            str(meta.get("source", "")).lower(),
            str(meta.get("file", "")).lower(),
            str(meta.get("path", "")).lower(),
        ])

        if _is_calendar_chunk(r.get("meta") or {}, r.get("text") or ""):
            if "summer 2" in q:
                if session == "summer 2" or "summer 2" in blob:
                    b += 8
                elif any(x in blob for x in ["summer 1", "spring", "fall"]):
                    b -= 8
            elif "summer 1" in q:
                if session == "summer 1" or "summer 1" in blob:
                    b += 8
                elif any(x in blob for x in ["summer 2", "spring", "fall"]):
                    b -= 8
            elif "spring" in q:
                if "spring" in term or "spring" in blob:
                    b += 8
                elif any(x in blob for x in ["summer", "fall"]):
                    b -= 8
            elif "fall" in q:
                if "fall" in term or "fall" in blob:
                    b += 8
                elif any(x in blob for x in ["summer", "spring"]):
                    b -= 8

        passfail_term = any(x in blob for x in ["pass/fail", "pass fail", "p/f", "s/u", "satisfactory", "unsatisfactory"])
        limit_term = any(x in blob for x in ["maximum", "limit", "no more than", "up to", "at most", "three", "3"])

        if wants_pass_fail and passfail_term:
            b += 6  # general boost for pass/fail policy chunks
            if limit_term:
                b += 4  # extra boost for chunks that mention the cap/limit
            
        if wants_grad_date and any(x in blob for x in [
            "transfer credit", "transfer credits", "credit transfer",
            "time to degree", "graduation", "degree requirements"
        ]):
            b += 5
        
        if wants_start and ("courses begin" in blob or "classes begin" in blob):
            b += 5

        if wants_max and any(x in blob for x in ["maximum", "max", "overload", "overload request", "credit overload"]):
            b += 6

        if wants_tuition and any(x in blob for x in ["tuition", "fees", "billing", "refund", "financial aid", "withdraw", "withdrawal", "flat-rate", "per credit"]):
            b += 4

        if wants_load and any(x in blob for x in ["full-time", "part-time", "credit hours", "academic load"]):
            b += 4

        # level-aware boosting ( use blob so URL/title matches help too)
        if wants_grad:
            if "graduate" in blob or "grad" in blob:
                b += 4
            elif "undergraduate" in blob or "undergrad" in blob:
                b -= 2

        if wants_ug:
            if "undergraduate" in blob or "undergrad" in blob:
                b += 4
            elif "graduate" in blob or "grad" in blob:
                b -= 2

            if any(x in blob for x in ["doctoral", "phd", "doctorate"]):
                b -= 10

        return b

    results.sort(key=lambda r: (boost(r), r["score"]), reverse=True)
    return results

_DATE_RE = re.compile(
    r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?(?:\s+\d{4})?\b)"
    r"|(\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b)"
    r"|(\b\d{4}-\d{2}-\d{2}\b)"
    r"|(\b(?:monday|tuesday|wednesday|thursday|friday)\b)",  # "last day is Friday"
    re.I
)
def _has_explicit_date(text: str, meta: Dict[str, Any]) -> bool:
    if meta and meta.get("date"):
        return True
    if not text:
        return False
    return bool(_DATE_RE.search(text))



def _blob(r: Dict[str, Any]) -> str:
    meta = r.get("meta") or {}
    return " ".join([
        (r.get("text") or "").lower(),
        str(meta.get("title", "")).lower(),
        str(meta.get("url", "")).lower(),
        str(meta.get("source", "")).lower(),
        str(meta.get("file", "")).lower(),
        str(meta.get("path", "")).lower(),
    ])

def search(
    store: Store,
    embedder: SentenceTransformer,
    query: str,
    k: int = 8,
    overfetch: int = 200
) -> List[Dict[str, Any]]:


    query2 = _expand_query_for_retrieval(query)

    qv = _embed_query(embedder, query2)
    scores, ids = store.index.search(qv, max(k, overfetch))

    out: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1 or idx >= len(store.chunks):
            continue
        ch = store.chunks[idx]
        out.append({
            "score": float(score),
            "id": int(idx),
            "text": ch.text,
            "meta": ch.meta,
        })

    qlow = (query or "").lower()
    wants_ug = _has_any(qlow, UNDERGRAD_WORDS)
    wants_grad = _has_any(qlow, GRAD_WORDS)
    wants_doctoral = any(x in qlow for x in ["phd", "doctoral", "doctorate"])

    #If NOT a calendar/date question, drop calendar chunks 
    if not _is_calendar_query(qlow):
        out = [r for r in out if not _is_calendar_chunk(r.get("meta") or {}, r.get("text") or "")]
    # Drop no-term calendar chunks when query specifies a term
    if _is_calendar_query(qlow) and _mentions_term(qlow):
        out = [r for r in out if not (
            _is_calendar_chunk(r.get("meta") or {}, r.get("text") or "") and
            not (r.get("meta") or {}).get("term")
        )]

    # Drop short title-only directory chunks (no name, useless for people questions)
    is_people_q = any(x in qlow for x in [
        "who is", "who are", "dean", "president", "director", "chair",
        "provost", "chancellor", "head of", "contact"
    ])
    if is_people_q:
        out = [r for r in out if not (
            (r.get("meta") or {}).get("chunk_type") == "directory_people"
            and len(r.get("text", "")) < 50
        )]

    #If it IS a date/event question, keep only chunks with an explicit date
    if _is_date_event_question(qlow):
        for r in out:
            if _has_explicit_date(r.get("text") or "", r.get("meta") or {}):
                r["_date_boost"] = 10  # add a temp score boost
        out.sort(key=lambda r: (r.pop("_date_boost", 0) + r["score"]), reverse=True)

    #Re-rank remaining results
    out = _rerank(out, query)
    ql = (query or "").lower()

    # extract meaningful terms from query
    terms = [w for w in re.findall(r"[a-z]+", ql) if len(w) > 4]

    # keep only chunks that match the query terms
    filtered = [
        r for r in out
        if sum(t in (r.get("text") or "").lower() for t in terms) >= 2
    ]

    if filtered:
        out = filtered

    # contact-specific filter
    if any(x in ql for x in ["contact", "email", "phone"]):
        # Prefer office/general contact chunks over individual directory entries
        office_filtered = [
            r for r in out
            if any(x in (r.get("text") or "").lower() for x in ["@", "email", "phone"])
            and (r.get("meta") or {}).get("chunk_type") != "directory_people"
        ]
        people_filtered = [
            r for r in out
            if any(x in (r.get("text") or "").lower() for x in ["@", "email", "phone"])
            and (r.get("meta") or {}).get("chunk_type") == "directory_people"
        ]
        # Use office contacts first; fall back to people only if nothing else found
        if office_filtered:
            out = office_filtered
        elif people_filtered:
            out = people_filtered

    # Level filtering
    def is_doctoral_only(r: Dict[str, Any]) -> bool:
        b = _blob(r)
        return any(x in b for x in ["phd", "doctoral", "doctorate"])

    # Remove doctoral-only content unless user explicitly wants doctoral
    if not wants_doctoral:
        out = [r for r in out if not is_doctoral_only(r)]

    # For UG vs Grad:
    # only delete if it's clearly a grad-only transfer-credit page and user wants UG.
    def looks_grad_only_policy(r: Dict[str, Any]) -> bool:
        b = _blob(r)
        # keep this conservative
        return ("catalog.iit.edu/graduate" in b) or ("graduate/academic-policies-procedures" in b)

    def looks_ug_only_policy(r: Dict[str, Any]) -> bool:
        b = _blob(r)
        return ("catalog.iit.edu/undergraduate" in b) or ("undergraduate/academic-policies-procedures" in b)

    if wants_ug:
        out = [r for r in out if not looks_grad_only_policy(r)]

    if wants_grad and not wants_ug:
        out = [r for r in out if not looks_ug_only_policy(r)]

    return out[:k]

def format_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for h in hits:
        meta = h.get("meta") or {}

        url = meta.get("url") or meta.get("source_url") or meta.get("source") or ""

        header_bits = []
        for k in ["term", "session", "date", "title", "file", "path"]:
            if meta.get(k):
                header_bits.append(f"{k.upper()}: {meta[k]}")
        header = " | ".join(header_bits) if header_bits else f"ID: {h.get('id')}"

        text = (h.get("text") or "").strip()

        if url:
            blocks.append(f"SOURCE_URL: {url}\n{header}\n{text}")
        else:
            blocks.append(f"{header}\n{text}")   # no SOURCE_URL label at all

    return "\n\n".join(blocks)


# Theta chat call 
def theta_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 500,
    temperature: float = 0.3,
    top_p: float = 0.7,
    stream: bool = False,
) -> str:
    token = os.environ.get("ON_DEMAND_API_ACCESS_TOKEN", "")
    if not token:
        raise ValueError("Missing env var ON_DEMAND_API_ACCESS_TOKEN")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    payload = {
        "input": {
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }
    }

    # NOTE: keep stream=False for now 
    # we can implement iter_lines parsing later.
    r = requests.post(THETA_URL, headers=headers, json=payload, timeout=120)
    if DEBUG_THETA:
        print("\n[THETA STATUS]", r.status_code)
        print("[THETA RESPONSE TEXT]", r.text[:2000])
    r.raise_for_status()

    if stream:
        return "[Streaming enabled — implement streaming response parsing if needed.]"

    data = r.json()
        # Theta EdgeCloud response shape:
    # {"status":"success","body":{"infer_requests":[{"output":{"message":"..."}}]}}
    if isinstance(data, dict):
        body = data.get("body")
        if isinstance(body, dict):
            reqs = body.get("infer_requests")
            if isinstance(reqs, list) and reqs:
                out = reqs[0].get("output")
                if isinstance(out, dict) and "message" in out:
                    return str(out["message"]).strip()

    # Response shapes vary by provider; handle common patterns safely.
    if isinstance(data, dict):
        # 1) {"output": "..."}
        if "output" in data and isinstance(data["output"], str):
            return data["output"].strip()

        # 2) {"choices":[{"message":{"content":"..."}}]}
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            c0 = data["choices"][0]
            if isinstance(c0, dict):
                if "message" in c0 and isinstance(c0["message"], dict):
                    return (c0["message"].get("content") or "").strip()
                if "text" in c0:
                    return str(c0["text"]).strip()

        # 3) {"result":{"text":"..."}}
        if "result" in data and isinstance(data["result"], dict) and "text" in data["result"]:
            return str(data["result"]["text"]).strip()

        # 4) Some providers return {"message":{"content":"..."}}
        if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
            return str(data["message"]["content"]).strip()

    return str(data)


# Main: interactive chatbot
def main():
    store = load_store()
    print(f"Embedding model (from pkl): {store.model_name}")
    print(f"Chunks: {len(store.chunks)} | FAISS ntotal: {store.index.ntotal}")

    embedder = SentenceTransformer(store.model_name)

    history: List[Dict[str, str]] = []
    pending_question: Optional[str] = None

    print("\nReady. Ask academic questions. Type 'quit' to exit.")

    while True:
        q = input("\nQ> ").strip()
        if q.lower() in {"quit", "exit"}:
            break

        # If previously asked a clarifying question, merge the user's answer into the original question
        if pending_question:
            if len(q.split()) <= 6:  # short answer = clarification keyword
                q = f"{pending_question} {q}".strip()
            else:  # long answer = user restated the question themselves
                q = q  # use their new phrasing as-is
            pending_question = None

        fu = needs_followup(q)
        if fu:
            print("\n" + fu)
            pending_question = q  # store the original question so the next user input refines it
            continue
        hits = search(store, embedder, q, k=8, overfetch=200)
       # print("\n[DEBUG] Top chunks retrieved:")
        #for h in hits[:5]:
           # meta = h.get("meta") or {}
           # print(f"  term={meta.get('term')} score={h['score']:.3f} | FULL TEXT: {h['text']}")
        
        if any(x in q.lower() for x in ["how long", "length of", "duration of"]):
            term_key = None
            if "spring" in q.lower(): term_key = "spring"
            elif "fall" in q.lower(): term_key = "fall"
            elif "summer 1" in q.lower(): term_key = "summer 1"
            elif "summer 2" in q.lower(): term_key = "summer 2"
    
            if term_key:
                direct = [
                    {"text": c.text, "meta": c.meta, "score": 0.9}
                    for c in store.chunks
                    if term_key in (c.meta or {}).get("term", "").lower()
                    and any(x in c.text.lower() for x in ["last day of", "courses end", "classes end"])
                ]   
                seen = {h["text"][:80] for h in hits}
                hits = [d for d in direct if d["text"][:80] not in seen] + hits

        if any(x in q.lower() for x in ["contact", "reach", "email", "phone"]) and \
            any(x in q.lower() for x in ["institute", "center", "office", "department", "college"]):
            # Prefer general office/institute contact chunks over individual staff entries
            direct = [
                {"text": c.text, "meta": c.meta, "score": 0.9}
                for c in store.chunks
                if (
                    sum(word in c.text.lower() for word in q.lower().split() if len(word) > 4) >= 2
                    and any(x in c.text.lower() for x in ["@", "email", "phone"])
                )
                and (c.meta or {}).get("chunk_type") != "directory_people"
            ][:4]

            # Only fall back to individual staff if no office-level contact found
            if not direct:
                direct = [
                    {"text": c.text, "meta": c.meta, "score": 0.7}
                    for c in store.chunks
                    if (
                        sum(word in c.text.lower() for word in q.lower().split() if len(word) > 4) >= 2
                        and any(x in c.text.lower() for x in ["@", "email", "phone"])
                    )
                    and (c.meta or {}).get("chunk_type") == "directory_people"
                ][:4]

            seen = {h["text"][:80] for h in hits}
            hits = [d for d in direct if d["text"][:80] not in seen] + hits

        if any(x in q.lower() for x in ["where is", "location of", "located", "address"]) and \
            any(x in q.lower() for x in ["registrar", "bursar", "financial aid", "dean", "advising"]):
            office_terms = ["registrar", "bursar", "financial aid", "dean", "advising"]
            target = next((term for term in office_terms if term in q.lower()), None)
            direct = []

            for c in store.chunks:
                text_lower = c.text.lower()
                if (
                    target
                    and target in text_lower
                    and any(x in text_lower for x in ["suite", "floor", "room", "tower", "hall", "located"])
                ):
                    for line in c.text.splitlines():
                        line_lower = line.lower()
                        if target in line_lower and any(x in line_lower for x in ["suite", "floor", "room", "tower", "hall", "located"]):
                            direct.append({"text": line, "meta": c.meta, "score": 0.95})

            direct = direct[:4]
            seen = {h["text"][:80] for h in hits}
            hits = [d for d in direct if d["text"][:80] not in seen] + hits
        
        if any(x in q.lower() for x in ["walk in", "ceremony", "commencement"]) and \
           "spring" in q.lower():
            direct = [
                {"text": c.text, "meta": c.meta, "score": 0.95}
                for c in store.chunks
                if "spring" in (c.meta or {}).get("term", "").lower()
                and any(x in c.text.lower() for x in ["degree conferral", "commencement", "graduation"])
            ]
            seen = {h["text"][:80] for h in hits}
            hits = [d for d in direct if d["text"][:80] not in seen] + hits

        # Fallback: if filters wiped everything, retry with fewer filters
        if not hits:
            hits = search(store, embedder, q, k=5, overfetch=50)
        if not hits:
            print("\nI couldn't find relevant information for that question. Could you rephrase or be more specific?")
            continue
        context = format_context(hits)

        system = (
        "You are an academic assistant for Illinois Institute of Technology.\n"
        "Use ONLY the provided context.\n"
        "Do NOT invent student attributes (e.g., non-degree seeking, completed credits, residency) unless the user explicitly states them.\n"
        "Do NOT restate the user's situation as a story; answer the question directly.\n"
        "ONLY answer what was asked — do NOT volunteer F-1, graduation, or contact sections unless the question asks about them.\n"

        "SEMESTER START DATE — CRITICAL:\n"
        "- 'Courses Begin' is the START date for ALL terms (Spring, Fall, Summer 1, Summer 2).\n"
        "- 'Add/Drop Deadline', 'Memorial Day', 'Juneteenth', 'Independence Day' are NOT the start date.\n"
        "- Example: 'Courses Begin: 5/18' / 'Memorial Day: 5/25' → Summer 1 starts May 18.\n"

        "DROP vs WITHDRAW — CRITICAL:\n"
        "- DROPPING (before add/drop deadline): NOT on transcript, tuition refunded.\n"
        "- WITHDRAWING (after add/drop deadline): 'W' grade on transcript, NO tuition refund.\n"
        "- NEVER say a dropped course gives an NA grade. NA = non-attendance without formal action only.\n"

        "FULL-TIME STATUS — CRITICAL ARITHMETIC:\n"
        "- UG full-time = 12+ credits. Grad full-time = 9+ credits.\n"
        "- When student says 'I have X credits, drop Y': calculate X - Y FIRST. Answer about status AFTER the drop.\n"
        "  Example: 'I have 12 credits, drop 3' → 12 - 3 = 9 → 9 < 12 → NOT full-time after drop.\n"
        "- ALWAYS show the arithmetic. NEVER state current status as the answer to a drop scenario.\n"

        "TUITION AFTER ADD/DROP DEADLINE:\n"
        "- Dropping after deadline = NO refund AND W grade on transcript. State BOTH facts.\n"

        "UG TAKING GRAD COURSE:\n"
        "- An enrolled UG student needs approval from course instructor AND their advisor. NOT conditional admission.\n"

        "- Directory chunks contain multiple people in 'Name - Title - email' format.\n"
        "- When answering 'who is X', find the line where the title matches and return the NAME from that same line.\n"
        "- Never return a chunk ID as a name. The name is always before the ' - ' separator.\n"

        "F-1 VISA — ONLY if question mentions F-1, visa, SEVIS, or international student:\n"
        "- Below full-time after drop: (a) not full-time + show math, (b) F-1/SEVIS at risk, (c) MUST contact DSO BEFORE dropping.\n"

       

        "CONTACT INFO — ONLY if asked who to contact:\n"
        "- Registration: registrar@illinoistech.edu or 312.567.3100\n"
        "- Tuition/billing: student-accounting@illinoistech.edu\n"
        "- Financial aid: financial-aid@iit.edu\n"

        "For date/event questions: prefer chunks with explicit dates. State the date directly.\n"
        "Format all dates as Month Day, Year (e.g. January 12, 2026). Never use YYYY-MM-DD.\n"
        "CRITICAL: Convert ANY YYYY-MM-DD date in context to Month Day, Year and use it — never say not found.\n"
        "CRITICAL GPA: W (withdrawal) grade does NOT affect GPA. NA grade does NOT affect GPA. Never say they do.\n"
        "CRITICAL FACTS: $1,851/credit grad, $25,824/sem UG flat rate, 100% refundable before add/drop, 9 credits max transfer for masters.\n"
        "CRITICAL F-1: If F-1/visa AND dropping below full-time: MUST warn about SEVIS risk and contact DSO before dropping.\n"
        "Answer the user's question directly first.\n"
        "After answering, you MAY offer ONE optional follow-up ONLY if it adds genuinely new actionable info.\n"
        "Be concise, accurate, and practical.\n"
        "If the user provides numeric credit values, use those exact numbers and do correct arithmetic.\n"
        "If the question involves full-time status, compare credits to the minimum and state the correct status.\n"
        "Do NOT include conversational phrases (e.g., 'I'd be happy to help'). Start directly with the policy answer.\n"
        "Never say 'based on the provided context' or 'according to the context' — just answer directly.\n"
        "Never cite chunk numbers — cite source URLs instead.\n"
        )

        user_prompt = (
            f"USER QUESTION:\n{q}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "RULES:\n"
            "- Answer in 2–5 sentences.\n"
            "- If the user asks multiple things (e.g., status AND tuition), you MUST answer each part explicitly.\n"
            "- For each part that is not explicitly supported by the context, write exactly: 'Not found in the provided context.'\n"
            "- Do NOT infer 'no impact' from missing information.\n"
            "- When using a fact from the context, cite the SOURCE URL directly in parentheses.\n"
            "- Do NOT use numbered citations.\n"
            "- Only ask a follow-up question if the question is ambiguous OR cannot be answered from the context.\n"
        )


        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_prompt}]
        reply = theta_chat(messages, max_tokens=400, temperature=0.3, top_p=0.7, stream=False)


        print("\n" + reply)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": reply})
        history = history[-12:]  # keep last 6 exchanges




"""
async def run_traffic_cop(question: str, session_id: Optional[str] = None) -> dict:
    
    FastAPI-compatible async entry point for the Traffic Cop pipeline.
    Uses shared FAISS loader and memory system.
    All retrieval, reranking, and clarification logic is the original code above.
    
    import asyncio
    import time as _time
    import httpx
    from core.faiss_loader import get_chunks, get_index, get_model
    from core.memory import get_context_string, save_turn, get_user_context
    from core.config import settings

    start = _time.time()

    # Load memory
    memory_context = ""
    user_context = {}
    if session_id:
        memory_context = get_context_string(session_id)
        user_context = get_user_context(session_id)

    # Check clarification (non-blocking — returned as suggestion in response)
    clarification = needs_followup(question)

    # Use shared FAISS loader instead of standalone store
    _index = get_index()
    _chunks_raw = get_chunks()
    _model = get_model()

    # Wrap shared chunks into Store-compatible format for existing search()
    _coerced = [_coerce_chunk(c) for c in _chunks_raw]
    _store = Store(index=_index, chunks=_coerced, model_name="all-mpnet-base-v2")

    # Use existing search() function unchanged
    # Expand query for specific topics before FAISS search
    _ql = question.lower()
    _search_q = question
    if "summer 1" in _ql or "summer i " in _ql:
        _search_q = question + " summer 1 courses begin may 18"
    elif any(x in _ql for x in ["graduation", "graduating", "commencement", "walk in", "ceremony"]):
        _search_q = question + " graduation degree conferral application deadline degree conferral applications due"
    elif any(x in _ql for x in ["withdraw", "drop"]) and any(x in _ql for x in ["difference", "vs", "between"]):
        _search_q = question + " withdrawal drop W grade transcript add/drop deadline"
    elif any(x in _ql for x in ["registration open", "when does registration", "fall 2026"]) and \
         any(x in _ql for x in ["contact", "issues", "help"]):
        _search_q = question + " registrar contact email registrar@illinoistech.edu"

    hits = search(_store, _model, _search_q, k=8, overfetch=200)

    # Inject relevant chunks directly to overcome 72% directory_people dominance
    _is_tuition_q2 = any(x in _ql for x in ["tuition", "per credit", "how much is tuition", "refund", "flat rate"])
    _is_transfer_q2 = any(x in _ql for x in ["maximum transfer", "max transfer", "transfer credit"])
    _is_hold_q2 = any(x in _ql for x in ["financial hold", "hold on my account", "cannot register"])
    _is_finals_q2 = any(x in _ql for x in ["final exam", "finals", "when are finals"])

    if _is_tuition_q2:
        _direct = [{"text": c.text, "meta": c.meta, "score": 0.5}
                   for c in _coerced
                   if (c.meta or {}).get("chunk_type") == "tuition"
                   and any(kw in c.text for kw in ["1851","1,851","25824","25,824","1612","1,612","100%","refundable"])][:6]
        seen = {h.get("text","")[:80] for h in hits}
        hits += [d for d in _direct if d["text"][:80] not in seen]

    if _is_transfer_q2:
        _direct = [{"text": c.text, "meta": c.meta, "score": 0.5}
                   for c in _coerced
                   if (c.meta or {}).get("chunk_type") == "policy"
                   and "nine" in c.text and "transfer" in c.text.lower()][:4]
        seen = {h.get("text","")[:80] for h in hits}
        hits += [d for d in _direct if d["text"][:80] not in seen]

    if _is_hold_q2:
        _direct = [{"text": c.text, "meta": c.meta, "score": 0.5}
                   for c in _coerced
                   if (c.meta or {}).get("chunk_type") == "registration"
                   and "3794" in c.text][:4]
        seen = {h.get("text","")[:80] for h in hits}
        hits += [d for d in _direct if d["text"][:80] not in seen]

    if _is_finals_q2:
        _direct = [{"text": c.text, "meta": c.meta, "score": 0.5}
                   for c in _coerced
                   if (c.meta or {}).get("chunk_type") == "calendar"
                   and ("final exam" in c.text.lower() or "may 4" in c.text.lower()
                        or "may 9" in c.text.lower() or "2026-05-04" in c.text)][:6]
        seen = {h.get("text","")[:80] for h in hits}
        hits += [d for d in _direct if d["text"][:80] not in seen]

    if not hits:
        hits = search(_store, _model, question, k=5, overfetch=50)

    # Augment with contact/directory chunks when question asks who to contact
    _contact_phrases = [
        "who do i contact", "who should i contact", "who to contact",
        "contact if", "contact for", "who do i call", "who can i call",
        "have issues", "have a problem", "have questions", "need help with",
        "who handles", "who is responsible", "if i have issues",
    ]
    if any(p in _ql for p in _contact_phrases):
        contact_hits = search(_store, _model,
                              "registrar contact email phone registrar@illinoistech.edu " + question,
                              k=4, overfetch=50)
        seen_ids = {id(h) for h in hits}
        hits = hits + [h for h in contact_hits if id(h) not in seen_ids]

    # Extract source URLs
    seen_urls: set = set()
    source_urls = []
    for h in hits:
        meta = h.get("meta") or {}
        url = meta.get("url") or meta.get("source_url") or ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            source_urls.append(url)

    # Build context using existing format_context()
    context = format_context(hits) if hits else "No relevant context found."

    # Build memory section
    memory_section = ""
    if memory_context:
        memory_section = f"\nConversation Memory:\n{memory_context}\n"

    # Use same system prompt and user_prompt as main()
    system = (
        "You are an academic assistant for Illinois Institute of Technology.\n"
        "Use ONLY the provided context.\n"
        "Do NOT invent student attributes (e.g., non-degree seeking, completed credits, residency) unless the user explicitly states them.\n"
        "ONLY answer what was asked — do NOT volunteer F-1, graduation, or contact sections unless the question asks about them.\n"

        "SEMESTER START DATE — CRITICAL:\n"
        "- 'Courses Begin' is the START date for ALL terms (Spring, Fall, Summer 1, Summer 2).\n"
        "- 'Add/Drop Deadline', 'Memorial Day', 'Juneteenth', 'Independence Day' are NOT the start date.\n"
        "- Example: 'Courses Begin: 5/18' / 'Memorial Day: 5/25' → Summer 1 starts May 18.\n"

        "DROP vs WITHDRAW — CRITICAL:\n"
        "- DROPPING (before add/drop deadline): NOT on transcript, tuition refunded.\n"
        "- WITHDRAWING (after add/drop deadline): 'W' grade on transcript, NO tuition refund.\n"
        "- NEVER say a dropped course gives an NA grade. NA = non-attendance without formal action only.\n"

        "FULL-TIME STATUS — CRITICAL ARITHMETIC:\n"
        "- UG full-time = 12+ credits. Grad full-time = 9+ credits.\n"
        "- When student says 'I have X credits, drop Y': calculate X - Y FIRST. Answer about status AFTER the drop.\n"
        "  Example: 'I have 12 credits, drop 3' → 12 - 3 = 9 → 9 < 12 → NOT full-time after drop.\n"
        "- ALWAYS show the arithmetic. NEVER state current status as the answer to a drop scenario.\n"

        "TUITION AFTER ADD/DROP DEADLINE:\n"
        "- Dropping after deadline = NO refund AND W grade on transcript. State BOTH facts.\n"

        "UG TAKING GRAD COURSE:\n"
        "- An enrolled UG student needs approval from course instructor AND their advisor. NOT conditional admission.\n"

        "F-1 VISA — ONLY if question mentions F-1, visa, SEVIS, or international student:\n"
        "- Below full-time after drop: (a) not full-time + show math, (b) F-1/SEVIS at risk, (c) MUST contact DSO BEFORE dropping.\n"

        "GRADUATION CEREMONY — ONLY if question is about graduation/ceremony:\n"
        "- 'Apply Summer, walk Spring': deadline = June 5, 2026 (Summer Degree Conferral Applications Due); ceremony = May 16-17.\n"

        "CONTACT INFO — ONLY if asked who to contact:\n"
        "- Registration: registrar@illinoistech.edu or 312.567.3100\n"
        "- Tuition/billing: student-accounting@illinoistech.edu\n"
        "- Financial aid: financial-aid@iit.edu\n"

        "For date/event questions: prefer chunks with explicit dates. State the date directly.\n"
        "Format all dates as Month Day, Year (e.g. January 12, 2026). Never use YYYY-MM-DD.\n"
        "CRITICAL: Convert ANY YYYY-MM-DD date in context to Month Day, Year and use it — never say not found.\n"
        "CRITICAL GPA: W (withdrawal) grade does NOT affect GPA. NA grade does NOT affect GPA. Never say they do.\n"
        "CRITICAL FACTS: $1,851/credit grad, $25,824/sem UG flat rate, 100% refundable before add/drop, 9 credits max transfer for masters.\n"
        "CRITICAL F-1: If F-1/visa AND dropping below full-time: MUST warn about SEVIS risk and contact DSO before dropping.\n"
        "Answer the user's question directly first.\n"
        "After answering, you MAY offer ONE optional follow-up ONLY if it adds genuinely new actionable info.\n"
        "Be concise, accurate, and practical.\n"
        "If the user provides numeric credit values, use those exact numbers and do correct arithmetic.\n"
        "If the question involves full-time status, compare credits to the minimum and state the correct status.\n"
        "Do NOT include conversational phrases (e.g., 'I'd be happy to help'). Start directly with the policy answer.\n"
        "Never say 'based on the provided context' or 'according to the context' — just answer directly.\n"
        "Never cite chunk numbers — cite source URLs instead.\n"
    )

    user_prompt = (
        f"USER QUESTION:\n{question}\n\n"
        f"{memory_section}"
        f"CONTEXT:\n{context}\n\n"
        "RULES:\n"
        "- Answer in 2-5 sentences.\n"
        "- Format all dates as Month Day, Year (never YYYY-MM-DD).\n"
        "- If the user asks multiple things, you MUST answer each part explicitly.\n"
        "- For each part that is not explicitly supported by the context, write exactly: 'Not found in the provided context.'\n"
        "- Do NOT infer 'no impact' from missing information.\n"
        "- When using a fact from the context, cite the SOURCE URL directly in parentheses.\n"
        "- Do NOT use numbered citations.\n"
        "- Only ask a follow-up question if the question is ambiguous OR cannot be answered from the context.\n"
    )

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]

    # Call Theta async
    answer = ""
    headers = {
        "Authorization": f"Bearer {settings.THETA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.3,
            "top_p": 0.7,
            "stream": False,
        }
    }
    url = f"{settings.THETA_BASE_URL}/{settings.THETA_MODEL}/completions"
    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(3):
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code == 409:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            try:
                answer = data["body"]["infer_requests"][0]["output"]["message"].strip()
                break
            except (KeyError, IndexError, TypeError):
                pass
            try:
                if "choices" in data:
                    answer = data["choices"][0]["message"]["content"].strip()
                    break
                if "output" in data and isinstance(data["output"], str):
                    answer = data["output"].strip()
                    break
            except (KeyError, IndexError, TypeError):
                pass

    # Save to memory
    if session_id and answer:
        save_turn(session_id, question, answer)

    result = {
        "pipeline": "traffic_cop",
        "answer": answer,
        "source_urls": source_urls,
        "chunks_used": len(hits),
        "response_time_ms": round((_time.time() - start) * 1000),
    }
    if clarification:
        result["clarification_suggestion"] = clarification

    return result
"""

if __name__ == "__main__":
    main()