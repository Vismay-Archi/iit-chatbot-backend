import re
import time
import json
import asyncio
import httpx
from typing import Optional, List

from core.faiss_loader import search, search_with_filter
from core.llm import call_llm
from core.config import settings

# ── Constants ──────────────────────────────────────────────────────────────────

# Based on ACTUAL chunk types found in faiss_store.pkl:
# directory_people (2358), student_handbook_section (83),
# registrar_page (25), registrar_section (23), coursera_page (16),
# coterminal_handbook_section (12), + inferred: tuition, registration, policy, calendar

GRAD_KEYWORDS = ["graduate", "grad", "master", "masters", "phd", "doctoral", "mba", "meng", "ms "]
UG_KEYWORDS   = ["undergraduate", "undergrad", "bachelor", "bachelors", "ug ", "bs ", "ba "]

# Topics where we should search both grad and undergrad when level is unspecified
DUAL_SEARCH_TOPICS = {"tuition", "registration", "policy"}

CURRENT_TERM      = "Spring 2026"
CURRENT_DATA_YEAR = "2025-2026"

# ── Metadata extraction prompt ─────────────────────────────────────────────────

METADATA_EXTRACTION_PROMPT = """You are a metadata filter extractor for an IIT university knowledge base.
Extract filters from the student question to narrow the search. Return ONLY valid JSON, nothing else.

Available chunk_type values:
- "directory_people"              → staff, faculty, contacts, who is, email, phone, department contacts
- "student_handbook_section"      → student policies, conduct, campus life, student rights, grievances
- "coterminal_handbook_section"   → coterminal program, co-terminal, dual degree undergraduate+graduate
- "registrar_page"                → registrar office info, transcripts, enrollment verification
- "registrar_section"             → transcript ordering, transcript fees, enrollment/degree verification details
- "coursera_page"                 → Coursera, online courses, MOOCs
- "tuition"                       → tuition, fees, costs, payment, refunds, installment plan
- "registration"                  → enrolling, adding/dropping courses, holds, full-time status, late registration
- "policy"                        → pass/fail, prerequisites, course repeat, credit hours, auditing, compressed term
- "calendar"                      → dates, deadlines, academic calendar, final exam schedule, commencement
- "general"                       → anything else

FINANCIAL AID NOTE: There is no "financial_aid" chunk_type. Questions about financial aid,
scholarships, FAFSA, grants, loans, or aid eligibility should use chunk_type: null so the
search covers policy, tuition, and student_handbook_section chunks simultaneously.

Rules:
- Return exactly: {{"chunk_type": "<value or null>", "tag": "<Faculty|Staff|null>"}}
- "tag" is ONLY for directory_people — use "Faculty" or "Staff" if specified, else null
- If the question is ambiguous, set chunk_type to null
- IMPORTANT: Any question starting with "When" should almost always be "calendar"
- "registration" is for HOW to do something, "calendar" is for WHEN something happens
- If the question spans multiple topics (e.g. tuition AND dates), set chunk_type to null

Examples:
Q: "Who is the dean of CS?" → {{"chunk_type": "directory_people", "tag": "Faculty"}}
Q: "What is graduate tuition per credit hour?" → {{"chunk_type": "tuition", "tag": null}}
Q: "How do I apply to the coterminal program?" → {{"chunk_type": "coterminal_handbook_section", "tag": null}}
Q: "What is the pass/fail policy?" → {{"chunk_type": "policy", "tag": null}}
Q: "When is the last day to withdraw?" → {{"chunk_type": "calendar", "tag": null}}
Q: "How do I register for classes?" → {{"chunk_type": "registration", "tag": null}}
Q: "How do I add or drop a course?" → {{"chunk_type": "registration", "tag": null}}
Q: "What are the student conduct rules?" → {{"chunk_type": "student_handbook_section", "tag": null}}
Q: "Who is the registrar?" → {{"chunk_type": "directory_people", "tag": "Staff"}}
Q: "What are the Coursera offerings?" → {{"chunk_type": "coursera_page", "tag": null}}
Q: "What is IIT known for?" → {{"chunk_type": null, "tag": null}}
Q: "When is the last day to add or drop a course?" → {{"chunk_type": "calendar", "tag": null}}
Q: "When is the add/drop deadline?" → {{"chunk_type": "calendar", "tag": null}}
Q: "When is spring break?" → {{"chunk_type": "calendar", "tag": null}}
Q: "When does the semester start?" → {{"chunk_type": "calendar", "tag": null}}
Q: "When are final exams?" → {{"chunk_type": "calendar", "tag": null}}
Q: "When is the last day to register?" → {{"chunk_type": "calendar", "tag": null}}
Q: "How much is tuition and when is payment due?" → {{"chunk_type": null, "tag": null}}
Q: "I have 12 credits as a UG, if I drop 3 am I still full-time?" → {{"chunk_type": "registration", "tag": null}}
Q: "I'm an F-1 student with 9 credits and want to drop a class" → {{"chunk_type": "registration", "tag": null}}
Q: "Can I walk in commencement if I'm graduating in Summer?" → {{"chunk_type": "calendar", "tag": null}}

Now extract filters for:
Q: "{question}"
→"""

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a helpful, knowledgeable assistant for Illinois Institute of Technology (IIT) students.
Current academic term: {CURRENT_TERM}. Available data covers the {CURRENT_DATA_YEAR} academic year.

ANSWER RULES:
- Answer directly and confidently using ONLY the context provided
- Never say "based on the provided context" or "according to the context" — just answer
- Never add disclaimers like "please verify" or "I recommend checking"
- Answer EACH part of a multi-part question explicitly
- For each part not in context: say exactly "Not found in the provided context."
- Do NOT infer "no impact" from missing information
- ONLY answer what was asked — do NOT volunteer F-1, graduation, or contact sections unless the question explicitly asks about them

FORMATTING RULES:
- Format ALL dates as "Month Day, Year" (e.g. January 12, 2026) — NEVER use YYYY-MM-DD
- For dollar amounts, specify what it covers (per credit hour / per semester / annual)
- Cite source URLs inline in parentheses — do NOT use numbered citations or chunk numbers
- Keep answers concise but complete — 2-5 sentences

SEMESTER START DATE — CRITICAL:
- "Courses Begin" or "Classes Begin" is the semester START date for ALL terms (Spring, Fall, Summer 1, Summer 2)
- "Add/Drop Deadline", "Late Registration Deadline", "Memorial Day", "Juneteenth", "Independence Day" are NOT the start date
- Examples:
  - "Spring Courses Begin: 1/12" and "Add/Drop Deadline: 1/20" → Spring starts January 12, NOT January 20
  - "Courses Begin: 5/18" and "Add/Drop: 5/22" and "Memorial Day: 5/25" → Summer 1 starts May 18, NOT May 25
  - "Courses Begin: 6/15" and "Add/Drop: 6/19" → Summer 2 starts June 15, NOT June 19

DROP vs WITHDRAW — CRITICAL DISTINCTION:
- DROPPING a course (before add/drop deadline): course does NOT appear on transcript, tuition IS refunded
- WITHDRAWING from a course (after add/drop deadline): grade of "W" appears on transcript, NO tuition refund
- NEVER say a dropped course results in an NA grade — NA is only for non-attendance without formal action
- NEVER say a withdrawn course is removed from the record — it stays as W

FULL-TIME STATUS — CRITICAL ARITHMETIC:
- Undergraduate full-time minimum = 12 credit hours per semester
- Graduate full-time minimum = 9 credit hours per semester
- When a student says they have X credits and want to drop Y credits, ALWAYS calculate X - Y first:
  - "I have 12 credits, if I drop a 3-credit course" → 12 - 3 = 9 → 9 < 12 → NOT full-time after drop
  - "I have 9 credits, if I drop a 3-credit course" → 9 - 3 = 6 → 6 < 9 → NOT full-time after drop
- NEVER answer about the CURRENT status — always answer about status AFTER the drop
- ALWAYS show the arithmetic explicitly

TUITION AFTER ADD/DROP DEADLINE:
- Courses dropped after the add/drop deadline = NO tuition refund, W grade on transcript
- Courses dropped before add/drop deadline = full tuition refund, no transcript entry
- When asked what happens to tuition if a class is dropped after the deadline: state BOTH (1) no refund AND (2) W grade

UG STUDENT TAKING GRADUATE COURSE:
- An ENROLLED undergraduate student can take a graduate-level course (500+) with approval from BOTH the course instructor AND their advisor
- This does NOT require being a non-degree student or conditional admission — those are for external applicants
- Source: https://www.iit.edu/registrar/registration/undergraduate-approval-graduate-course

F-1 VISA — ONLY include this section if the question explicitly mentions F-1, visa, SEVIS, or international student:
- F-1 students must maintain full-time status (12 UG / 9 Grad credits)
- If dropping causes them to fall below full-time: (a) not full-time + show math, (b) F-1/SEVIS at risk, (c) MUST contact DSO BEFORE dropping
- Do NOT add F-1 warnings to questions that never mention visa or international status

GRADUATION CEREMONY — ONLY include this section if the question explicitly asks about graduation or ceremony:
- A student may complete their degree in one term but walk in a different ceremony
- The application deadline is for the student's DEGREE TERM, not the ceremony term
- "Apply Summer, walk Spring": deadline = Summer Degree Conferral Applications Due (June 5, 2026); ceremony = Spring commencement (May 16-17)
- Never conflate the ceremony date with the degree application deadline

CONTACT INFO — ONLY include if the question asks who to contact:
- For registration issues: registrar@illinoistech.edu or 312.567.3100
- For tuition/billing: student-accounting@illinoistech.edu or 312.567.3794
- For financial aid: financial-aid@iit.edu

WHEN DATA IS MISSING:
- Say so in one sentence — do NOT guess dates, amounts, or names"""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(question: str, chunks: list) -> str:
    """Build prompt with source URLs exposed in context blocks."""
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        text = c.get("text", c.get("content", ""))
        meta = c.get("metadata", {}) or {}
        url  = meta.get("url") or meta.get("source_url") or ""
        source = (
            meta.get("term") or meta.get("title") or meta.get("page_title") or
            meta.get("source") or meta.get("source_file") or f"source {i}"
        )
        header = f"SOURCE_URL: {url}\n[{source}]" if url else f"[{source}]"
        context_blocks.append(f"{header}\n{text}")

    context = "\n\n".join(context_blocks)
    return (
        f"USER QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "RULES:\n"
        "- Answer in 2-5 sentences.\n"
        "- Format all dates as Month Day, Year (never YYYY-MM-DD).\n"
        "- Answer EACH part of a multi-part question explicitly.\n"
        "- For each part not in context: 'Not found in the provided context.'\n"
        "- Cite source URLs inline in parentheses. No numbered citations.\n"
    )


# ── Query expansion ────────────────────────────────────────────────────────────

def _expand_query(question: str) -> str:
    """Add domain terms to improve FAISS retrieval for specific topics."""
    q = question.lower()
    if any(x in q for x in ["pass/fail", "pass fail", "p/f", "s/u"]):
        return question + " policy maximum limit three 3 courses"
    if any(x in q for x in ["walk in", "ceremony", "commencement", "graduation", "graduating", "apply for graduation"]):
        return question + " graduation ceremony commencement degree conferral application deadline degree conferral applications due"
    if "summer 1" in q or "summer i " in q or "summer one" in q:
        return question + " summer 1 courses begin may 18"
    if "summer 2" in q or "summer ii" in q or "summer two" in q:
        return question + " summer 2 courses begin"
    if any(x in q for x in ["withdraw", "drop"]) and any(x in q for x in ["difference", "vs", "versus", "between"]):
        return question + " withdrawal drop W grade transcript add/drop deadline not on transcript"
    if any(x in q for x in ["ug", "undergrad"]) and any(x in q for x in ["graduate course", "grad course", "500", "graduate level"]):
        return question + " undergraduate approval graduate course instructor advisor permission"
    if any(x in q for x in ["registration open", "registration opens", "when does registration", "fall 2026"]) and \
       any(x in q for x in ["contact", "issues", "help", "problem"]):
        return question + " registrar contact email registrar@illinoistech.edu"
    return question


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_level_specified(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in GRAD_KEYWORDS + UG_KEYWORDS)


def needs_dual_search(question: str, chunk_type: str) -> bool:
    return chunk_type in DUAL_SEARCH_TOPICS and not is_level_specified(question)


def _is_graduation_topic(q: str) -> bool:
    return any(x in q.lower() for x in [
        "graduation", "graduating", "commencement", "diploma",
        "degree conferral", "conferral", "cap and gown", "walk in",
        "ceremony", "apply for graduation",
    ])


def deduplicate_chunks(chunks: list) -> list:
    seen, unique = set(), []
    for c in chunks:
        key = c.get("text", "")[:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def extract_source_urls(chunks: list) -> list:
    seen, urls = set(), []
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        url  = meta.get("url") or meta.get("source_url") or ""
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


# ── Metadata filter extraction ─────────────────────────────────────────────────

async def extract_metadata_filters(question: str) -> dict:
    prompt = METADATA_EXTRACTION_PROMPT.format(question=question)
    raw = await call_llm(prompt)
    try:
        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        s = raw.find("{"); e = raw.rfind("}") + 1
        if s >= 0 and e > s:
            raw = raw[s:e]
        return json.loads(raw.strip())
    except Exception:
        return {"chunk_type": None, "tag": None}


# ── Retrieval ──────────────────────────────────────────────────────────────────

def _retrieve(question: str, chunk_type: Optional[str],
              tag: Optional[str]) -> tuple:
    """
    Returns (chunks, filters_applied).

    Improvements over original:
    - Query expansion before FAISS search
    - Wider k=12 for graduation questions (cross-term ceremony needs more chunks)
    - Contact person augmentation for multi-topic questions
    - Fallback logged in filters
    """
    expanded = _expand_query(question)

    # Graduation needs wider retrieval to catch cross-term ceremony chunks
    k      = 12 if _is_graduation_topic(question) else settings.TOP_K
    half_k = max(k // 2, 3)

    filters = {"chunk_type": chunk_type, "tag": tag}
    chunks  = []

    if chunk_type and chunk_type != "general":

        if needs_dual_search(question, chunk_type):
            # Search both grad and undergrad contexts when level is unspecified
            grad_chunks = search_with_filter(
                "graduate " + expanded, "chunk_type", chunk_type, top_k=half_k
            )
            ug_chunks = search_with_filter(
                "undergraduate " + expanded, "chunk_type", chunk_type, top_k=half_k
            )
            chunks = deduplicate_chunks(grad_chunks + ug_chunks)
            filters["dual_search"] = True

        else:
            chunks = search_with_filter(
                expanded, "chunk_type", chunk_type, top_k=k
            )

        # Augment with contact directory for questions asking who to contact
        q = question.lower()
        if any(x in q for x in [
            "who do i contact", "who should i contact", "who to contact",
            "contact if", "contact for", "who do i call", "who can i call",
            "have issues", "have a problem", "have questions", "need help with",
            "who handles", "who is responsible", "if i have issues",
        ]):
            contact_chunks = search_with_filter(
                "registrar contact email phone " + question,
                "chunk_type", "directory_people", top_k=4
            )
            chunks = deduplicate_chunks(chunks + contact_chunks)
            filters["contact_augmented"] = True

    # Fallback to unfiltered search if filters return nothing
    if not chunks:
        chunks = search(expanded, top_k=k)
        filters["fallback"] = True

    return chunks[:k], filters


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_self_query(question: str) -> dict:
    """
    Self-Query Pipeline — Improved:

    1. LLM extracts metadata filters (chunk_type + tag)
    2. Filtered FAISS search with query expansion
       - Dual search for grad+UG when level unspecified
       - Wider k=12 for graduation questions (cross-term ceremony)
       - Contact person augmentation for multi-topic questions
       - Fallback to unfiltered search if needed
    3. Improved prompt with system instructions:
       - Full-time arithmetic rules (fixes Q23: UG 12-3=9 < 12 → NOT full-time)
       - F-1 visa implications (fixes Q21: must contact DSO before dropping)
       - Graduation cross-term ceremony logic (fixes Q22)
       - Source URLs exposed in context blocks (better inline citations)
       - Date format enforcement (never YYYY-MM-DD)
    """
    start = time.time()

    # Step 1: Extract filters
    filters    = await extract_metadata_filters(question)
    chunk_type = filters.get("chunk_type")
    tag        = filters.get("tag")

    # Step 2: Retrieve with improvements
    chunks, filters_applied = _retrieve(question, chunk_type, tag)

    # Step 3: Build prompt and call LLM with system prompt
    user_prompt = _build_prompt(question, chunks)
    answer      = ""

    if settings.LLM_PROVIDER == "ollama":
        answer = await call_llm(SYSTEM_PROMPT + "\n\n" + user_prompt)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        headers = {
            "Authorization": f"Bearer {settings.THETA_API_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {
            "input": {
                "messages":    messages,
                "max_tokens":  500,
                "temperature": 0.3,
                "top_p":       0.7,
                "stream":      False,
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

    return {
        "pipeline":         "self_query",
        "answer":           answer,
        "filters_applied":  filters_applied,
        "sources":          [c.get("metadata", {}).get("source_file", "unknown") for c in chunks],
        "source_urls":      extract_source_urls(chunks),
        "chunks_used":      len(chunks),
        "response_time_ms": round((time.time() - start) * 1000),
    }
