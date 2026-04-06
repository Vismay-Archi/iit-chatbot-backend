"""
IIT Chatbot — Automated Test Suite
Run with: python test_pipeline.py
Server must be running at http://127.0.0.1:8000
"""

import requests
import json
import uuid
import time

BASE = "http://127.0.0.1:8000/api"
SESSION = str(uuid.uuid4())   # fresh session for each run

TESTS = [
    # (category, question, must_contain, must_not_contain)

    # ── Calendar ──────────────────────────────────────────────────────────────
    ("Calendar/Easy",
     "When does Spring 2026 semester start?",
     ["january 12", "jan 12"],
     ["march 23", "not found"]),

    ("Calendar/Easy",
     "When are Spring 2026 final exams?",
     ["may 4", "may 9"],
     ["not found"]),

    ("Calendar/Normal",
     "When is the last day to withdraw from a Spring 2026 course?",
     ["april 17"],
     ["not found"]),

    ("Calendar/Hard - Graduation",
     "When is the Spring 2026 graduation ceremony?",
     ["may 16", "credit union", "commencement"],
     ["not found", "not explicitly"]),

    # ── Tuition ───────────────────────────────────────────────────────────────
    ("Tuition/Easy",
     "What is the tuition per credit hour for graduate students on the Mies campus?",
     ["1,851", "1851"],
     ["51,648", "not found"]),

    ("Tuition/Easy",
     "What is the tuition per credit hour for undergraduate students?",
     ["25,824", "1,612"],
     ["not found"]),

    ("Tuition/Normal",
     "Do I get a refund if I drop a course before the add/drop deadline?",
     ["100%", "refundable"],
     ["not found"]),

    ("Tuition/Hard",
     "If I withdraw from all classes after the fourth week, do I receive a tuition refund?",
     ["no", "not"],
     ["yes, you will receive"]),

    # ── Policy ────────────────────────────────────────────────────────────────
    ("Policy/Easy",
     "How many credits do graduate students need to be full-time?",
     ["9"],
     ["12", "not found"]),

    ("Policy/Easy",
     "How many credits do undergraduate students need to be full-time?",
     ["12"],
     ["not found"]),

    ("Policy/Normal",
     "What is the difference between withdrawing and dropping a course?",
     ["W", "transcript", "add/drop"],
     ["not found"]),

    ("Policy/Normal",
     "If I withdraw from a course, does it affect my GPA?",
     ["not", "no"],
     ["will affect", "does affect"]),

    # ── Transcripts ───────────────────────────────────────────────────────────
    ("Transcripts/Easy",
     "How do I get an official transcript from IIT?",
     ["portal", "parchment"],
     ["not found"]),

    ("Transcripts/Normal",
     "How long does it take to process a transcript request?",
     ["hour", "1 hour"],
     ["not found"]),

    # ── Transfer Credit ───────────────────────────────────────────────────────
    ("Transfer/Easy",
     "What is the maximum number of transfer credits allowed for a master's program?",
     ["9"],
     ["not found"]),

    ("Transfer/Normal",
     "Do transfer credits affect GPA calculations?",
     ["not", "no"],
     ["yes", "will affect"]),

    # ── Contact ───────────────────────────────────────────────────────────────
    ("Contact/Easy",
     "What is the phone number for the Registrar's office?",
     ["312.567.3100", "312-567-3100"],
     ["not found"]),

    ("Contact/Hard",
     "If I have a financial hold and cannot register, who should I contact?",
     ["student accounting", "312.567.3794"],
     ["not found"]),

    # ── Multi-topic ───────────────────────────────────────────────────────────
    ("Multi/Hard",
     "I'm an undergraduate student with 12 credits — if I drop a 3-credit course after the add/drop deadline, am I still full-time?",
     ["9", "not full-time", "no"],
     ["yes, you are full-time"]),

    ("Multi/Hard",
     "I'm an F-1 grad student taking 9 credits in Spring 2026 — am I full-time, and what happens to my visa status if I drop one 3-credit course?",
     ["full-time", "dso", "sevis", "f-1"],
     ["not found"]),

    # ── Scope guards ──────────────────────────────────────────────────────────
    ("Scope/OOS",
     "What is the acceptance rate at IIT?",
     ["iit.edu", "knowledge base", "don't have"],
     ["acceptance rate is", "%"]),

    ("Scope/Trap",
     "What is the Spring 2030 academic calendar?",
     ["2025-2026", "don't have"],
     ["spring 2030 starts", "courses begin"]),

    # ── Clarification (these should ask a question back) ──────────────────────
    ("Clarification",
     "When does the semester start?",
     ["?", "spring", "fall", "summer"],
     ["january", "august", "courses begin"]),

    ("Clarification",
     "How much is tuition?",
     ["?", "undergraduate", "graduate"],
     ["25,824", "1,851"]),
]


def run_test(category, question, must_contain, must_not_contain, session_id):
    try:
        r = requests.post(
            f"{BASE}/query",
            json={"question": question, "method": "hybrid", "session_id": session_id},
            timeout=30
        )
        data = r.json()
        answer = data.get("answer", "").lower()
        winner = data.get("results", {}).get("hybrid", {}).get("winner", "?")
        scores = data.get("results", {}).get("hybrid", {}).get("scores", {})
        urls = data.get("results", {}).get("hybrid", {}).get("source_urls", [])

        hits   = [kw for kw in must_contain     if kw.lower() in answer]
        misses = [kw for kw in must_contain     if kw.lower() not in answer]
        bads   = [kw for kw in must_not_contain if kw.lower() in answer]

        passed = len(misses) == 0 and len(bads) == 0
        return {
            "passed": passed,
            "answer": data.get("answer", "")[:120],
            "winner": winner,
            "scores": scores,
            "urls":   urls[:2],
            "misses": misses,
            "bads":   bads,
        }
    except Exception as e:
        return {"passed": False, "answer": f"ERROR: {e}", "winner": "?", "scores": {}, "urls": [], "misses": [], "bads": []}


def main():
    print("=" * 70)
    print("IIT CHATBOT — AUTOMATED TEST SUITE")
    print("=" * 70)
    print()

    passed = 0
    failed = 0
    results = []

    for category, question, must_contain, must_not_contain in TESTS:
        # Use fresh session per test to avoid clarification state bleeding
        sid = str(uuid.uuid4())
        result = run_test(category, question, must_contain, must_not_contain, sid)
        results.append((category, question, result))
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        if result["passed"]:
            passed += 1
        else:
            failed += 1

        print(f"{status}  [{category}]")
        print(f"       Q: {question[:70]}")
        print(f"       A: {result['answer'][:110]}")
        print(f"       Winner: {result['winner']} | Scores: {result['scores']}")
        if result["misses"]:
            print(f"       ⚠ Missing keywords: {result['misses']}")
        if result["bads"]:
            print(f"       ⚠ Bad keywords found: {result['bads']}")
        print()

        time.sleep(0.5)  # avoid hammering Theta

    print("=" * 70)
    print(f"RESULTS: {passed}/{len(TESTS)} passed  ({passed/len(TESTS)*100:.0f}%)")
    print("=" * 70)

    if failed:
        print("\nFAILED TESTS:")
        for category, question, result in results:
            if not result["passed"]:
                print(f"  ❌ [{category}] {question[:60]}")
                if result["misses"]:
                    print(f"       Missing: {result['misses']}")
                if result["bads"]:
                    print(f"       Bad: {result['bads']}")


if __name__ == "__main__":
    main()
