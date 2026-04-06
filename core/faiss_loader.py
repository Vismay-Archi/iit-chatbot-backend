import faiss
import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from core.config import settings

# Global state — loaded once at startup
_index = None
_chunks = None
_model = None

# URL keyword → category mapping for chunks missing chunk_type
URL_CATEGORY_MAP = {
    # Tuition
    "tuition": "tuition",
    "fees": "tuition",
    "payment": "tuition",
    "credits-and-refunds": "tuition",
    "installment": "tuition",
    "student_accounting": "tuition",
    "student-accounting": "tuition",
    "mies_grad_tuition": "tuition",
    "mies_ug_tuition": "tuition",
    "mies_graduate_tuition": "tuition",
    "mies_undergraduate_tuition": "tuition",
    "mandatory_fees": "tuition",
    "refund": "tuition",

    # Registration
    "hold-information": "registration",
    "hold_info": "registration",
    "registration": "registration",
    "full-time-status": "registration",
    "full_time": "registration",
    "late-registration": "registration",
    "late_registration": "registration",
    "advising": "registration",
    "students-and-alumni": "registration",
    "students_and_alumni": "registration",
    "internet_course": "registration",
    "internet-course": "registration",
    "withdraw": "registration",
    "withdrawal": "registration",

    # Policy
    "course-repeat": "policy",
    "course_repeat": "policy",
    "pass-fail": "policy",
    "pass_fail": "policy",
    "passfail": "policy",
    "prerequisites": "policy",
    "prerequisite": "policy",
    "auditing": "policy",
    "compressed-term": "policy",
    "compressed_term": "policy",
    "contact-and-credit": "policy",
    "contact_and_credit": "policy",
    "transfer-credit": "policy",
    "transfer_credit": "policy",
    "undergrad_approval": "policy",
    "credit_hour": "policy",
    "coterminal_policy": "policy",
    "coterminal-policy": "policy",
    "gaa": "policy",
    "grades_and_transcripts": "policy",
    "grades-and-transcripts": "policy",
    "grade_legend": "policy",
    "grade-legend": "policy",
    "hardship": "policy",
    "withdraw_vs_drop": "policy",
    "withdraw-vs-drop": "policy",
    "course_cancellation": "policy",
    "course-cancellation": "policy",
    "policies_procedures": "policy",
    "registrar_policies": "policy",
    "important_information": "policy",

    # Calendar
    "academic-calendar": "calendar",
    "academic_calendar": "calendar",
    "final-exam-schedule": "calendar",
    "final_exam_schedule": "calendar",
    "commencement": "calendar",
    "event-details": "calendar",
    "event_details": "calendar",

    # Directory / people
    "registrar_people": "directory_people",
    "registrar-people": "directory_people",
    "directory": "directory_people",

    # Transcripts
    "transcripts": "registrar_page",
    "transcript": "registrar_page",

    # Coursera
    "coursera": "coursera_page",
}


def _infer_chunk_type(chunk: dict) -> str:
    """
    For chunks missing chunk_type, infer it from URL or source_file.
    """
    meta = chunk.get("metadata", {}) or {}
    url = meta.get("url", meta.get("source_url", "")).lower()
    source_file = meta.get("source_file", "").lower()

    for keyword, category in URL_CATEGORY_MAP.items():
        if keyword in url or keyword in source_file:
            return category

    # Fallback by title
    title = meta.get("title", meta.get("page_title", "")).lower()
    if any(k in title for k in ["tuition", "fee", "cost", "payment", "refund"]):
        return "tuition"
    if any(k in title for k in ["register", "registration", "enroll", "hold", "withdraw"]):
        return "registration"
    if any(k in title for k in ["policy", "policies", "rule", "pass/fail", "prerequisite", "grade"]):
        return "policy"
    if any(k in title for k in ["calendar", "schedule", "deadline", "exam", "commencement"]):
        return "calendar"

    return "general"


def _normalize_chunks(chunks: list) -> list:
    """
    Normalize chunks from either format:
    - Our format: {chunk_id, source_file, chunk_type, text, metadata}
    - Teammate's pkl format: {text, metadata: {chunk_type, source_file, ...}}

    Also fills in missing chunk_types using URL inference.
    """
    normalized = []
    for c in chunks:
        # Get chunk_type from top-level or metadata
        chunk_type = c.get("chunk_type") or c.get("metadata", {}).get("chunk_type")

        # Infer if missing
        if not chunk_type:
            chunk_type = _infer_chunk_type(c)

        # Normalize to consistent format
        normalized.append({
            "text": c.get("text", c.get("content", "")),
            "chunk_type": chunk_type,
            "source_file": c.get("source_file") or c.get("metadata", {}).get("source_file", "unknown"),
            "title": c.get("title") or c.get("metadata", {}).get("title", ""),
            "metadata": {
                **c.get("metadata", {}),
                "chunk_type": chunk_type,  # ensure it's in metadata too
            }
        })
    return normalized


def load_faiss_index():
    global _index, _chunks, _model

    _model = SentenceTransformer(settings.EMBEDDING_MODEL)

    faiss_path = Path(settings.FAISS_INDEX_PATH)
    chunks_path = Path(settings.CHUNKS_PATH)

    # ── Load FAISS index ──
    _index = faiss.read_index(str(faiss_path))

    # ── Load chunks — supports both .json and .pkl formats ──
    suffix = chunks_path.suffix.lower()

    if suffix == ".pkl":
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        raw_chunks = data["chunks"] if isinstance(data, dict) else data
    else:
        with open(chunks_path, "r") as f:
            raw_chunks = json.load(f)
        if isinstance(raw_chunks, dict):
            raw_chunks = raw_chunks.get("chunks", [])

    _chunks = _normalize_chunks(raw_chunks)
    print(f"✅ Loaded {len(_chunks)} chunks | FAISS vectors: {_index.ntotal} | Dim: {_index.d}")

    # Print chunk type breakdown
    from collections import Counter
    ct = Counter(c["chunk_type"] for c in _chunks)
    print("📊 Chunk types:", dict(ct))


def get_index(): return _index
def get_chunks(): return _chunks
def get_model(): return _model


def search(query: str, top_k: int = None) -> list[dict]:
    """Embed query and return top_k matching chunks with scores."""
    k = top_k or settings.TOP_K
    query_vec = _model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = _index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if idx >= len(_chunks):
            # Orphan vector — exists in FAISS but has no matching chunk in PKL.
            # This happens when the index has 3,256 vectors but PKL has 2,824 chunks.
            # Skip silently instead of crashing with IndexError.
            continue
        chunk = _chunks[idx].copy()
        chunk["score"] = float(dist)
        results.append(chunk)
    return results


def search_with_filter(query: str, filter_field: str, filter_value: str, top_k: int = None) -> list[dict]:
    """
    Search FAISS then post-filter by field value.
    Checks top-level fields and metadata dict.
    Fetches 8x more results to survive filtering.
    """
    k = (top_k or settings.TOP_K) * 8
    all_results = search(query, top_k=k)

    def get_field(chunk, field):
        if field in chunk:
            return str(chunk[field]).lower()
        meta = chunk.get("metadata", {}) or {}
        if field in meta:
            return str(meta[field]).lower()
        return ""

    target = filter_value.lower()
    filtered = [c for c in all_results if get_field(c, filter_field) == target]
    return filtered[:top_k or settings.TOP_K]
