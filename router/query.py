# router/query.py

import asyncio
import time
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal

from pipeline.traffic_cop import run_traffic_cop   
from core.memory import clear_session, get_session, save_turn

router = APIRouter()

class QueryRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None
    top_k:      Optional[int] = None

class MessageItem(BaseModel):
    role:    Literal["user", "assistant"]
    content: str

class QueryResponse(BaseModel):
    question:   str
    method:     str
    session_id: str
    answer:     str
    results:    Dict[str, Any]
    history:    List[MessageItem]

def _build_history(session_id: str) -> List[MessageItem]:
    session = get_session(session_id)
    turns   = session.get("history", [])
    recent  = turns[-6:] if len(turns) > 6 else turns
    items: List[MessageItem] = []
    for t in recent:
        items.append(MessageItem(role="user",      content=t["question"]))
        items.append(MessageItem(role="assistant", content=t["answer"]))
    return items

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    question   = req.question.strip()
    session_id = req.session_id or str(uuid.uuid4())
    start      = time.time()

    # Always use traffic_cop only
    r = await run_traffic_cop(question, session_id=session_id)
    results = {"traffic_cop": r}
    results["total_time_ms"] = round((time.time() - start) * 1000)

    answer = r.get("answer") or r.get("clarification_suggestion", "")

    return QueryResponse(
        question=question,
        method="traffic_cop",
        session_id=session_id,
        answer=answer,
        results=results,
        history=_build_history(session_id),
    )

# ── Session endpoints ──────────────────────────────────────────────────────────

@router.post("/session/new")
async def new_session():
    return {"session_id": str(uuid.uuid4())}

@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    session = get_session(session_id)
    return {
        "session_id":     session_id,
        "user_context":   session.get("user_context", {}),
        "history_length": len(session.get("history", [])),
        "history":        session.get("history", []),
    }

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared."}