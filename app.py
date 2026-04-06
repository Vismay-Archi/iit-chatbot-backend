# app.py  (FastAPI backend)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from router import query
from core.faiss_loader import load_faiss_index
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_faiss_index()
    print("✅ FAISS index and embedding model loaded.")
    yield
    print("🔻 Shutting down.")


app = FastAPI(
    title="IIT RAG API",
    description="RAG pipeline comparing Linear, Self-Query, and Traffic Cop methods",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE: /api prefix here
app.include_router(query.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "message": "IIT RAG API is running"}
