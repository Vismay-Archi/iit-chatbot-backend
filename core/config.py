from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # LLM Provider: "theta" or "ollama"
    LLM_PROVIDER: str = "theta"

    # Theta EdgeCloud
    THETA_API_KEY: str = "your-theta-api-key-here"
    THETA_BASE_URL: str = "https://ondemand.thetaedgecloud.com/infer_request"
    THETA_MODEL: str = "llama_3_8b"

    # Ollama (fallback)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"

    # FAISS — supports .faiss/.json or .faiss/.pkl
    FAISS_INDEX_PATH: str = "data/faiss_store.faiss"
    CHUNKS_PATH: str = "data/faiss_store.pkl"   # or chunks.json

    # Embedding model — must match what was used to build the index
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"

    # API
    ALLOWED_ORIGINS: List[str] = ["*"]
    TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
