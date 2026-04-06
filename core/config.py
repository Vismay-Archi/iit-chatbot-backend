from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    LLM_PROVIDER: str = "azure_openai"

    # Azure OpenAI / Foundry
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"

    # Ollama (fallback)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"

    # FAISS
    FAISS_INDEX_PATH: str = "data/faiss_store.faiss"
    CHUNKS_PATH: str = "data/faiss_store.pkl"
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"

    # API
    ALLOWED_ORIGINS: List[str] = ["*"]
    TOP_K: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
