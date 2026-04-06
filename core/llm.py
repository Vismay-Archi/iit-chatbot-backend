import httpx
from core.config import settings


async def call_llm(prompt: str) -> str:
    """
    Call Azure OpenAI GPT-4o via Azure AI Foundry.
    Accepts a plain string prompt (system + user combined).
    """
    headers = {
        "api-key": settings.AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.3,
        "top_p": 0.7,
    }
    url = (
        f"{settings.AZURE_OPENAI_ENDPOINT.rstrip('/')}/"
        f"openai/deployments/{settings.AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version={settings.AZURE_OPENAI_API_VERSION}"
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
