import asyncio
import httpx
from core.config import settings

CURRENT_TERM = "Spring 2026"
CURRENT_DATA_YEAR = "2025-2026"


def build_prompt(question: str, chunks: list, memory_context: str = "") -> str:
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        if hasattr(c, "page_content"):
            text = c.page_content
            meta = c.metadata or {}
        else:
            text = c.get("text", c.get("content", ""))
            meta = c.get("metadata", {}) or {}

        source = (
            meta.get("term") or
            meta.get("title") or
            meta.get("page_title") or
            meta.get("source") or
            meta.get("chunk_id") or
            meta.get("source_file") or
            f"source {i}"
        )
        url = meta.get("source_url") or meta.get("url") or ""
        url_line = f"Source: {url}\n" if url else ""
        context_blocks.append(f"[{source}]\n{url_line}{text}")

    context = "\n\n".join(context_blocks)

    # Build memory section if available
    memory_section = ""
    if memory_context:
        memory_section = f"""
Conversation Memory:
{memory_context}

Use the above memory to personalize your answer. If the student's level or visa status is known,
apply the correct rules without asking again.
"""

    return f"""You are a helpful, knowledgeable assistant for Illinois Institute of Technology (IIT) students.
Current academic term: {CURRENT_TERM}. Available data covers the {CURRENT_DATA_YEAR} academic year.
{memory_section}
ANSWER RULES:
- Answer directly and confidently using the context provided
- Never reference sources by number like [1], [2], [3] — just state the facts
- Never say "based on the provided context" or "according to the context" — just answer
- Never add disclaimers like "please verify with the university" or "I recommend checking"
- If the answer is clearly in the context, state it with full confidence
- When citing a source URL, use a short readable form — only cite URLs directly relevant to the answer
- If the student's level is known from memory, use it to give a personalized answer

FORMATTING RULES:
- Format all dates as "Month Day, Year" (e.g. January 12, 2026) — never use YYYY-MM-DD format
- For dollar amounts, always specify what the amount covers (per credit hour / per semester / annual)
- Use bullet points only when listing 3 or more distinct items
- Keep answers concise but complete — do not leave out relevant facts

ACADEMIC RULES — apply these carefully:
- Full-time status: Undergraduate students need 12+ credits per semester. Graduate students need 9+ credits per semester.
- CRITICAL OVERRIDE: An undergraduate student with 9 credits is NOT full-time. Undergrad needs 12+. Do not let retrieved text override this rule.
- F-1 international students must maintain full-time status as defined above — DSO approval needed for exceptions
- Tuition data available is for {CURRENT_DATA_YEAR} — if asked about a future year, state what year the data covers and give the most recent available rate
- If the question does not specify graduate or undergraduate AND it is not known from memory, provide information for BOTH levels

WHEN DATA IS MISSING:
- If the specific information is genuinely not in the context, say so in one sentence
- Then direct the student to: registrar@illinoistech.edu, student-accounting@illinoistech.edu, or iit.edu
- Never make up information or guess at specific dates, amounts, or names

Retrieved Context:
{context}

Question: {question}

Answer:""".strip()


async def call_llm(prompt: str) -> str:
    if settings.LLM_PROVIDER == "ollama":
        return await call_ollama(prompt)
    else:
        return await call_theta(prompt)


async def call_ollama(prompt: str) -> str:
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a direct, accurate assistant for IIT students. "
                    "Give specific answers with exact numbers, dates, and names. "
                    "Never cite chunk numbers. Never add disclaimers. "
                    "Always write dates as Month Day, Year format. Never use YYYY-MM-DD. "
                    "Undergraduate full-time = 12+ credits. Graduate full-time = 9+ credits. "
                    "Use conversation memory to personalize answers when student level is known."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.7}
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


async def call_theta(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {settings.THETA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a direct, accurate assistant for IIT students. "
                        "Give specific answers with exact numbers, dates, and names. "
                        "Never cite chunk numbers like [1] or [3]. Never add disclaimers. "
                        "Always write dates as Month Day, Year format. Never use YYYY-MM-DD. "
                        "Undergraduate full-time = 12+ credits. Graduate full-time = 9+ credits. "
                        "Use conversation memory to personalize answers when student level is known."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 768,
            "temperature": 0.1,
            "top_p": 0.7,
            "stream": False
        }
    }

    url = f"{settings.THETA_BASE_URL}/{settings.THETA_MODEL}/completions"

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code == 409:
                wait = 5 * (attempt + 1)
                print(f"⚠️  Theta 409 - Retrying in {wait}s... (attempt {attempt+1}/3)")
                await asyncio.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            try:
                return data["body"]["infer_requests"][0]["output"]["message"].strip()
            except (KeyError, IndexError, TypeError):
                pass

            try:
                if "output" in data:
                    output = data["output"]
                    if isinstance(output, dict) and "choices" in output:
                        return output["choices"][0]["message"]["content"].strip()
                    if isinstance(output, str):
                        return output.strip()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError):
                pass

            raise Exception(f"Could not parse Theta response: {data}")

        raise Exception("Theta EdgeCloud has no available instances after 3 retries.")
