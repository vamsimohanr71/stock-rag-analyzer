"""
Builds the RAG prompt from retrieved context + technical signals and calls
an LLM (hosted free via NVIDIA Build / NIM) for a narrative analysis.
Deliberately instructed to avoid stating numeric price targets as fact -
see README for why LLMs shouldn't be used as numeric forecasters.
"""
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

# NVIDIA Build (NIM) exposes an OpenAI-compatible chat completions endpoint,
# so the standard openai SDK works unchanged - just point it at NVIDIA's
# base_url with your nvapi- key. Swap LLM_MODEL in .env for any chat model
# in the catalog at https://build.nvidia.com/models
nim_client = OpenAI(api_key=settings.NVIDIA_API_KEY, base_url=settings.NVIDIA_BASE_URL)

SYSTEM_PROMPT = """You are a financial research assistant. You analyze a stock using
ONLY the retrieved context and technical data provided to you - never your own
outside knowledge of the company, and never speculation presented as fact.

Rules:
- If the provided context is thin or contradictory, say so explicitly.
- Never state a specific future price or percentage move as a prediction.
- Always separate "what the sources say" from "what the technicals say".
- List concrete risks mentioned in the sources.
- End with a clear statement that this is not financial advice."""


def build_prompt(ticker: str, retrieved: list[dict], technicals: dict) -> str:
    if retrieved:
        context_block = "\n\n".join(
            f"[{i+1}] ({r['source']}, {r['date']}): {r['text']}"
            for i, r in enumerate(retrieved)
        )
    else:
        context_block = "(No news or filing context retrieved for this ticker yet.)"

    tech_block = "\n".join(f"- {k}: {v}" for k, v in technicals.items())

    return f"""Analyze {ticker}.

RETRIEVED CONTEXT (numbered, cite by number):
{context_block}

TECHNICAL INDICATORS:
{tech_block}

Provide:
1. Sentiment summary from the retrieved context (cite sources by number)
2. How the technicals align or conflict with that sentiment
3. Key risks explicitly mentioned in the sources
4. A balanced outlook framed as scenarios, not a single prediction
5. Confidence level (low/medium/high) based on how much and how recent the retrieved context is"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_analysis(ticker: str, retrieved: list[dict], technicals: dict) -> str:
    prompt = build_prompt(ticker, retrieved, technicals)
    response = nim_client.chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=1500,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
