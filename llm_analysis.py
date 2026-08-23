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

BASE_SYSTEM_PROMPT = """You are a financial research assistant. You analyze a stock using
ONLY the retrieved context and technical data provided to you - never your own
outside knowledge of the company, and never speculation presented as fact.

Rules:
- If the provided context is thin or contradictory, say so explicitly.
- Never state a specific future price or percentage move as a prediction.
- Always separate "what the sources say" from "what the technicals say".
- List concrete risks mentioned in the sources.
- End with a clear statement that this is not financial advice."""

INDIA_SYSTEM_ADDENDUM = """

This is an Indian-listed equity (NSE/BSE). Apply India-specific context:
- Prices are in INR; do not assume USD unless the source explicitly says so.
- Consider India-specific risk factors where the sources support them:
  regulatory action by SEBI or RBI, FII/DII (foreign/domestic institutional
  investor) flow trends, monsoon or agri-linked demand for consumer-facing
  companies, rupee depreciation/appreciation impact on import- or
  export-heavy businesses, and sector-specific policy changes (e.g. GST,
  import duties, PLI schemes).
- News coverage for Indian equities is typically thinner than for US
  large-caps - reflect this honestly in the confidence level rather than
  overstating certainty from limited sources.
- Do not invent regulatory or macroeconomic claims not present in the
  retrieved context - only mention these factors if the sources actually
  raise them."""


def _is_indian_ticker(ticker: str) -> bool:
    return ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")


def build_system_prompt(ticker: str) -> str:
    if _is_indian_ticker(ticker):
        return BASE_SYSTEM_PROMPT + INDIA_SYSTEM_ADDENDUM
    return BASE_SYSTEM_PROMPT


def build_prompt(ticker: str, retrieved: list[dict], technicals: dict) -> str:
    is_indian = _is_indian_ticker(ticker)

    if retrieved:
        context_block = "\n\n".join(
            f"[{i+1}] ({r['source']}, {r['date']}): {r['text']}"
            for i, r in enumerate(retrieved)
        )
    else:
        context_block = "(No news or filing context retrieved for this ticker yet.)"

    tech_block = "\n".join(f"- {k}: {v}" for k, v in technicals.items())

    currency_note = "Note: prices/technicals below are in INR." if is_indian else ""
    india_risk_line = (
        "3. Key risks explicitly mentioned in the sources, including any "
        "India-specific factors (regulatory, currency, sectoral policy) if "
        "the sources raise them - do not invent ones they don't mention"
        if is_indian
        else "3. Key risks explicitly mentioned in the sources"
    )

    return f"""Analyze {ticker}. {currency_note}

RETRIEVED CONTEXT (numbered, cite by number):
{context_block}

TECHNICAL INDICATORS:
{tech_block}

Provide:
1. Sentiment summary from the retrieved context (cite sources by number)
2. How the technicals align or conflict with that sentiment
{india_risk_line}
4. A balanced outlook framed as scenarios, not a single prediction
5. Confidence level (low/medium/high) based on how much and how recent the retrieved context is"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_analysis(ticker: str, retrieved: list[dict], technicals: dict) -> str:
    prompt = build_prompt(ticker, retrieved, technicals)
    system_prompt = build_system_prompt(ticker)
    response = nim_client.chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=1500,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content