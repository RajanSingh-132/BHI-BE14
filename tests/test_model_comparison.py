"""
test_model_comparison.py — side-by-side Gemini vs Gemma output quality check.

Runs the same intent extraction and analysis prompts against both models
and prints a structured comparison. Does NOT touch MongoDB or the full
generate_ai_response pipeline — isolated to the LLM call only.

Usage:
    # Default: compares gemini-2.5-flash vs gemma-4-31b-it
    python -m pytest tests/test_model_comparison.py -s

    # Override models via env:
    PRIMARY_MODEL=gemini-2.5-flash COMPARE_MODEL=gemini-2.0-flash \
        python -m pytest tests/test_model_comparison.py -s
"""

import json
import logging
import os
import sys
import time

import pytest
from dotenv import load_dotenv
from google import genai
from google.genai import types as _genai_types

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    pytest.skip("GEMINI_API_KEY not set — skipping model comparison", allow_module_level=True)

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-2.5-flash")
COMPARE_MODEL = os.getenv("COMPARE_MODEL", "gemma-4-31b-it")

_client = genai.Client(api_key=API_KEY)
_config  = _genai_types.GenerateContentConfig(
    automatic_function_calling=_genai_types.AutomaticFunctionCallingConfig(disable=True)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(model: str, prompt: str) -> tuple[str, float]:
    """Returns (response_text, latency_seconds)."""
    t0 = time.monotonic()
    try:
        resp = _client.models.generate_content(model=model, contents=prompt, config=_config)
        text = resp.text if hasattr(resp, "text") else str(resp)
        return text, time.monotonic() - t0
    except Exception as e:
        return f"[ERROR] {e}", time.monotonic() - t0


def _try_parse_json(raw: str) -> tuple[bool, str]:
    """Returns (is_valid_json, cleaned_text)."""
    clean = raw.strip()
    # Strip markdown fences
    for fence in ["```json", "```"]:
        clean = clean.replace(fence, "")
    clean = clean.strip()
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start == -1 or end <= 0:
        return False, clean
    try:
        json.loads(clean[start:end])
        return True, clean[start:end]
    except json.JSONDecodeError:
        return False, clean[start:end]


def _print_comparison(label: str, prompt_chars: int, p_raw, p_lat, c_raw, c_lat):
    p_valid, p_json = _try_parse_json(p_raw)
    c_valid, c_json = _try_parse_json(c_raw)

    print(f"\n{'='*70}")
    print(f"TEST: {label}")
    print(f"Prompt chars: {prompt_chars:,}")
    print(f"{'─'*70}")
    print(f"  [{PRIMARY_MODEL}]  latency={p_lat:.2f}s  JSON_valid={p_valid}")
    print(f"  Response snippet: {p_raw[:300]}")
    print(f"{'─'*70}")
    print(f"  [{COMPARE_MODEL}]  latency={c_lat:.2f}s  JSON_valid={c_valid}")
    print(f"  Response snippet: {c_raw[:300]}")
    print(f"{'='*70}")

    return p_valid, c_valid


# ---------------------------------------------------------------------------
# Test fixtures — realistic prompts from your actual pipeline
# ---------------------------------------------------------------------------

INTENT_PROMPT = """
You are an intent extractor. Given a user query and dataset schema, return JSON.

Dataset type: marketing_campaign
Available metrics: leads_total, leads_qualified, leads_converted, cost_total, revenue_actual, profit
Dimension keys: campaign, channel, date
Dimension values:
  campaign: [Email, Social, Paid Search, Display, Video…]
  channel: [Google, Facebook, LinkedIn, Email]

User query: "what is the total revenue by campaign?"

Return ONLY valid JSON in this format:
{
  "metric": "revenue_actual",
  "aggregation": "sum",
  "group_by": "campaign",
  "filters": {}
}
"""

ANALYSIS_PROMPT = """
You are a CRM analytics assistant. Analyse these results and return JSON.

Dataset type: marketing_campaign
Query: "what is the total revenue by campaign?"
Computed results:
{
  "metric": "revenue_actual",
  "aggregation": "sum",
  "group_by": "campaign",
  "breakdown": [
    {"group": "Paid Search", "value": 245000},
    {"group": "Email",       "value": 187000},
    {"group": "Social",      "value": 134000}
  ],
  "row_count": 240,
  "filter_applied": "none"
}

Return ONLY valid JSON:
{
  "answer": "<p>HTML summary</p>",
  "kpis": [{"name": "...", "value": ..., "unit": "$", "insight": "..."}],
  "charts": [{"type": "bar", "title": "...", "x": [...], "y": [...]}]
}
"""

AMBIGUOUS_PROMPT = """
You are an intent extractor. The dataset has ambiguous column names.

Dataset type: generic
Available metrics: Amt, Total, Val, Qty
Dimension keys: cat, grp, ref
Dimension values:
  cat: [A, B, C]

User query: "show me the total amount by category"

Return ONLY valid JSON:
{
  "metric": "<best guess>",
  "aggregation": "sum",
  "group_by": "<dimension>",
  "filters": {}
}
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _is_api_error(raw: str) -> bool:
    """True if the response is a transport/infrastructure error, not a model quality failure."""
    return raw.startswith("[ERROR]") and any(
        code in raw for code in ["503", "429", "500", "502", "504"]
    )


def test_intent_extraction_comparison():
    """Structured JSON output — the most critical capability for our pipeline."""
    prompt = INTENT_PROMPT.strip()
    p_raw, p_lat = _call(PRIMARY_MODEL, prompt)
    c_raw, c_lat = _call(COMPARE_MODEL, prompt)

    p_valid, c_valid = _print_comparison(
        "Intent Extraction (structured JSON)", len(prompt), p_raw, p_lat, c_raw, c_lat
    )

    if _is_api_error(p_raw):
        pytest.skip(f"{PRIMARY_MODEL} returned a transient API error — not a quality failure")
    assert p_valid, f"{PRIMARY_MODEL} produced invalid JSON for intent extraction"
    if not c_valid and not _is_api_error(c_raw):
        logger.warning(f"[COMPARE] {COMPARE_MODEL} did not return valid JSON — check output above")


def test_analysis_comparison():
    """Rich HTML + kpis + charts JSON — most complex output shape."""
    prompt = ANALYSIS_PROMPT.strip()
    p_raw, p_lat = _call(PRIMARY_MODEL, prompt)
    c_raw, c_lat = _call(COMPARE_MODEL, prompt)

    p_valid, c_valid = _print_comparison(
        "Analysis (HTML + KPIs + charts)", len(prompt), p_raw, p_lat, c_raw, c_lat
    )

    if _is_api_error(p_raw):
        pytest.skip(f"{PRIMARY_MODEL} returned a transient API error — not a quality failure")
    assert p_valid, f"{PRIMARY_MODEL} produced invalid JSON for analysis"
    if not c_valid and not _is_api_error(c_raw):
        logger.warning(f"[COMPARE] {COMPARE_MODEL} did not return valid JSON — check output above")


def test_ambiguous_columns_comparison():
    """
    Ambiguous column names (Amt, Total, Val) — this is where Gemma vs Gemini
    divergence is most visible. Tests the Tier 3 Claudeparsing scenario.
    """
    prompt = AMBIGUOUS_PROMPT.strip()
    p_raw, p_lat = _call(PRIMARY_MODEL, prompt)
    c_raw, c_lat = _call(COMPARE_MODEL, prompt)

    _print_comparison(
        "Ambiguous columns (Amt, Total, Val)", len(prompt), p_raw, p_lat, c_raw, c_lat
    )
    # No hard assertion — this is purely observational
