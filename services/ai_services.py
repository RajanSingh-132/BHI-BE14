"""
AI Services — multi-dataset orchestration layer.

Pipeline (dataset path):
  Stage 1 — Intent Extraction   LLM call #1 — merged schema → single JSON intent
  Stage 2 — Calculation         pandas/numpy per dataset — deterministic, no LLM
  Stage 3 — Analysis            LLM call #2 — receives all N computed results

Total Gemini calls per query: 2, regardless of how many datasets are active.

Session design:
  - generate_ai_response receives session_id (UUID from X-Session-ID header).
  - _resolve_active_datasets(session_id) reads ONLY the session keyed by that UUID.
  - No app.state globals. No hardcoded user_id. No cross-session bleed.
  - Server restart is safe: if session_id has no MongoDB entry (expired TTL or
    first request), returns [] → user is prompted to upload a dataset.

Fallback paths:
  - No schema profile  → legacy single-LLM path (per dataset) — ONLY if schema truly absent;
                         if intent extraction fails (transient error) → graceful error, NO legacy loop
  - No active dataset  → RAG fallback (single LLM call)

Gemini call budget per request (worst case):
  - Intent:   max 2 attempts (1 initial + 1 retry) = 2
  - Analysis: max 2 attempts = 2
  - Total:    4 calls maximum, never N × retries.

Rate-limit handling:
  - 429 responses include a 'retryDelay' field (e.g. '39s').
    _parse_retry_delay() extracts this and uses it as the actual sleep duration
    instead of the fixed exponential backoff (which was always too short).
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types as _genai_types

from mongo_client import mongo_client, _make_dataset_key
from prompts.analysis_prompt import ANALYSIS_PROMPT, MULTI_DATASET_ANALYSIS_PROMPT
from prompts.intent_prompt import INTENT_EXTRACTION_PROMPT
from prompt import SYSTEM_PROMPT
from rag_retriever import RAGRetriever
from services.calculation_engine import calculate as run_calculation
from utils.request_tracker import tracker, start_request, get_stats
from config.display_config import get_fields_for_prompt

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("[AI_SERVICES] GEMINI_API_KEY is not set")

# Model selection — switch via env var, no code change required.
# Supported values (Google GenAI API):
#   gemini-2.5-flash        — default, best JSON reliability, 1M context
#   gemma-4-31b-it          — Gemma 4 31B instruction-tuned, 128K context
#   gemini-2.0-flash        — previous gen Flash
# Example: LLM_MODEL=gemma-4-31b-it uvicorn main:app
_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

_gemini    = genai.Client(api_key=API_KEY)
_retriever = RAGRetriever()

logger.info(f"[AI_SERVICES] LLM model = {_LLM_MODEL!r}")

# ── Retry config ──────────────────────────────────────────────────────────────
# 2 total attempts per call (1 initial + 1 retry).
# Keeping this low is the primary defence against quota exhaustion:
# worst case is 2 (intent) + 2 (analysis) = 4 Gemini calls per request.
_GEMINI_MAX_RETRIES = 2
_GEMINI_RETRY_BASE  = 4.0   # fallback sleep seconds when retryDelay is absent

# Disable Automatic Function Calling (AFC) globally.
# The google-genai SDK enables AFC by default for Gemini 2.5 models, which can
# silently make up to 10 additional API calls per generate_content invocation.
# We do not use tools, so AFC must be disabled to prevent unexpected API usage.
_GEMINI_CONFIG = _genai_types.GenerateContentConfig(
    automatic_function_calling=_genai_types.AutomaticFunctionCallingConfig(
        disable=True
    )
)

# Errors that are safe to retry (transient server-side issues)
_RETRYABLE_CODES = frozenset(["429", "500", "502", "503", "504"])


# ---------------------------------------------------------------------------
# Retry delay parser — extracts the server-supplied wait time from 429 bodies
# ---------------------------------------------------------------------------

def _parse_retry_delay(err_str: str) -> Optional[float]:
    """
    Gemini 429 bodies contain: 'retryDelay': '39s'
    This extracts the numeric seconds so we honour the server's own backoff.
    Returns None if the field is absent (caller falls back to _GEMINI_RETRY_BASE).

    Pattern handles both JSON/repr forms:
      'retryDelay': '39s'
      "retryDelay": "39.5s"
      retryDelay=42s
    """
    m = re.search(
        r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s",
        err_str,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Gemini call wrapper — retry + backoff + per-request transparent logging
# ---------------------------------------------------------------------------

def _gemini_generate(
    prompt: str,
    label:  str = "LLM",
    extra:  str = "",
) -> Optional[str]:
    """
    Call Gemini with up to _GEMINI_MAX_RETRIES attempts.

    Logging (one line per call attempt, one summary at completion):
      [LLM_CALL #n] purpose=INTENT | model=gemini-2.5-flash | prompt_chars=1,234
      [LLM_CALL #n] purpose=INTENT | status=success | waited=0.0s
      — or —
      [LLM_CALL #n] purpose=INTENT | status=FAILED | waited=39.0s | error=…

    Key behaviour:
    - AFC is disabled → no hidden tool-call rounds.
    - 429 retryDelay is parsed and used as the actual sleep duration.
    - tracker.gemini_hit() counted only on success (global legacy counter).
    - Returns text string on success, None on permanent failure.
    """
    stats = get_stats()
    rec   = stats.record(purpose=label, prompt_chars=len(prompt), extra=extra, model=_LLM_MODEL)

    total_wait = 0.0

    for attempt in range(1, _GEMINI_MAX_RETRIES + 1):
        try:
            response = _gemini.models.generate_content(
                model=_LLM_MODEL,
                contents=prompt,
                config=_GEMINI_CONFIG,
            )
            text = response.text if hasattr(response, "text") else str(response)
            tracker.gemini_hit()
            stats.complete(rec, success=True, wait_s=total_wait)
            return text

        except Exception as e:
            err_str = str(e)
            is_retryable = any(code in err_str for code in _RETRYABLE_CODES)

            if is_retryable and attempt < _GEMINI_MAX_RETRIES:
                # Honour the server's own retryDelay if present, else fall back.
                server_delay = _parse_retry_delay(err_str)
                wait         = server_delay if server_delay is not None else _GEMINI_RETRY_BASE

                logger.warning(
                    f"[{label}] Transient error (attempt {attempt}/{_GEMINI_MAX_RETRIES}): "
                    f"{err_str[:120]}. "
                    f"{'Server retryDelay' if server_delay else 'Fixed backoff'} = {wait:.1f}s"
                )
                total_wait += wait
                time.sleep(wait)
            else:
                logger.error(
                    f"[{label}] Permanent failure after {attempt} attempt(s): {e}"
                )
                stats.complete(rec, success=False, wait_s=total_wait, error=err_str[:200])
                return None

    return None


# ---------------------------------------------------------------------------
# Conversation history formatter
# ---------------------------------------------------------------------------

def _format_history(history) -> str:
    """Format the last 6 conversation turns (3 exchanges) as plain text."""
    if not history:
        return ""
    turns = list(history)[-6:]
    lines = []
    for msg in turns:
        if hasattr(msg, "role"):
            role, content = ("User" if msg.role == "human" else "Assistant"), msg.content
        else:
            role, content = ("User" if msg.get("role") == "human" else "Assistant"), msg.get("content", "")
        if content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Active dataset resolution — keyed by session_id ONLY
# ---------------------------------------------------------------------------

def _resolve_active_datasets(session_id: str) -> List[str]:
    """
    Return the list of active datasets for this session.

    Reads from MongoDB session_state keyed by session_id.
    Returns [] if the session doesn't exist or has expired.

    No fallback to app.state globals. No hardcoded user_id.
    An empty return means "no dataset uploaded yet" — the caller
    will route to the RAG fallback and the user will be prompted to upload.
    """
    if not session_id:
        return []
    try:
        state    = mongo_client.get_session_state(session_id)
        datasets = state.get("active_datasets", [])
        if datasets:
            logger.info(
                f"[AI_SERVICES] session_id={session_id!r} → "
                f"active_datasets={datasets}"
            )
        return datasets
    except Exception as e:
        logger.error(f"[AI_SERVICES] _resolve_active_datasets error: {e}")
        return []


# ---------------------------------------------------------------------------
# Dataset + schema fetchers
# ---------------------------------------------------------------------------

def _fetch_dataset(file_name: str) -> List[Dict]:
    try:
        doc = mongo_client.db["documents"].find_one({"type": "dataset", "file_name": file_name})
        if doc:
            rows = doc.get("data", [])
            logger.info(f"[AI_SERVICES] Loaded {len(rows)} rows for '{file_name}'")
            return rows
        logger.warning(f"[AI_SERVICES] Dataset '{file_name}' not found in DB")
        return []
    except Exception as e:
        logger.error(f"[AI_SERVICES] fetch_dataset error: {e}")
        return []


def _fetch_schema_profile(file_name: str) -> Optional[Dict]:
    try:
        profile = mongo_client.db["schema_profiles"].find_one({"file_name": file_name})
        if profile:
            profile.pop("_id", None)
            return profile
        logger.warning(f"[AI_SERVICES] No schema profile for '{file_name}'")
        return None
    except Exception as e:
        logger.error(f"[AI_SERVICES] fetch_schema_profile error: {e}")
        return None


# ---------------------------------------------------------------------------
# Schema merging — builds a unified view across all active datasets
# ---------------------------------------------------------------------------

def _merge_schemas(schemas: List[Dict]) -> Dict:
    """
    Produce a merged schema_profile for intent extraction.
    Exposes all metrics and dimensions from all datasets so the intent
    extractor can select the right metric even if it only exists in one.

    Rules:
      - available_metrics: union (first-seen order)
      - dimension_map: union (first dataset wins on key conflict)
      - dimension_values: union (all unique values merged per col)
      - dataset_type: common type or "multi_dataset" when mixed
    """
    merged_metrics:  List[str]       = []
    merged_dim_map:  Dict[str, str]  = {}
    merged_dim_vals: Dict[str, List] = {}
    all_types:       List[str]       = []

    for schema in schemas:
        for m in schema.get("available_metrics", []):
            if m not in merged_metrics:
                merged_metrics.append(m)

        for dim, col in schema.get("dimension_map", {}).items():
            if dim not in merged_dim_map:
                merged_dim_map[dim] = col

        for col, vals in schema.get("dimension_values", {}).items():
            if col not in merged_dim_vals:
                merged_dim_vals[col] = list(vals)
            else:
                existing = set(str(v) for v in merged_dim_vals[col])
                merged_dim_vals[col].extend(
                    v for v in vals if str(v) not in existing
                )

        if schema.get("dataset_type"):
            all_types.append(schema["dataset_type"])

    unique_types = list(dict.fromkeys(all_types))
    dataset_type = unique_types[0] if len(unique_types) == 1 else "multi_dataset"

    return {
        "dataset_type":      dataset_type,
        "available_metrics": merged_metrics,
        "dimension_map":     merged_dim_map,
        "dimension_values":  merged_dim_vals,
    }


# ---------------------------------------------------------------------------
# Stage 1 — Intent extraction (runs once on merged schema)
# ---------------------------------------------------------------------------

def _extract_intent(query: str, schema_profile: Dict, dataset_count: int = 1) -> Optional[Dict]:
    """
    LLM call #1.  Returns a JSON intent dict or None on failure.
    Uses the merged schema when multiple datasets are active.

    dataset_count is passed to the transparent call logger only (no logic effect).
    """
    available_metrics = schema_profile.get("available_metrics", [])
    dimension_map     = schema_profile.get("dimension_map", {})
    dimension_values  = schema_profile.get("dimension_values", {})
    dataset_type      = schema_profile.get("dataset_type", "generic")

    dim_val_lines = []
    for col, vals in dimension_values.items():
        sample = ", ".join(str(v) for v in vals[:5])
        suffix = "…" if len(vals) > 5 else ""
        dim_val_lines.append(f"  {col}: [{sample}{suffix}]")
    dim_summary = "\n".join(dim_val_lines) or "  (none)"

    prompt = INTENT_EXTRACTION_PROMPT.format(
        dataset_type             = dataset_type,
        available_metrics        = ", ".join(available_metrics) or "none",
        dimension_map_keys       = ", ".join(dimension_map.keys()) or "none",
        dimension_values_summary = dim_summary,
        query                    = query,
    )

    raw = _gemini_generate(
        prompt,
        label = "INTENT",
        extra = f"dataset_count={dataset_count}",
    )
    if not raw:
        return None

    try:
        raw   = _strip_fences(raw)
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end <= 0:
            logger.warning(f"[INTENT] No JSON in response: {raw[:200]}")
            return None
        intent = json.loads(raw[start:end])
        logger.info(f"[INTENT] Extracted: {intent}")
        return intent
    except json.JSONDecodeError as e:
        logger.warning(f"[INTENT] JSON parse failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 3 — Analysis (receives all N computed results)
# ---------------------------------------------------------------------------

def _analyze_results_multi(
    query:           str,
    dataset_results: List[Dict],   # [{dataset_name, display_name, calc_result}, ...]
    history:         Any,
) -> Dict:
    """
    LLM call #2 for multi-dataset path.
    Passes all N results labeled by dataset name.
    Returns {answer, kpis, charts} with deduplication applied.
    """
    history_text    = _format_history(history)
    history_section = f"\nConversation History:\n{history_text}\n" if history_text else ""

    labeled = [
        {
            "dataset":     dr["display_name"],
            "file_name":   dr["dataset_name"],
            "calc_result": dr["calc_result"],
        }
        for dr in dataset_results
    ]

    dataset_names  = ", ".join(dr["display_name"] for dr in dataset_results)
    results_json   = json.dumps(labeled, indent=2, default=str)

    # Use the first dataset's type to drive KPI card selection
    first_type = dataset_results[0].get("dataset_type", "") if dataset_results else ""

    prompt = MULTI_DATASET_ANALYSIS_PROMPT.format(
        dataset_names            = dataset_names,
        query                    = query,
        dataset_results_json     = results_json,
        kpi_display_instructions = get_fields_for_prompt(query, first_type),
    ) + history_section

    raw = _gemini_generate(
        prompt,
        label = "ANALYSIS",
        extra = f"dataset_count={len(dataset_results)}",
    )
    if not raw:
        return _fallback_response_multi(dataset_results, query)

    try:
        raw   = _strip_fences(raw)
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end <= 0:
            return _fallback_response_multi(dataset_results, query)

        parsed = json.loads(raw[start:end])
        answer = _clean_text(parsed.get("answer", ""))
        kpis   = parsed.get("kpis", [])
        charts = _deduplicate_charts(parsed.get("charts", []))

        if not answer:
            return _fallback_response_multi(dataset_results, query)

        return {"answer": answer, "kpis": kpis, "charts": charts}

    except json.JSONDecodeError as e:
        logger.warning(f"[ANALYSIS] JSON parse failed: {e}")
        return _fallback_response_multi(dataset_results, query)


def _analyze_results(
    query:          str,
    calc_result:    Dict,
    schema_profile: Dict,
    history:        Any,
) -> Dict:
    """LLM call #2 for single-dataset path (backward compat)."""
    history_text    = _format_history(history)
    history_section = f"\nConversation History:\n{history_text}\n" if history_text else ""

    dataset_type = schema_profile.get("dataset_type", "generic")
    prompt = ANALYSIS_PROMPT.format(
        dataset_type             = dataset_type,
        row_count                = calc_result.get("row_count", "unknown"),
        filter_applied           = calc_result.get("filter_applied", "none"),
        computed_results_json    = json.dumps(calc_result, indent=2, default=str),
        query                    = query,
        kpi_display_instructions = get_fields_for_prompt(query, dataset_type),
    ) + history_section

    raw = _gemini_generate(prompt, label="ANALYSIS")
    if not raw:
        return _fallback_response(calc_result, query)

    try:
        raw   = _strip_fences(raw)
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end <= 0:
            return _fallback_response(calc_result, query)

        parsed = json.loads(raw[start:end])
        answer = _clean_text(parsed.get("answer", ""))
        kpis   = parsed.get("kpis", [])
        charts = _deduplicate_charts(parsed.get("charts", []))

        if not answer:
            return _fallback_response(calc_result, query)

        return {"answer": answer, "kpis": kpis, "charts": charts}

    except json.JSONDecodeError as e:
        logger.warning(f"[ANALYSIS] JSON parse failed: {e}")
        return _fallback_response(calc_result, query)
    except Exception as e:
        logger.error(f"[ANALYSIS] Gemini call failed: {e}")
        return {"answer": "AI analysis service unavailable.", "kpis": [], "charts": []}


# ---------------------------------------------------------------------------
# Fallback responses (no LLM available)
# ---------------------------------------------------------------------------

def _fallback_response(calc_result: Dict, query: str) -> Dict:
    result  = calc_result.get("result")
    metric  = calc_result.get("metric", "")
    unit    = calc_result.get("unit", "")
    formula = calc_result.get("formula", "")

    if result is not None:
        answer = (
            f"<p><strong>Result:</strong> {result} {unit}</p>"
            f"<p><strong>Formula:</strong> {formula}</p>"
        )
        kpis = [{"name": metric, "value": result, "unit": unit or "", "insight": ""}]
    elif calc_result.get("breakdown"):
        top    = calc_result["breakdown"][0]
        answer = f"<p><strong>Top result:</strong> {top['group']} — {top['value']} {unit}</p>"
        kpis   = [{"name": top["group"], "value": top["value"], "unit": unit or "", "insight": ""}]
    else:
        answer = "<p>Could not generate analysis for this query.</p>"
        kpis   = []

    return {"answer": answer, "kpis": kpis, "charts": []}


def _fallback_response_multi(dataset_results: List[Dict], query: str) -> Dict:
    parts = []
    kpis  = []
    for dr in dataset_results:
        cr = dr["calc_result"]
        r  = cr.get("result")
        if r is not None:
            parts.append(
                f"<p><strong>{dr['display_name']}:</strong> "
                f"{cr.get('metric', '')} = {r} {cr.get('unit', '')}</p>"
            )
            kpis.append({
                "name":    f"{dr['display_name']}: {cr.get('metric', '')}",
                "value":   r,
                "unit":    cr.get("unit", ""),
                "insight": "",
            })
        else:
            parts.append(
                f"<p><strong>{dr['display_name']}:</strong> "
                "Data not available for this metric.</p>"
            )
    return {
        "answer": "".join(parts) or "<p>Could not generate analysis.</p>",
        "kpis":   kpis[:3],
        "charts": [],
    }


# ---------------------------------------------------------------------------
# Legacy dataset path (no schema profile)
# Note: ONLY called when a dataset genuinely has no schema profile stored.
#       It is NOT called as a fallback when intent extraction fails due to
#       rate-limiting — that case returns a graceful retry message instead.
# ---------------------------------------------------------------------------

def _legacy_dataset_path(
    dataset: str,
    message: str,
    history: Any,
    data:    List[Dict],
) -> Dict:
    logger.info(f"[AI_SERVICES] Legacy path for '{dataset}' (no schema profile)")
    dataset_json    = json.dumps(data[:50], indent=2)
    history_text    = _format_history(history)
    history_section = f"\nConversation History:\n{history_text}\n" if history_text else ""

    prompt = f"{SYSTEM_PROMPT}\n\nDataset:\n{dataset_json}\n{history_section}\nUser Query:\n{message}"

    raw = _gemini_generate(prompt, label="LEGACY", extra=f"dataset={dataset}")
    if not raw:
        return {"answer": "AI service unavailable.", "kpis": [], "charts": []}

    raw   = _strip_fences(raw)
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end <= 0:
        return {"answer": raw.strip(), "kpis": [], "charts": []}

    try:
        parsed = json.loads(raw[start:end])
        return {
            "answer": _clean_text(parsed.get("answer", "")),
            "kpis":   parsed.get("kpis", []),
            "charts": _deduplicate_charts(parsed.get("charts", [])),
        }
    except json.JSONDecodeError:
        return {"answer": raw.strip(), "kpis": [], "charts": []}


# ---------------------------------------------------------------------------
# Conversational / no-dataset short-circuit
# ---------------------------------------------------------------------------

# Patterns that are clearly conversational and need no LLM or RAG call.
# Matched against the lowercased, stripped query.
_CONVERSATIONAL_PREFIXES = (
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "what can you do", "what are you", "who are you",
    "thanks", "thank you", "ok", "okay", "great", "awesome", "cool",
    "help", "what is this", "what does this do",
)

_NO_DATASET_REPLY = (
    "<p>Hi! I'm your Business Intelligence assistant. "
    "Please upload a dataset (Excel file) using the attachment button, "
    "and I'll answer questions about your data.</p>"
)


def _is_conversational(query: str) -> bool:
    """True for greetings and meta-questions that need no data context."""
    q = query.strip().rstrip("?!.,")
    return q in _CONVERSATIONAL_PREFIXES or any(
        q.startswith(p) for p in _CONVERSATIONAL_PREFIXES
    )


# ---------------------------------------------------------------------------
# RAG fallback (no active dataset — genuine question)
# ---------------------------------------------------------------------------

_RAG_TOP_K    = 5      # max docs sent to LLM — 100 irrelevant docs = noise + token waste
_RAG_MIN_SCORE = 0.40  # stricter than retriever's own threshold; skip LLM if nothing relevant

def _rag_fallback(message: str, history: Any) -> Dict:
    logger.info("[AI_SERVICES] No active dataset — RAG fallback")
    docs = _retriever.get_relevant_documents(message)

    # Filter to genuinely relevant docs and cap at _RAG_TOP_K
    relevant = [d for d in docs if getattr(d, "score", 1.0) >= _RAG_MIN_SCORE][:_RAG_TOP_K]

    if not relevant:
        # Nothing relevant — no point spending an LLM call on noise
        logger.info(
            f"[AI_SERVICES] RAG: 0 docs above score={_RAG_MIN_SCORE} — "
            "returning no-dataset prompt without LLM call"
        )
        return {
            "answer": _NO_DATASET_REPLY,
            "kpis":   [],
            "charts": [],
        }

    context = "\n\n".join(d.page_content for d in relevant)
    logger.info(f"[AI_SERVICES] RAG: {len(relevant)} relevant docs passed to LLM")

    history_text    = _format_history(history)
    history_section = f"\nConversation History:\n{history_text}\n" if history_text else ""

    prompt = (
        f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n"
        f"{history_section}\nUser Query:\n{message}"
    )

    raw = _gemini_generate(prompt, label="RAG")
    return {"answer": (raw or "AI service unavailable.").strip(), "kpis": [], "charts": []}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_ai_response(
    session_id: str,
    message:    str,
    history:    Any  = None,
    request:    Any  = None,   # kept for signature compatibility; not used for state
) -> Dict:
    """
    Orchestrates the full multi-dataset query pipeline.

    1. Initialise per-request LLM call tracker
    2. Resolve active datasets by session_id (MongoDB only — no globals)
    3. Build stable cache key from sorted dataset names
    4. Cache check
    5. Fetch data + schema profiles for all active datasets
    6. Merge schemas → extract intent (LLM call #1)
       → on failure: return graceful retry message (NO per-dataset legacy loop)
    7. Calculate per dataset (deterministic, N pandas calls)
    8. Analyse all results together (LLM call #2)
    9. Cache result
    10. Emit [LLM_SUMMARY] log line

    Maximum Gemini calls per request: 4
      (2 intent attempts + 2 analysis attempts — regardless of N datasets)
    """
    # ── 1. Start per-request transparent tracking ─────────────────────────────
    stats = start_request()
    logger.info(
        f"[AI_SERVICES] ▶ generate_ai_response | session={session_id!r} | query={message!r}"
    )

    query    = message.lower().strip()
    datasets = _resolve_active_datasets(session_id)

    # ── No active dataset ─────────────────────────────────────────────────────
    if not datasets:
        # Greetings and meta-questions: answer instantly, no LLM or RAG call.
        if _is_conversational(query):
            logger.info("[AI_SERVICES] Conversational message with no dataset — static reply")
            stats.summary()
            return {"answer": _NO_DATASET_REPLY, "kpis": [], "charts": []}

        logger.info(
            f"[AI_SERVICES] session_id={session_id!r} — no active dataset, RAG fallback"
        )
        result = _rag_fallback(message, history)
        stats.summary()
        return result

    dataset_key = _make_dataset_key(datasets)
    logger.info(
        f"[AI_SERVICES] session_id={session_id!r} | datasets={datasets} | key={dataset_key!r}"
    )

    # ── Cache check ───────────────────────────────────────────────────────────
    cached = mongo_client.get_cached_result(dataset_key, query)
    if cached:
        logger.info(f"[AI_SERVICES] Cache hit — key={dataset_key!r}, query={query!r}")
        stats.summary()
        return {"answer": cached["answer"], "kpis": cached["kpis"], "charts": cached["charts"]}

    # ── Fetch data and schemas for all datasets ───────────────────────────────
    dataset_payloads: List[Dict] = []
    all_schemas:      List[Dict] = []

    for file_name in datasets:
        data   = _fetch_dataset(file_name)
        schema = _fetch_schema_profile(file_name)

        if not data:
            logger.warning(f"[AI_SERVICES] No data for '{file_name}' — skipping")
            continue

        display_name = file_name.rsplit(".", 1)[0].replace("_", " ")
        dataset_payloads.append({
            "name":         file_name,
            "display_name": display_name,
            "data":         data,
            "schema":       schema,
        })
        if schema:
            all_schemas.append(schema)

    if not dataset_payloads:
        logger.warning(
            f"[AI_SERVICES] session_id={session_id!r} — all datasets empty, RAG fallback"
        )
        result = _rag_fallback(message, history)
        stats.summary()
        return result

    # ── Split: datasets with schema vs without ────────────────────────────────
    with_schema    = [dp for dp in dataset_payloads if dp["schema"]]
    without_schema = [dp for dp in dataset_payloads if not dp["schema"]]

    # Datasets with NO stored schema profile → single legacy call each.
    # This is the "truly missing schema" case (upload happened without profiling).
    # This is NOT triggered when intent extraction fails.
    legacy_results: List[Dict] = []
    for dp in without_schema:
        legacy_result = _legacy_dataset_path(dp["name"], message, history, dp["data"])
        legacy_results.append({
            "dataset_name": dp["name"],
            "display_name": dp["display_name"],
            "calc_result":  {"result": None, "error": "legacy path — no schema profile"},
            "llm_result":   legacy_result,
        })

    if not with_schema:
        result = legacy_results[0]["llm_result"] if legacy_results else _rag_fallback(message, history)
        stats.summary()
        return result

    # ── Stage 1: Extract intent from merged schema (single LLM call) ──────────
    merged_schema = _merge_schemas(all_schemas)
    query_intent  = _extract_intent(message, merged_schema, dataset_count=len(with_schema))

    if not query_intent:
        # !! CRITICAL: Do NOT fall back to per-dataset legacy loop here.
        #
        # The intent failure is almost certainly a transient API error (503/429),
        # not a data problem. Running _legacy_dataset_path for each dataset would
        # multiply calls by N, making quota exhaustion certain.
        #
        # Instead: return a user-facing retry message. The calculation engine
        # and analysis stage still haven't been called, so we've burned at most
        # 2 Gemini calls (intent attempts) in the worst case.
        logger.warning(
            "[AI_SERVICES] Intent extraction failed — returning retry message "
            "(no legacy fallback loop to avoid quota exhaustion)"
        )
        stats.summary()
        return {
            "answer": (
                "<p>I wasn't able to process that query right now — "
                "the AI service may be temporarily rate-limited or unavailable. "
                "Please wait a moment and try again. "
                "Your data is loaded and ready.</p>"
            ),
            "kpis":   [],
            "charts": [],
        }

    # ── Stage 2: Calculate per dataset (deterministic — no LLM) ──────────────
    dataset_results: List[Dict] = []

    for dp in with_schema:
        calc_result = run_calculation(dp["data"], query_intent, dp["schema"])
        logger.info(
            f"[AI_SERVICES] [{dp['display_name']}] "
            f"metric={calc_result.get('metric')}, "
            f"result={calc_result.get('result')}, "
            f"source={calc_result.get('source')}"
        )
        dataset_results.append({
            "dataset_name": dp["name"],
            "display_name": dp["display_name"],
            "calc_result":  calc_result,
            "dataset_type": dp["schema"].get("dataset_type", ""),
        })

    # ── Stage 3: Analyse all results (single LLM call) ───────────────────────
    if len(dataset_results) == 1 and not legacy_results:
        result = _analyze_results(
            message,
            dataset_results[0]["calc_result"],
            with_schema[0]["schema"],
            history,
        )
    else:
        result = _analyze_results_multi(message, dataset_results, history)

    # ── Cache the result ──────────────────────────────────────────────────────
    if result.get("answer"):
        try:
            mongo_client.save_result({
                "dataset_key": dataset_key,
                "file_name":   datasets[0],
                "query":       query,
                "answer":      result["answer"],
                "kpis":        result.get("kpis", []),
                "charts":      result.get("charts", []),
            })
        except Exception as e:
            logger.error(f"[AI_SERVICES] Cache save failed: {e}")

    stats.summary()
    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text)
    return text.replace("```", "").strip()


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\n",  " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.replace("Answer:", "").strip()


def _deduplicate_charts(charts: List[Dict]) -> List[Dict]:
    """
    Remove duplicate chart specs by (type, title) key.
    Preserves first occurrence; discards subsequent duplicates.
    """
    seen:   set        = set()
    unique: List[Dict] = []

    for chart in charts:
        key = (
            str(chart.get("type", "")).lower().strip(),
            str(chart.get("title", "")).lower().strip(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(chart)
        else:
            logger.debug(
                f"[AI_SERVICES] Duplicate chart removed: type={key[0]!r}, title={key[1]!r}"
            )

    return unique
