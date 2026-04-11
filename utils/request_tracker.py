"""
request_tracker.py — Per-request transparent LLM call logging.

Uses Python ContextVar so each async request gets its own isolated counter.
No global state that bleeds across concurrent requests.

Usage in route handler:
    from utils.request_tracker import start_request
    start_request()  # call once at top of handler

Usage in _gemini_generate (automatic via get_stats()):
    [LLM_CALL #1] purpose=INTENT | model=gemini-2.5-flash | prompt_chars=1,234 | dataset_count=2
    [LLM_CALL #2] purpose=ANALYSIS | model=gemma-4-31b-it | prompt_chars=3,210
    [LLM_SUMMARY] total_calls=2 | successful=2 | failed=0 | total_wait=0.0s | elapsed=4.2s
"""

import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-call record
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    call_num:     int
    purpose:      str          # INTENT | ANALYSIS | LEGACY | RAG
    prompt_chars: int
    status:       str = "pending"   # pending | success | failed
    wait_s:       float = 0.0
    error:        str   = ""


# ---------------------------------------------------------------------------
# Per-request stats container
# ---------------------------------------------------------------------------

@dataclass
class RequestStats:
    calls:      List[CallRecord] = field(default_factory=list)
    start_time: float            = field(default_factory=time.monotonic)

    def record(
        self,
        purpose:      str,
        prompt_chars: int,
        extra:        str = "",
        model:        str = "gemini-2.5-flash",
    ) -> CallRecord:
        """
        Log the start of an LLM call and return a CallRecord to update later.
        model:  pass _LLM_MODEL from ai_services so the log reflects the actual model in use.
        extra:  optional one-line context (e.g. 'dataset_count=3')
        """
        rec = CallRecord(
            call_num     = len(self.calls) + 1,
            purpose      = purpose,
            prompt_chars = prompt_chars,
        )
        self.calls.append(rec)

        extra_part = f" | {extra}" if extra else ""
        logger.info(
            f"[LLM_CALL #{rec.call_num}] purpose={purpose}"
            f" | model={model}"
            f" | prompt_chars={prompt_chars:,}"
            f"{extra_part}"
        )
        return rec

    def complete(
        self,
        rec:     CallRecord,
        success: bool,
        wait_s:  float = 0.0,
        error:   str   = "",
    ) -> None:
        rec.status = "success" if success else "failed"
        rec.wait_s = wait_s
        rec.error  = error

        if success:
            logger.info(
                f"[LLM_CALL #{rec.call_num}] purpose={rec.purpose}"
                f" | status=success | waited={wait_s:.1f}s"
            )
        else:
            logger.warning(
                f"[LLM_CALL #{rec.call_num}] purpose={rec.purpose}"
                f" | status=FAILED | waited={wait_s:.1f}s"
                + (f" | error={error[:120]}" if error else "")
            )

    def summary(self) -> None:
        """
        Emit one [LLM_SUMMARY] line. Call this at the end of every
        generate_ai_response() execution path.
        """
        elapsed    = time.monotonic() - self.start_time
        total      = len(self.calls)
        successful = sum(1 for c in self.calls if c.status == "success")
        failed     = sum(1 for c in self.calls if c.status == "failed")
        total_wait = sum(c.wait_s for c in self.calls)

        logger.info(
            f"[LLM_SUMMARY] total_calls={total}"
            f" | successful={successful}"
            f" | failed={failed}"
            f" | total_wait={total_wait:.1f}s"
            f" | elapsed={elapsed:.1f}s"
        )


# ---------------------------------------------------------------------------
# ContextVar — one RequestStats per coroutine chain (async-safe)
# ---------------------------------------------------------------------------

_current_stats: contextvars.ContextVar[Optional[RequestStats]] = contextvars.ContextVar(
    "_current_stats", default=None
)


def start_request() -> RequestStats:
    """
    Initialise a fresh RequestStats for the current request.
    Call this at the top of generate_ai_response() (or in the route handler).
    Returns the stats object (rarely needed by the caller directly).
    """
    stats = RequestStats()
    _current_stats.set(stats)
    return stats


def get_stats() -> RequestStats:
    """
    Return the RequestStats for the current request.
    Creates one lazily if start_request() was never called (safety net).
    """
    stats = _current_stats.get()
    if stats is None:
        stats = RequestStats()
        _current_stats.set(stats)
    return stats


# ---------------------------------------------------------------------------
# Legacy global counter — kept so existing tracker.gemini_hit() calls compile.
# The real logging now lives in RequestStats / _gemini_generate.
# ---------------------------------------------------------------------------

class _LegacyTracker:
    def __init__(self) -> None:
        self.total_api_calls    = 0
        self.total_gemini_calls = 0

    def api_hit(self) -> None:
        self.total_api_calls += 1

    def gemini_hit(self) -> None:
        self.total_gemini_calls += 1


tracker = _LegacyTracker()
