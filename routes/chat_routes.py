"""
Chat route.

Session contract:
  - Reads session_id from the X-Session-ID request header.
  - Passes session_id (not a hardcoded user_id) to generate_ai_response.
  - The frontend must generate a UUID on page load (sessionStorage) and
    attach it as X-Session-ID on every request.
"""

import uuid
import logging

from fastapi import APIRouter, HTTPException, Request

from models import ChatRequest
from utils.request_tracker import tracker
from services.ai_services import generate_ai_response

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_session_id(request: Request) -> str:
    """
    Extract the session UUID from the X-Session-ID header.
    Generates a fallback UUID when the header is absent (e.g. curl / tests).
    """
    sid = request.headers.get("x-session-id", "").strip()
    if not sid:
        sid = str(uuid.uuid4())
        logger.warning(
            f"[CHAT] No X-Session-ID header — generated fallback session_id={sid!r}. "
            "The frontend should send this header on every request."
        )
    return sid


@router.post("/leads-chat")
async def leads_chat(req: ChatRequest, request: Request):
    try:
        tracker.api_hit()
        if not req.chat_history:
            raise HTTPException(status_code=400, detail="Chat history empty")
        user_message = req.chat_history[-1].content.strip()
        session_id   = _get_session_id(request)
        # Force context to leads
        context      = "leads"
        conversation_history = req.chat_history[:-1]
        result = generate_ai_response(
            session_id=session_id,
            message=user_message,
            history=conversation_history,
            request=request,
            context=context,
            dashboard_summary=req.dashboard_summary,
            chat_mode=True
        )
        return result
    except Exception as e:
        logger.error(f"[LEADS-CHAT] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat service unavailable")


@router.post("/sales-chat")
async def sales_chat(req: ChatRequest, request: Request):
    try:
        tracker.api_hit()
        if not req.chat_history:
            raise HTTPException(status_code=400, detail="Chat history empty")
        user_message = req.chat_history[-1].content.strip()
        session_id   = _get_session_id(request)
        # Force context to Sales
        context      = "Sales"
        conversation_history = req.chat_history[:-1]
        result = generate_ai_response(
            session_id=session_id,
            message=user_message,
            history=conversation_history,
            request=request,
            context=context,
            dashboard_summary=req.dashboard_summary,
            chat_mode=True
        )
        return result
    except Exception as e:
        logger.error(f"[SALES-CHAT] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat service unavailable")


@router.post("/productivity-chat")
async def productivity_chat(req: ChatRequest, request: Request):
    try:
        tracker.api_hit()
        if not req.chat_history:
            raise HTTPException(status_code=400, detail="Chat history empty")
        user_message = req.chat_history[-1].content.strip()
        session_id   = _get_session_id(request)
        # Force context to Productivity
        context      = "Productivity"
        conversation_history = req.chat_history[:-1]
        result = generate_ai_response(
            session_id=session_id,
            message=user_message,
            history=conversation_history,
            request=request,
            context=context,
            dashboard_summary=req.dashboard_summary,
            chat_mode=True
        )
        return result
    except Exception as e:
        logger.error(f"[PROD-CHAT] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat service unavailable")


@router.post("/summary-chat")
async def summary_chat(req: ChatRequest, request: Request):
    try:
        tracker.api_hit()
        if not req.chat_history:
            raise HTTPException(status_code=400, detail="Chat history empty")
        user_message = req.chat_history[-1].content.strip()
        session_id   = _get_session_id(request)
        # Force context to Summary
        context      = "Summary"
        conversation_history = req.chat_history[:-1]
        result = generate_ai_response(
            session_id=session_id,
            message=user_message,
            history=conversation_history,
            request=request,
            context=context,
            dashboard_summary=req.dashboard_summary,
            chat_mode=True
        )
        return result
    except Exception as e:
        logger.error(f"[SUMMARY-CHAT] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat service unavailable")
