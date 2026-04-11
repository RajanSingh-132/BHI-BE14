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


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    try:
        tracker.api_hit()

        if not req.chat_history:
            raise HTTPException(status_code=400, detail="Chat history empty")

        user_message = req.chat_history[-1].content.strip()
        session_id   = _get_session_id(request)

        # Pass all messages except the current one as conversation context.
        conversation_history = req.chat_history[:-1]

        # File upload acknowledgement — client sends this magic string after
        # a successful upload so the UI shows a confirmation message.
        if "file uploaded" in user_message.lower():
            return {
                "answer": "File uploaded successfully ✅\n\n👉 Now ask your question from the dataset.",
                "kpis":   [],
                "charts": [],
            }

        result = generate_ai_response(
            session_id=session_id,
            message=user_message,
            history=conversation_history,
            request=request,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CHAT] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat service unavailable")
