"""
Upload route — multi-dataset support with UUID-based session locking.

Session design:
  - session_id is read from the X-Session-ID request header.
  - The frontend generates this UUID on page load and stores it in
    sessionStorage (cleared on tab close). Every API call carries it.
  - All session state in MongoDB is keyed by session_id — not by a
    hardcoded user_id — so sessions are fully isolated per browser tab.
  - A MongoDB TTL index (24 h) on session_state.updated_at means sessions
    auto-expire; no stale state survives server restarts or long inactivity.

Upload rules:
  - Accepts 1–4 datasets per session (JSON array of file payloads).
  - Once a batch is uploaded the session is LOCKED (atomic find_one_and_update).
  - Subsequent upload attempts are rejected HTTP 409 until /reset-session is called.
  - Each file goes through three processing stages:
      1. Raw dataset stored in documents collection.
      2. Embeddings generated (or skipped when content hash unchanged).
      3. Schema profile computed and stored.

Endpoint: POST /upload-json
Body: {
  "files": [
    {"file_name": str, "data": list[dict]},
    ...
  ]
}
Also accepts legacy single-file format:
  {"file_name": str, "data": list[dict]}
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, Request

from embeddingclient import BedrockEmbeddingClient
from mongo_client import mongo_client as _mongo, _make_dataset_key
from semanticstore import process_dataset
from services.dataset_profiler import profile as build_profile

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_DATASETS_PER_SESSION = 4


# ---------------------------------------------------------------------------
# Session ID extraction
# ---------------------------------------------------------------------------

def _get_session_id(request: Request) -> str:
    """
    Extract the session UUID from the X-Session-ID header.
    Falls back to a generated UUID for API clients that omit the header
    (e.g. curl during development). Logs a warning so the omission is visible.
    """
    sid = request.headers.get("x-session-id", "").strip()
    if not sid:
        sid = str(uuid.uuid4())
        logger.warning(
            f"[UPLOAD] No X-Session-ID header — generated fallback session_id={sid!r}. "
            "The frontend should send this header on every request."
        )
    return sid


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@router.post("/upload-json")
async def upload_json(request: Request):
    session_id = _get_session_id(request)

    try:
        db = request.app.state.mongo.db

        body = await request.json()

        # ---- Normalise to list-of-files format (accept both legacy and new) ----
        if "files" in body:
            file_payloads: List[dict] = body["files"]
        elif "file_name" in body and "data" in body:
            file_payloads = [{"file_name": body["file_name"], "data": body["data"]}]
        else:
            raise HTTPException(
                status_code=400,
                detail="Request must contain 'files' (list) or 'file_name' + 'data'.",
            )

        # ---- Validate payload ----
        if not file_payloads:
            raise HTTPException(status_code=400, detail="No files provided.")

        if len(file_payloads) > MAX_DATASETS_PER_SESSION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Maximum {MAX_DATASETS_PER_SESSION} datasets per session. "
                    f"You sent {len(file_payloads)}."
                ),
            )

        for fp in file_payloads:
            if not fp.get("file_name"):
                raise HTTPException(status_code=400, detail="Each file must include 'file_name'.")
            if not fp.get("data") or not isinstance(fp["data"], list):
                raise HTTPException(
                    status_code=400,
                    detail=f"'data' for '{fp.get('file_name')}' must be a non-empty JSON array.",
                )

        # ---- Session lock check (read-then-check for early UX feedback) ----
        session = _mongo.get_session_state(session_id)
        if session.get("locked"):
            existing = session.get("active_datasets", [])
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Session is locked with {len(existing)} dataset(s): "
                    f"{', '.join(existing)}. "
                    "Call POST /reset-session to start a new analysis session."
                ),
            )

        # ---- Process each file ----
        processed_files = []
        warnings = []

        for fp in file_payloads:
            file_name     = fp["file_name"]
            original_data = fp["data"]

            result = await _process_single_file(
                db=db,
                file_name=file_name,
                original_data=original_data,
                mongo=request.app.state.mongo,
            )
            processed_files.append(result)

            if result.get("warning"):
                warnings.append(f"[{file_name}] {result['warning']}")

        # ---- Detect schema type mismatches and warn ----
        dataset_types = [r["dataset_type"] for r in processed_files if r.get("dataset_type")]
        if len(set(dataset_types)) > 1:
            warnings.append(
                f"Mixed dataset types detected: {', '.join(set(dataset_types))}. "
                "Cross-dataset metric comparisons may not be meaningful for all metrics."
            )

        # ---- Atomic session lock ----
        active_names = [r["file_name"] for r in processed_files]
        lock_acquired = _mongo.try_lock_session(session_id, active_names)
        if not lock_acquired:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Session was locked by a concurrent upload request. "
                    "Call POST /reset-session to start fresh."
                ),
            )

        # Invalidate stale cached results for this dataset combination
        _mongo.invalidate_cache_for_datasets(active_names)

        logger.info(
            f"[UPLOAD] session_id={session_id!r} locked with datasets: {active_names}"
        )

        return {
            "status":   "success",
            "message":  f"{len(processed_files)} dataset(s) uploaded and profiled successfully ✅",
            "datasets": processed_files,
            "warnings": warnings,
            "session":  {
                "session_id":      session_id,
                "active_datasets": active_names,
                "locked":          True,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPLOAD] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Reset-session endpoint
# ---------------------------------------------------------------------------

@router.post("/reset-session")
async def reset_session(request: Request):
    """
    Unlock the current session so a new set of datasets can be uploaded.
    Does NOT delete dataset documents or embeddings — only clears the session
    lock and cached query results for the current dataset set.
    """
    session_id = _get_session_id(request)
    try:
        _mongo.clear_session(session_id)
        logger.info(f"[UPLOAD] Session reset for session_id={session_id!r}")
        return {
            "status":  "success",
            "message": "Session reset. You can now upload new datasets.",
        }
    except Exception as e:
        logger.error(f"[UPLOAD] reset_session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Private: process a single file
# ---------------------------------------------------------------------------

def _hash_data(data: list) -> str:
    """
    Stable MD5 fingerprint of a dataset for embedding cache validation.
    MD5 is sufficient here — this is a content-equality check, not a security hash.
    Sorts row keys before hashing to be robust against column reordering.
    """
    canonical = json.dumps(
        [{str(k): v for k, v in sorted(row.items())} for row in data],
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.md5(canonical).hexdigest()


async def _process_single_file(
    db,
    file_name: str,
    original_data: list,
    mongo,
) -> dict:
    """
    Three-phase processing for one dataset file.
    Returns a summary dict. Never raises — errors are captured in 'warning' key.
    """
    warning = None

    # ---- Phase 1: Store raw dataset ----
    # Strip whitespace only — do NOT lowercase or replace spaces with underscores.
    # The profiler keyword-matching and ColumnMapper both use the original column
    # names as stored here. Normalising them (lower + snake_case) creates a
    # schema/data mismatch: profiler stores 'Expected Revenue (₹)' but the
    # DataFrame at query time would have 'expected_revenue_(₹)' → not found.
    # MongoDB handles any UTF-8 string (₹, £, €, spaces) as a document key.
    cleaned_data = [
        {
            str(k).strip(): v
            for k, v in row.items()
            if k and str(k).strip() and not str(k).strip().lower().startswith("unnamed:")
        }
        for row in original_data
    ]
    columns = list(cleaned_data[0].keys()) if cleaned_data else []

    db["documents"].delete_many({"file_name": file_name, "type": "dataset"})
    db["documents"].insert_one({
        "type":        "dataset",
        "file_name":   file_name,
        "columns":     columns,
        "data":        cleaned_data,
        "rows":        len(cleaned_data),
        "uploaded_at": datetime.utcnow(),
    })
    logger.info(f"[UPLOAD] [{file_name}] Stored {len(cleaned_data)} rows")

    # ---- Phase 2: Embeddings (skip if content unchanged) ----
    try:
        data_hash     = _hash_data(original_data)
        existing_meta = db["documents"].find_one(
            {"file_name": file_name, "type": "embedding_meta"},
            {"data_hash": 1},
        )

        if existing_meta and existing_meta.get("data_hash") == data_hash:
            logger.info(
                f"[UPLOAD] [{file_name}] Embeddings unchanged (hash match) — skipping re-embed"
            )
        else:
            logger.info(f"[UPLOAD] [{file_name}] Generating embeddings (hash changed or first upload)")
            embedding_client = BedrockEmbeddingClient()
            db["documents"].delete_many({"file_name": file_name, "type": "embedding"})
            process_dataset(
                data=cleaned_data,
                file_name=file_name,
                embedding_client=embedding_client,
                mongo_client=mongo,
            )
            db["documents"].replace_one(
                {"file_name": file_name, "type": "embedding_meta"},
                {
                    "type":       "embedding_meta",
                    "file_name":  file_name,
                    "data_hash":  data_hash,
                    "row_count":  len(original_data),
                    "updated_at": datetime.utcnow(),
                },
                upsert=True,
            )
            logger.info(f"[UPLOAD] [{file_name}] Embeddings generated and hash stored")
    except Exception as e:
        logger.error(f"[UPLOAD] [{file_name}] Embedding error: {e}")
        warning = f"Embedding generation failed: {e}. RAG fallback will be unavailable for this file."

    # ---- Phase 3: Schema profile ----
    schema = build_profile(cleaned_data, file_name)
    db["schema_profiles"].replace_one(
        {"file_name": file_name},
        {**schema, "created_at": datetime.utcnow()},
        upsert=True,
    )
    logger.info(
        f"[UPLOAD] [{file_name}] Schema profile stored — "
        f"type={schema.get('dataset_type')}, "
        f"metrics={schema.get('available_metrics')}"
    )

    return {
        "file_name":         file_name,
        "rows":              len(cleaned_data),
        "dataset_type":      schema.get("dataset_type"),
        "available_metrics": schema.get("available_metrics", []),
        "dimensions":        list(schema.get("dimension_map", {}).keys()),
        "warning":           warning,
    }
