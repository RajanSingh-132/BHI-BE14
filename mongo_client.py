"""
MongoDB client — single module-level singleton.

Session management design (v2 — UUID-based):
  - All session state is keyed by session_id (UUID), not by a hardcoded user_id.
  - session_state collection has a TTL index on updated_at (24 h). Sessions
    auto-expire; no manual cleanup is required and no stale state survives
    across server restarts or extended inactivity.
  - ensure_ttl_index() is called at startup (idempotent — safe to call repeatedly).

Cache design:
  - Cache key: _make_dataset_key(file_names) — sorted "|"-joined names.
  - Cache is global per (dataset_key, query) — shared across sessions that
    happen to have the same datasets. invalidate_cache_for_datasets() is
    called on upload to bust stale entries.
  - save_result uses replace_one (upsert=True) — one document per key, no
    non-deterministic duplicates.
"""

import logging
import os

import certifi
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# print("MONGO_URI:", os.getenv("MONGO_URI"))
print("DB_NAME:", os.getenv("DB_NAME"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_SESSION_TTL_SECONDS = 86_400   # 24 hours


class MongoDBClient:

    def __init__(self, uri: str = None, db_name: str = None):
        self.uri     = uri     or os.getenv("MONGO_URI")
        self.db_name = db_name or os.getenv("DB_NAME")
        self.client  = None
        self.db      = None
        self.collection         = None   # documents
        self.results_collection = None   # cached query results
        self.connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self):
        try:
            logger.info("[MONGO] Connecting to MongoDB…")
            kwargs = {
                "retryWrites": True,
                "retryReads": True,
                "serverSelectionTimeoutMS": 20_000,
            }
            if self.uri and "localhost" not in self.uri:
                kwargs["tls"] = True
                kwargs["tlsCAFile"] = certifi.where()
            elif not self.uri:
                # If uri is None, pymongo connects to localhost by default
                pass

            self.client = MongoClient(
                self.uri,
                **kwargs
            )
            self.client.admin.command("ping")
            self.db                 = self.client[self.db_name]
            self.collection         = self.db["documents"]
            self.results_collection = self.db["results"]
            logger.info("[MONGO] Connected ✅")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"[MONGO] Connection failed: {e}")
            raise

    # ------------------------------------------------------------------
    # TTL index — call once at startup (idempotent)
    # ------------------------------------------------------------------
    def ensure_ttl_index(self) -> None:
        """
        Create a TTL index on session_state.updated_at so that sessions
        automatically expire after _SESSION_TTL_SECONDS (24 h).

        MongoDB honours only one TTL index per collection.  create_index is
        idempotent when called with the same parameters — safe to call on
        every server startup.
        """
        try:
            self.db["session_state"].create_index(
                [("updated_at", ASCENDING)],
                expireAfterSeconds=_SESSION_TTL_SECONDS,
                name="session_ttl",
            )
            logger.info(
                f"[MONGO] session_state TTL index ensured "
                f"(expiry={_SESSION_TTL_SECONDS}s)"
            )
        except Exception as e:
            # Non-fatal: log and continue.  Missing TTL means sessions won't
            # auto-expire but the system still functions correctly.
            logger.warning(f"[MONGO] Could not create TTL index: {e}")

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------
    def insert_documents(self, documents: list) -> bool:
        try:
            if not documents:
                logger.warning("[MONGO] No documents to insert")
                return False
            result = self.collection.insert_many(documents)
            logger.info(f"[MONGO] Inserted {len(result.inserted_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"[MONGO] Insert error: {e}")
            return False

    # ------------------------------------------------------------------
    # Cache — save result (upsert)
    # ------------------------------------------------------------------
    def save_result(self, result_data: dict) -> bool:
        """
        Upsert a query result into the cache.

        Key: (dataset_key, query) — both normalised to lowercase + stripped.

        Using replace_one with upsert=True guarantees there is never more
        than one document per (dataset_key, query) pair.
        """
        try:
            if not result_data.get("answer"):
                logger.warning("[MONGO] Empty answer — skipping cache save")
                return False

            dataset_key = result_data.get("dataset_key") or result_data.get("file_name", "")
            query       = result_data.get("query", "").strip().lower()

            if not dataset_key or not query:
                logger.warning("[MONGO] Missing dataset_key or query — skipping cache save")
                return False

            doc = {
                "dataset_key": dataset_key,
                "file_name":   result_data.get("file_name", dataset_key),
                "query":       query,
                "answer":      result_data["answer"],
                "kpis":        result_data.get("kpis", []),
                "timestamp":   __import__("datetime").datetime.utcnow(),
            }

            self.results_collection.replace_one(
                {"dataset_key": dataset_key, "query": query},
                doc,
                upsert=True,
            )
            logger.info(f"[MONGO] Cache saved — dataset_key={dataset_key!r}, query={query!r}")
            return True

        except Exception as e:
            logger.error(f"[MONGO] save_result error: {e}")
            return False

    # ------------------------------------------------------------------
    # Cache — lookup
    # ------------------------------------------------------------------
    def get_cached_result(self, dataset_key: str, query: str) -> dict | None:
        """
        Look up a cached result by (dataset_key, query).
        dataset_key is a sorted "|"-joined string of active file_names.
        """
        try:
            return self.results_collection.find_one({
                "dataset_key": dataset_key,
                "query":       query.strip().lower(),
            })
        except Exception as e:
            logger.error(f"[MONGO] get_cached_result error: {e}")
            return None

    # ------------------------------------------------------------------
    # Session state — UUID-keyed, TTL-managed
    # ------------------------------------------------------------------
    def get_session_state(self, session_id: str) -> dict:
        """
        Returns the current session state for session_id.

        Shape: {
          "session_id":     str,
          "active_datasets": list[str],
          "locked":          bool,
        }
        Returns a default (unlocked, empty) state if no document exists.
        This means a session not found in MongoDB == no dataset uploaded yet,
        which is the correct post-restart behaviour.
        """
        if not session_id:
            return {"session_id": session_id, "active_datasets": [], "locked": False}
        try:
            doc = self.db["session_state"].find_one({"session_id": session_id})
            if doc:
                doc.pop("_id", None)
                return doc
        except Exception as e:
            logger.error(f"[MONGO] get_session_state error: {e}")
        return {"session_id": session_id, "active_datasets": [], "locked": False}

    def try_lock_session(self, session_id: str, active_datasets: list) -> bool:
        """
        Atomically lock the session if it is not already locked.

        Uses findOneAndUpdate with a filter on locked=False (or non-existent)
        so two concurrent upload requests cannot both proceed.

        Returns True if the lock was acquired, False if already locked.
        """
        try:
            result = self.db["session_state"].find_one_and_update(
                {
                    "session_id": session_id,
                    "$or": [{"locked": False}, {"locked": {"$exists": False}}],
                },
                {
                    "$set": {
                        "session_id":      session_id,
                        "active_datasets": active_datasets,
                        "locked":          True,
                        "updated_at":      __import__("datetime").datetime.utcnow(),
                    }
                },
                upsert=True,
                return_document=True,
            )
            return result is not None
        except Exception as e:
            logger.error(f"[MONGO] try_lock_session error: {e}")
            return False

    def invalidate_cache_for_datasets(self, file_names: list) -> int:
        """
        Remove all cached query results for a given dataset combination.
        Called on upload so stale cached answers are not served after data changes.
        Returns the count of deleted documents.
        """
        try:
            dataset_key = _make_dataset_key(file_names)
            result = self.results_collection.delete_many({"dataset_key": dataset_key})
            deleted = result.deleted_count
            if deleted:
                logger.info(
                    f"[MONGO] Cache invalidated — {deleted} result(s) removed "
                    f"for dataset_key={dataset_key!r}"
                )
            return deleted
        except Exception as e:
            logger.error(f"[MONGO] invalidate_cache_for_datasets error: {e}")
            return 0

    def clear_session(self, session_id: str) -> bool:
        """
        Unlock the session and remove all associated cached results.
        Called by the /reset-session endpoint.
        Does NOT delete dataset documents or embeddings.
        """
        try:
            state      = self.get_session_state(session_id)
            active     = state.get("active_datasets", [])
            dataset_key = _make_dataset_key(active)

            if dataset_key:
                self.results_collection.delete_many({"dataset_key": dataset_key})

            self.db["session_state"].delete_one({"session_id": session_id})

            logger.info(f"[MONGO] Session cleared for session_id={session_id!r}")
            return True
        except Exception as e:
            logger.error(f"[MONGO] clear_session error: {e}")
            return False

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------
    def vector_search(
        self,
        query_embedding,
        limit: int = 5,
        similarity_threshold: float = 0.65,
        metadata_filters: dict = None,
    ) -> list:
        try:
            logger.info("[MONGO] Running manual cosine similarity search")
            docs      = list(self.collection.find(metadata_filters or {}))
            query_vec = np.array(query_embedding)
            results   = []

            for doc in docs:
                embedding = doc.get("embedding")
                if not embedding:
                    continue
                doc_vec    = np.array(embedding)
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                )
                if similarity >= similarity_threshold:
                    doc["score"] = float(similarity)
                    results.append(doc)

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"[MONGO] vector_search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------
    def close(self):
        if self.client:
            self.client.close()
            logger.info("[MONGO] Connection closed")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_dataset_key(file_names: list) -> str:
    """
    Produce a stable, order-independent cache key from a list of file names.
    e.g. ["B.xlsx", "A.xlsx"] → "A.xlsx|B.xlsx"
    """
    return "|".join(sorted(file_names))


# ---------------------------------------------------------------------------
# Singleton instance — import this everywhere
# ---------------------------------------------------------------------------
mongo_client = MongoDBClient()
