"""
RAGRetriever — semantic document retrieval for the RAG fallback path.

Fixes applied:
  1. Always filters by type="embedding" — skips raw dataset documents.
  2. Accepts optional file_name to scope search to one dataset's embeddings,
     reducing memory load significantly on multi-dataset deployments.
  3. Warns when the working set is large (>2000 docs) — a signal to add an
     Atlas Vector Search index when moving off the free tier.
  4. Exposes retrieve() dict method alongside get_relevant_documents() list method.
  5. Uses the module-level MongoDBClient singleton (no new connection pool).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from langchain_core.documents import Document

from mongo_client import mongo_client as _mongo_singleton

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_LARGE_COLLECTION_THRESHOLD = 2000   # warn above this doc count


class RAGRetriever:

    def __init__(self):
        from embeddingclient import BedrockEmbeddingClient

        self.embedding_client    = BedrockEmbeddingClient()
        # Reuse the module-level singleton — no extra connection pool.
        self.mongo_client        = _mongo_singleton
        self.collection          = self.mongo_client.collection
        self.similarity_threshold = 0.65
        self.max_results          = 100

        logger.info("[RAG_RETRIEVER] Initialised")

    # ------------------------------------------------------------------
    # Primary retrieval method — returns LangChain Document objects
    # ------------------------------------------------------------------

    def get_relevant_documents(
        self,
        query:     str,
        file_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve semantically similar documents for `query`.

        Args:
            query:     Natural language query string.
            file_name: If supplied, restrict search to embeddings from this
                       dataset only. Pass None to search all datasets.

        Returns:
            List of LangChain Document objects ordered by descending similarity.
        """
        try:
            logger.info(f"[RAG_RETRIEVER] Query='{query}', file_name={file_name!r}")

            # 1. Embed the query
            query_embedding = self.embedding_client.generate_embedding(query)
            if not query_embedding:
                logger.warning("[RAG_RETRIEVER] Embedding returned empty — returning []")
                return []

            # 2. Build MongoDB filter
            #    Always restrict to embeddings only — never load raw dataset docs.
            mongo_filter: Dict = {"type": "embedding"}
            if file_name:
                mongo_filter["file_name"] = file_name

            # 3. Fetch candidate docs
            docs = list(self.collection.find(mongo_filter))

            if not docs:
                logger.warning(
                    f"[RAG_RETRIEVER] No embedding documents found "
                    f"(filter={mongo_filter})"
                )
                return []

            if len(docs) > _LARGE_COLLECTION_THRESHOLD:
                logger.warning(
                    f"[RAG_RETRIEVER] Large working set: {len(docs)} docs loaded into memory "
                    f"for cosine search. Consider adding an Atlas Vector Search index "
                    f"when upgrading from the free tier."
                )

            # 4. Compute cosine similarity for every candidate
            query_vec = np.array(query_embedding, dtype=np.float64)
            q_norm    = np.linalg.norm(query_vec)

            if q_norm == 0:
                logger.warning("[RAG_RETRIEVER] Zero-norm query vector — returning []")
                return []

            scored: List[Dict] = []

            for doc in docs:
                embedding = doc.get("embedding")
                if not embedding:
                    continue

                doc_vec  = np.array(embedding, dtype=np.float64)
                d_norm   = np.linalg.norm(doc_vec)
                if d_norm == 0:
                    continue

                similarity = float(np.dot(query_vec, doc_vec) / (q_norm * d_norm))
                doc["_score"] = similarity
                scored.append(doc)

            # 5. Sort descending, apply threshold
            scored.sort(key=lambda x: x["_score"], reverse=True)
            filtered = [d for d in scored if d["_score"] >= self.similarity_threshold]

            if not filtered:
                logger.info(
                    f"[RAG_RETRIEVER] No docs above threshold={self.similarity_threshold}. "
                    f"Returning top-{self.max_results} regardless."
                )
                filtered = scored[: self.max_results]
            else:
                filtered = filtered[: self.max_results]

            logger.info(
                f"[RAG_RETRIEVER] {len(filtered)} docs returned "
                f"(top score={filtered[0]['_score']:.3f} if filtered else 0)"
            )

            # 6. Convert to LangChain Documents
            return [
                Document(
                    page_content=doc.get("content", ""),
                    metadata={
                        **doc.get("data", {}),
                        "_score":     doc["_score"],
                        "_file_name": doc.get("file_name", ""),
                    },
                )
                for doc in filtered
            ]

        except Exception as e:
            logger.error(f"[RAG_RETRIEVER] Error: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Dict-format retrieval (used by rag_engine.py)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query:     str,
        file_name: Optional[str] = None,
    ) -> Dict:
        """
        Convenience wrapper that returns a dict instead of Document list.

        Returns:
            {"chunks": [str], "structured_data": [dict]}
        """
        docs = self.get_relevant_documents(query, file_name=file_name)
        return {
            "chunks":          [d.page_content for d in docs],
            "structured_data": [d.metadata     for d in docs],
        }

    # ------------------------------------------------------------------
    # Async shim (keeps compatibility with any async callers)
    # ------------------------------------------------------------------

    async def aget_relevant_documents(
        self,
        query:     str,
        file_name: Optional[str] = None,
    ) -> List[Document]:
        return self.get_relevant_documents(query, file_name=file_name)
