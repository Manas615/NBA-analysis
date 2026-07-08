"""
Vector Store — ChromaDB integration for NBA knowledge retrieval.

Stores and retrieves NBA rules, salary cap regulations, CBA documents,
historical trades, and player contracts using semantic search.
"""

from __future__ import annotations

import os
from typing import Any

from observability.logging import get_logger

logger = get_logger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = "nba_knowledge"


class NBAVectorStore:
    """ChromaDB vector store for NBA knowledge base."""

    def __init__(self):
        self._client = None
        self._collection = None

    def _init_client(self):
        """Lazy-initialize ChromaDB client."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

            self._client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=CHROMA_PERSIST_DIR,
                anonymized_telemetry=False,
            ))

            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "vector_store_initialized",
                persist_dir=CHROMA_PERSIST_DIR,
                collection=COLLECTION_NAME,
            )
        except ImportError:
            logger.warning("chromadb_not_installed", message="pip install chromadb to enable RAG")
        except Exception as e:
            logger.error("vector_store_init_error", error=str(e))

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to the vector store."""
        self._init_client()
        if self._collection is None:
            return

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info("documents_added", count=len(documents))

    def query(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Query the vector store for relevant documents.

        Returns a list of dicts with:
            - document: the text content
            - metadata: associated metadata
            - distance: similarity distance (lower = more similar)
        """
        self._init_client()
        if self._collection is None:
            return []

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
            )

            docs = []
            for i in range(len(results["documents"][0])):
                docs.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })

            return docs
        except Exception as e:
            logger.error("vector_query_error", error=str(e))
            return []

    def get_count(self) -> int:
        """Get the number of documents in the collection."""
        self._init_client()
        if self._collection is None:
            return 0
        return self._collection.count()


# Global instance
_store: NBAVectorStore | None = None


def get_vector_store() -> NBAVectorStore:
    """Get the global vector store instance."""
    global _store
    if _store is None:
        _store = NBAVectorStore()
    return _store
