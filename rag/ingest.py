"""
Document Ingestion — Load NBA rules and CBA documents into the vector store.

Reads markdown documents from rag/documents/ and ingests them
into ChromaDB for semantic retrieval.
"""

from __future__ import annotations

import os
from pathlib import Path

from rag.vector_store import get_vector_store
from observability.logging import get_logger

logger = get_logger(__name__)

DOCUMENTS_DIR = Path(__file__).parent / "documents"


def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split a document into overlapping chunks for embedding."""
    lines = text.split("\n")
    chunks = []
    current_chunk: list[str] = []
    current_len = 0

    for line in lines:
        current_chunk.append(line)
        current_len += len(line)

        if current_len >= chunk_size:
            chunks.append("\n".join(current_chunk))
            # Keep last few lines for overlap
            overlap_lines = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk[-1:]
            current_chunk = overlap_lines
            current_len = sum(len(l) for l in current_chunk)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def ingest_documents() -> int:
    """
    Ingest all markdown documents from rag/documents/ into ChromaDB.

    Returns the total number of chunks ingested.
    """
    store = get_vector_store()

    if not DOCUMENTS_DIR.exists():
        logger.warning("documents_dir_not_found", path=str(DOCUMENTS_DIR))
        return 0

    total_chunks = 0

    for doc_path in sorted(DOCUMENTS_DIR.glob("*.md")):
        logger.info("ingesting_document", file=doc_path.name)

        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_document(text)

        ids = [f"{doc_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": doc_path.name, "chunk_index": i}
            for i in range(len(chunks))
        ]

        store.add_documents(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

        total_chunks += len(chunks)
        logger.info("document_ingested", file=doc_path.name, chunks=len(chunks))

    logger.info("ingestion_complete", total_chunks=total_chunks)
    return total_chunks


def search_nba_rules(query: str, n_results: int = 3) -> list[dict]:
    """Search NBA rules and CBA documents for relevant context."""
    store = get_vector_store()
    return store.query(query, n_results=n_results)
