"""
Embedding Generation — OpenAI embeddings for the RAG pipeline.
"""

from __future__ import annotations

import os
from typing import Any

from observability.logging import get_logger

logger = get_logger(__name__)


def get_embedding(text: str) -> list[float] | None:
    """Generate an embedding for a text string using OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("embedding_error", error=str(e))
        return None


def get_embeddings_batch(texts: list[str]) -> list[list[float]] | None:
    """Generate embeddings for a batch of texts."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [d.embedding for d in response.data]
    except Exception as e:
        logger.error("batch_embedding_error", error=str(e))
        return None
