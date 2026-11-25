"""Local embedding and reranking models, run in-process with ONNX.

Two different kinds of model do two different jobs:

- The *bi-encoder* embeds the query and each chunk separately, so chunk
  vectors can be computed once and searched with a single matrix product.
- The *cross-encoder* reads the query and one chunk together. It is far more
  accurate and far too slow to run on every chunk, so it only reorders the
  shortlist that the cheap retrievers produce.

Models are downloaded on first use and cached by fastembed.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(RERANK_MODEL)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Unit-length vectors, so a dot product is cosine similarity."""
    return _normalize(np.array(list(_embedder().embed(texts)), dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    # BGE models expect a retrieval instruction on the query side only;
    # fastembed's query_embed adds it.
    return _normalize(np.array(next(iter(_embedder().query_embed(text))), dtype=np.float32))


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    return [float(score) for score in _reranker().rerank(query, texts)]
