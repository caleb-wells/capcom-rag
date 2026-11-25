"""Build the index and search it: BM25, dense, hybrid, and hybrid + rerank."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import embed
from .bm25 import BM25
from .chunking import Chunk, chunk_utterances
from .corpus import Utterance

INDEX_DIR = Path(".index")
MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")

CANDIDATES = 30  # how many results each retriever contributes before fusion
RRF_K = 60  # the constant from the original RRF paper; larger flattens ranks


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # only comparable between hits from the same mode
    similarity: float  # cosine similarity to the query, comparable everywhere


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge ranked lists using ranks alone.

    BM25 scores and cosine similarities live on unrelated scales, so adding
    them needs calibration. RRF sidesteps that: each list gives a document
    1 / (k + rank), and documents ranked well by both retrievers win.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = BM25([chunk.search_text for chunk in chunks])

    # -- building and persistence -------------------------------------------

    @classmethod
    def build(cls, utterances: list[Utterance], **chunk_options) -> "Retriever":
        chunks = chunk_utterances(utterances, **chunk_options)
        vectors = embed.embed_documents([chunk.search_text for chunk in chunks])
        return cls(chunks, vectors)

    def save(self, directory: Path = INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        payload = [asdict(chunk) for chunk in self.chunks]
        (directory / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> "Retriever":
        payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**{**item, "timestamps": tuple(item["timestamps"])}) for item in payload]
        return cls(chunks, np.load(directory / "vectors.npy"))

    # -- search ---------------------------------------------------------------

    def search(self, query: str, k: int = 5, mode: str = "hybrid+rerank") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        similarities = self.vectors @ embed.embed_query(query)

        if mode == "bm25":
            ranked = self.bm25.search(query, k)
        elif mode == "dense":
            order = np.argsort(-similarities)[:k]
            ranked = [(int(i), float(similarities[i])) for i in order]
        else:
            keyword = [doc for doc, _ in self.bm25.search(query, CANDIDATES)]
            dense = [int(i) for i in np.argsort(-similarities)[:CANDIDATES]]
            ranked = reciprocal_rank_fusion([keyword, dense])
            if mode == "hybrid+rerank":
                shortlist = [doc for doc, _ in ranked[:CANDIDATES]]
                scores = embed.rerank_scores(
                    query, [self.chunks[doc].search_text for doc in shortlist]
                )
                ranked = sorted(zip(shortlist, scores), key=lambda item: -item[1])
            ranked = ranked[:k]

        return [Hit(self.chunks[doc], score, float(similarities[doc])) for doc, score in ranked]
