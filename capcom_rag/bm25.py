"""Okapi BM25 keyword search, written out so the mechanics are visible.

BM25 scores a document by summing, over the query's terms, how rare the term
is in the corpus (IDF) times how often it appears in the document, with two
corrections: term frequency saturates (k1), and long documents are penalised
(b). It is the right tool for exact tokens like "1202" or "P64" that an
embedding model tends to blur.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.term_counts = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / max(len(documents), 1)

        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        n = len(documents)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        # Inverted index: only documents containing a query term get scored.
        self.postings: dict[str, list[int]] = {}
        for index, counts in enumerate(self.term_counts):
            for term in counts:
                self.postings.setdefault(term, []).append(index)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to k (document index, score) pairs, best first."""
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index in self.postings[term]:
                tf = self.term_counts[index][term]
                norm = 1 - self.b + self.b * self.lengths[index] / self.average_length
                scores[index] = scores.get(index, 0.0) + idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * norm
                )
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
