"""Minimal Okapi BM25 over pre-tokenized documents."""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / len(corpus)) if corpus else 0.0
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        n = len(corpus)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for freqs, dl in zip(self.doc_freqs, self.doc_lens):
            score = 0.0
            for term in query:
                tf = freqs.get(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += self.idf[term] * tf * (self.k1 + 1) / denom
            out.append(score)
        return out
