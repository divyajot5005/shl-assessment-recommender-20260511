from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

from app.catalog import CatalogRecord, normalize_text


@dataclass(slots=True)
class SearchCandidate:
    record: CatalogRecord
    score: float
    reasons: list[str] = field(default_factory=list)


class CatalogIndex:
    def __init__(self, records: list[CatalogRecord]) -> None:
        self.records = records
        self.search_texts = [normalize_text(record.search_text) for record in records]
        self.word_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(self.search_texts)
        self.char_matrix = self.char_vectorizer.fit_transform(self.search_texts)

        max_components = min(64, self.word_matrix.shape[0] - 1, self.word_matrix.shape[1] - 1)
        self.svd: TruncatedSVD | None = None
        self.dense_matrix: np.ndarray | None = None
        if max_components >= 2:
            self.svd = TruncatedSVD(n_components=max_components, random_state=42)
            dense_matrix = self.svd.fit_transform(self.word_matrix)
            self.dense_matrix = normalize(dense_matrix)

    def search(self, query: str, limit: int = 50) -> list[SearchCandidate]:
        normalized_query = normalize_text(query)
        if not normalized_query:
            return []

        word_query = self.word_vectorizer.transform([normalized_query])
        char_query = self.char_vectorizer.transform([normalized_query])

        word_scores = linear_kernel(word_query, self.word_matrix).ravel()
        char_scores = linear_kernel(char_query, self.char_matrix).ravel()
        scores = (0.5 * word_scores) + (0.25 * char_scores)

        if self.svd is not None and self.dense_matrix is not None:
            dense_query = normalize(self.svd.transform(word_query))
            dense_scores = dense_query @ self.dense_matrix.T
            scores += 0.25 * dense_scores.ravel()

        candidates: list[SearchCandidate] = []
        for index, record in enumerate(self.records):
            score = float(scores[index])
            reasons: list[str] = []
            if normalized_query in record.aliases:
                score += 1.25
                reasons.append("exact_alias")
            if normalized_query in normalize_text(record.name):
                score += 0.8
                reasons.append("name_overlap")
            if score > 0:
                candidates.append(SearchCandidate(record=record, score=score, reasons=reasons))

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:limit]
