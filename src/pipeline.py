import re

import numpy as np
from retrieval import DocumentRetriever
from preprocessor import expand_query, combine_query_embeddings, extract_tags


VARIANT_FEATURES = {
    "V0": [],
    "V1": ["tag_filtering"],
    "V2": ["tag_filtering", "summary_embedding"],
    "V3": ["tag_filtering", "summary_embedding", "query_expansion"],
    "V4": ["tag_filtering", "summary_embedding", "query_expansion", "rerank"],
    "V5": [
        "tag_filtering",
        "summary_embedding",
        "query_expansion",
        "rerank",
        "importance",
    ],
}


_RERANK_SIM_WEIGHT = 0.7
_RERANK_IMPORTANCE_WEIGHT = 0.3
_RERANK_DIVERSITY_PENALTY = 0.1
_SUMMARY_BOOST = 0.05
_SUMMARY_BOOST_PER_HIT = 0.05
_SUMMARY_BOOST_MAX = 0.2


def _query_word_set(query):
    return {w for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", query.lower())
            if len(w) > 1}


def _summary_match_count(query, summary):
    if not summary:
        return 0
    qwords = _query_word_set(query)
    if not qwords:
        return 0
    s_tokens = {w for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", summary.lower())
                if len(w) > 1}
    return len(qwords & s_tokens)


class Pipeline:
    def __init__(self, retriever, variant="V0", top_k=5, top_n=20):
        if variant not in VARIANT_FEATURES:
            raise ValueError(
                f"Unknown variant {variant!r}. Valid variants: {sorted(VARIANT_FEATURES)}"
            )
        self.retriever = retriever
        self.variant = variant
        self.top_k = top_k
        self.top_n = top_n
        self.features = list(VARIANT_FEATURES[variant])

    def has(self, feature):
        return feature in self.features

    def _build_query_embedding(self, query):
        if "query_expansion" in self.features:
            variants = expand_query(query)
            return combine_query_embeddings(self.retriever.embed_model, variants)
        return np.array(
            self.retriever.embed_model.get_text_embedding(query), dtype="float32"
        )

    def _normalize_candidate(self, chunk, dist):
        return {
            "text": chunk.get("text", ""),
            "distance": float(dist),
            "file_name": chunk.get("file_name", ""),
            "tags": chunk.get("tags", []) or [],
            "summary": chunk.get("summary", "") or "",
            "importance": float(chunk.get("importance", 0.0) or 0.0),
        }

    def _search_candidates(self, query_emb):
        if (
            self.retriever.index is None
            or self.retriever.chunk_store is None
            or self.retriever.chunk_store.count() == 0
        ):
            return []
        query_vector = np.array([query_emb]).astype("float32")
        distances, indices = self.retriever.index.search(query_vector, self.top_k)
        ids = [int(i) for i in indices[0] if int(i) >= 0]
        chunks = self.retriever.chunk_store.get_many(ids)
        candidates = []
        for chunk, dist in zip(chunks, distances[0]):
            if chunk is not None:
                candidates.append(self._normalize_candidate(chunk, dist))
        return candidates

    def _filter_by_tags(self, query, candidates):
        qtags = set(extract_tags(query))
        if not qtags:
            return candidates
        return [c for c in candidates if set(c.get("tags", [])) & qtags]

    def _boost_by_summary(self, query, candidates):
        if not candidates:
            return candidates
        for c in candidates:
            hits = _summary_match_count(query, c.get("summary", ""))
            boost = min(_SUMMARY_BOOST_MAX, _SUMMARY_BOOST + hits * _SUMMARY_BOOST_PER_HIT)
            c["distance"] = max(0.0, c["distance"] - boost)
        return candidates

    def retrieve_top_n(self, query, n=None):
        if n is None:
            n = self.top_n
        if (
            self.retriever.index is None
            or self.retriever.chunk_store is None
            or self.retriever.chunk_store.count() == 0
        ):
            return []
        query_emb = self._build_query_embedding(query)
        query_vector = np.array([query_emb]).astype("float32")
        distances, indices = self.retriever.index.search(query_vector, n)
        ids = [int(i) for i in indices[0] if int(i) >= 0]
        chunks = self.retriever.chunk_store.get_many(ids)
        candidates = []
        for chunk, dist in zip(chunks, distances[0]):
            if chunk is not None:
                candidates.append(self._normalize_candidate(chunk, dist))
        return candidates

    @staticmethod
    def rerank(candidates, query, importance_fn=None):
        if not candidates:
            return []
        distances = [c["distance"] for c in candidates]
        max_dist = max(distances)
        min_dist = min(distances)
        spread = max(max_dist - min_dist, 1e-9)
        norm_sim = [(max_dist - d) / spread for d in distances]
        scored = []
        file_seen_count = {}
        for i, c in enumerate(candidates):
            importance = importance_fn(c) if importance_fn is not None else 0.0
            file_seen_count[c["file_name"]] = file_seen_count.get(c["file_name"], 0) + 1
            penalty = _RERANK_DIVERSITY_PENALTY * (file_seen_count[c["file_name"]] - 1)
            score = (
                _RERANK_SIM_WEIGHT * norm_sim[i]
                + _RERANK_IMPORTANCE_WEIGHT * importance
                - penalty
            )
            scored.append((score, i, c))
        scored.sort(key=lambda x: -x[0])
        return [item[2] for item in scored]

    def retrieve_top_k(self, query, k=None, importance_fn=None):
        if k is None:
            k = self.top_k
        candidates = self.retrieve_top_n(query, n=self.top_n)
        reranked = self.rerank(candidates, query, importance_fn=importance_fn)
        trimmed = reranked[:k]
        return [(c["text"], c["distance"]) for c in trimmed]

    def _build_importance_fn(self, user_fn):
        def _fn(candidate):
            base = float(candidate.get("importance", 0.0) or 0.0)
            if user_fn is None:
                return base
            return base + float(user_fn(candidate) or 0.0)
        return _fn

    def retrieve(self, query, importance_fn=None):
        if "rerank" in self.features:
            wrapped = self._build_importance_fn(importance_fn)
            return self.retrieve_top_k(query, importance_fn=wrapped)
        query_emb = self._build_query_embedding(query)
        candidates = self._search_candidates(query_emb)
        if "tag_filtering" in self.features:
            candidates = self._filter_by_tags(query, candidates)
        if "summary_embedding" in self.features:
            candidates = self._boost_by_summary(query, candidates)
        return [(c["text"], c["distance"]) for c in candidates]


def build_pipeline(variant="V0", docs_path="docs/", cache_dir=".cache", **options):
    retriever = DocumentRetriever(docs_path=docs_path, cache_dir=cache_dir, **options)
    retriever.setup()
    return Pipeline(retriever, variant=variant)
