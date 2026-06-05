import numpy as np
from src.query import combine_query_embeddings, expand_query
from src.reranking import boost_by_summary, filter_by_tags, rerank
from src.retrieval import DocumentRetriever


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


def _extract_tags(query):
    from src.query import extract_tags

    return extract_tags(query)


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
        return rerank(candidates, query, importance_fn=importance_fn)

    def retrieve_top_k(self, query, k=None, importance_fn=None):
        if k is None:
            k = self.top_k
        candidates = self.retrieve_top_n(query, n=self.top_n)
        reranked = rerank(candidates, query, importance_fn=importance_fn)
        trimmed = reranked[:k]
        return [(c["text"], c["distance"], self._candidate_metadata(c)) for c in trimmed]

    @staticmethod
    def _candidate_metadata(candidate):
        return {
            "file_name": candidate.get("file_name", ""),
            "tags": candidate.get("tags", []) or [],
            "summary": candidate.get("summary", "") or "",
            "importance": float(candidate.get("importance", 0.0) or 0.0),
        }

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
            candidates = filter_by_tags(query, candidates, _extract_tags)
        if "summary_embedding" in self.features:
            candidates = boost_by_summary(query, candidates)
        return [(c["text"], c["distance"], self._candidate_metadata(c)) for c in candidates]


def build_pipeline(variant="V0", docs_path="docs/", cache_dir=".cache", **options):
    retriever = DocumentRetriever(docs_path=docs_path, cache_dir=cache_dir, **options)
    retriever.setup()
    return Pipeline(retriever, variant=variant)
