import numpy as np
from retrieval import DocumentRetriever
from preprocessor import expand_query, combine_query_embeddings


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

    def _search(self, query_emb):
        query_vector = np.array([query_emb]).astype("float32")
        distances, indices = self.retriever.index.search(query_vector, self.top_k)
        ids = [int(i) for i in indices[0]]
        chunks = self.retriever.chunk_store.get_many(ids)
        results = []
        for chunk, dist in zip(chunks, distances[0]):
            if chunk is not None:
                results.append((chunk["text"], float(dist)))
        return results

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
                candidates.append(
                    {
                        "text": chunk["text"],
                        "distance": float(dist),
                        "file_name": chunk.get("file_name", ""),
                    }
                )
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

    def retrieve(self, query, importance_fn=None):
        if "rerank" in self.features:
            return self.retrieve_top_k(query, importance_fn=importance_fn)
        if "summary_embedding" in self.features or "tag_filtering" in self.features:
            raise NotImplementedError(
                "Per-chunk metadata extraction is implemented in Feature 13."
            )
        query_emb = self._build_query_embedding(query)
        return self._search(query_emb)


def build_pipeline(variant="V0", docs_path="docs/", cache_dir=".cache", **options):
    retriever = DocumentRetriever(docs_path=docs_path, cache_dir=cache_dir, **options)
    retriever.setup()
    return Pipeline(retriever, variant=variant)
