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

    def retrieve(self, query):
        if "rerank" in self.features:
            raise NotImplementedError(
                "Two-stage retrieval is implemented in Feature 12; rerank requires it."
            )
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
