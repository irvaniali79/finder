import re


_RERANK_SIM_WEIGHT = 0.7
_RERANK_IMPORTANCE_WEIGHT = 0.3
_RERANK_DIVERSITY_PENALTY = 0.1
_SUMMARY_BOOST = 0.05
_SUMMARY_BOOST_PER_HIT = 0.05
_SUMMARY_BOOST_MAX = 0.2


def query_word_set(query):
    return {
        w
        for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", query.lower())
        if len(w) > 1
    }


def summary_match_count(query, summary):
    if not summary:
        return 0
    qwords = query_word_set(query)
    if not qwords:
        return 0
    s_tokens = {
        w
        for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", summary.lower())
        if len(w) > 1
    }
    return len(qwords & s_tokens)


def filter_by_tags(query, candidates, extract_tags_fn):
    qtags = set(extract_tags_fn(query))
    if not qtags:
        return candidates
    return [c for c in candidates if set(c.get("tags", [])) & qtags]


def boost_by_summary(query, candidates):
    if not candidates:
        return candidates
    for c in candidates:
        hits = summary_match_count(query, c.get("summary", ""))
        boost = min(_SUMMARY_BOOST_MAX, _SUMMARY_BOOST + hits * _SUMMARY_BOOST_PER_HIT)
        c["distance"] = max(0.0, c["distance"] - boost)
    return candidates


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
