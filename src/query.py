import re
import numpy as np


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "when", "where", "who", "whom", "which", "why", "how",
    "do", "does", "did", "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "so",
    "for", "to", "of", "in", "on", "at", "by", "from", "with",
    "as", "about", "into", "through", "during",
    "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "than", "too", "very", "can", "will", "just",
    "should", "would", "could", "may", "might", "must", "shall",
    "tell", "me", "please", "give", "get",
}

QUESTION_PREFIXES = (
    "what is", "what are", "what was", "what were",
    "how do", "how can", "how many", "how much", "how long",
    "when is", "when are", "when do", "when can",
    "where is", "where are", "where do", "where can",
    "why is", "why are", "why do", "why does",
    "who is", "who are", "which is", "which are",
    "tell me about",
)


def extract_tags(query):
    if not query:
        return []
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", query.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def detect_question_intent(query):
    lower = query.lower().strip()
    for prefix in QUESTION_PREFIXES:
        if lower.startswith(prefix):
            return prefix
    return ""


def expand_query(query):
    if not query or not query.strip():
        return [query] if query else []
    variants = [query]
    tags = extract_tags(query)
    if tags:
        variants.append(" ".join(tags))
        intent = detect_question_intent(query)
        if intent:
            variants.append(f"{intent} {' '.join(tags)}")
    return variants


def combine_query_embeddings(embed_model, variants):
    embeddings = [np.array(embed_model.get_text_embedding(v), dtype="float32")
                  for v in variants]
    if not embeddings:
        return np.zeros(0, dtype="float32")
    return np.mean(embeddings, axis=0)
