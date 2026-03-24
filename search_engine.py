import json
import math
import re
import time
from collections import defaultdict

# ── Stop words básicas en inglés ──────────────────────────────────────────────
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "as",
    "not", "also", "such", "which", "when", "there", "their", "than", "up",
    "if", "so", "no", "more", "about", "one", "two", "after", "into", "over"
}

# ── Stemmer muy simple (sufijos comunes en inglés) ────────────────────────────
def simple_stem(word):
    suffixes = ["ing", "tion", "ed", "er", "ly", "es", "s"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# ── Tokenización ──────────────────────────────────────────────────────────────
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [simple_stem(t) for t in tokens]
    return tokens


# ── Motor de búsqueda ─────────────────────────────────────────────────────────
class SearchEngine:
    def __init__(self, corpus_path="corpus.json"):
        self.documents = []
        self.index = defaultdict(dict)      # term -> {doc_id: freq}
        self.doc_lengths = {}               # doc_id -> num tokens
        self.avg_dl = 0
        self.vocab = set()

        # BM25 params
        self.k1 = 1.5
        self.b = 0.75

        self._load_and_index(corpus_path)

    # ── Carga e indexa el corpus ──────────────────────────────────────────────
    def _load_and_index(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        total_length = 0
        for doc in self.documents:
            doc_id = doc["id"]
            tokens = tokenize(doc["title"] + " " + doc["text"])
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            freq_map = defaultdict(int)
            for token in tokens:
                freq_map[token] += 1
                self.vocab.add(token)

            for term, freq in freq_map.items():
                self.index[term][doc_id] = freq

        self.avg_dl = total_length / len(self.documents) if self.documents else 1

    # ── BM25 ──────────────────────────────────────────────────────────────────
    def _bm25_score(self, term, doc_id):
        N = len(self.documents)
        df = len(self.index.get(term, {}))
        if df == 0:
            return 0.0

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        tf = self.index[term].get(doc_id, 0)
        dl = self.doc_lengths.get(doc_id, 0)
        tf_norm = (tf * (self.k1 + 1)) / (
            tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
        )
        return idf * tf_norm

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    def _tfidf_score(self, term, doc_id):
        N = len(self.documents)
        df = len(self.index.get(term, {}))
        if df == 0:
            return 0.0
        tf = self.index[term].get(doc_id, 0)
        if tf == 0:
            return 0.0
        idf = math.log(N / df)
        return tf * idf

    # ── Búsqueda principal ────────────────────────────────────────────────────
    def search(self, query, top_k=10):
        start = time.time()
        terms = tokenize(query)

        # Candidatos: docs que contienen al menos un término
        candidate_ids = set()
        for term in terms:
            candidate_ids.update(self.index.get(term, {}).keys())

        # Calcular scores BM25 y TF-IDF
        bm25_scores = {}
        tfidf_scores = {}
        for doc_id in candidate_ids:
            bm25_scores[doc_id] = sum(self._bm25_score(t, doc_id) for t in terms)
            tfidf_scores[doc_id] = sum(self._tfidf_score(t, doc_id) for t in terms)

        # Ordenar por BM25
        ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Armar resultados
        doc_map = {doc["id"]: doc for doc in self.documents}
        results = []
        for doc_id, bm25 in ranked:
            doc = doc_map[doc_id]
            results.append({
                "id": doc_id,
                "title": doc["title"],
                "text": doc["text"],
                "source": doc.get("source", ""),
                "bm25_score": round(bm25, 4),
                "tfidf_score": round(tfidf_scores.get(doc_id, 0), 4),
            })

        elapsed = round(time.time() - start, 4)
        return results, elapsed, terms

    # ── Estadísticas ──────────────────────────────────────────────────────────
    def stats(self):
        return {
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vocab),
            "avg_doc_length": round(self.avg_dl, 1),
        }

    # ── Autocomplete (sugerencias de términos) ────────────────────────────────
    def autocomplete(self, prefix, limit=6):
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        return [w for w in sorted(self.vocab) if w.startswith(prefix)][:limit]
