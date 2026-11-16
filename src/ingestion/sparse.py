"""
BM25 sparse (keyword) vectors for hybrid search.

Dense embeddings capture meaning but are weak on exact technical terms (clone
codes like "TV24", chemicals like "Hexaconazole", Latin names like "Exobasidium
vexans"). BM25 matches those words exactly. Each chunk gets a sparse vector:
    index = a stable hash of the word, value = BM25 term-frequency weight.
The other half of BM25 - IDF (rare words count more than common ones) - is
computed by Qdrant itself via the collection's `Modifier.IDF` setting, so it
always reflects the current collection.
"""
import hashlib
import re
from collections import Counter

from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector

# Standard BM25 parameters
K1 = 1.2   # how quickly repeated words stop adding weight
B = 0.75   # how much long chunks are penalised

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers him his how i if in into is it its itself just me more most my no nor not now
of off on once only or other our out over own same she should so some such than that the their them
then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours
""".split())


def _stem(word):
    """Very light stemming so 'drains'/'drain', 'infected'/'infect' match."""
    if len(word) <= 4 or word.isdigit():
        return word
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def tokenize(text):
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _token_id(token):
    """Stable 32-bit id for a word (same word -> same index in every run)."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


class BM25SparseEmbeddings(SparseEmbeddings):
    """
    avg_doc_length: average number of tokens per chunk in the collection, used by
    BM25's length normalisation. Only needed when embedding documents.
    """

    def __init__(self, avg_doc_length=150.0):
        self.avg_doc_length = avg_doc_length

    @staticmethod
    def average_length(texts):
        lengths = [len(tokenize(t)) for t in texts]
        return sum(lengths) / max(len(lengths), 1)

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            tokens = tokenize(text)
            counts = Counter(tokens)
            length_norm = 1 - B + B * len(tokens) / self.avg_doc_length
            weights = {}
            for token, tf in counts.items():
                weights[_token_id(token)] = tf * (K1 + 1) / (tf + K1 * length_norm)
            vectors.append(SparseVector(indices=list(weights), values=list(weights.values())))
        return vectors

    def embed_query(self, text):
        # Each distinct query word counts once; Qdrant applies IDF on top.
        ids = sorted({_token_id(t) for t in tokenize(text)})
        return SparseVector(indices=ids, values=[1.0] * len(ids))
