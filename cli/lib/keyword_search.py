import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stop_words, BM25_K1, BM25_B

from .text_utils import to_stem, to_token, preprocess_text
from .inverted_index import InvertedIndex
import sys


def remove_stop_words(stop_words: list[str], query: list[str]) -> list[str]:
    return [word for word in query if word not in stop_words]

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    try:
        idx = InvertedIndex()
        idx.load()
    except:
        print("Error: indexes does not exist")
        sys.exit(0)

    doc_ids = set()
    ordered_ids = []
    stop_words = load_stop_words()
    preprocessed_query = to_stem(remove_stop_words(stop_words, to_token(preprocess_text(query))))
    for token in preprocessed_query:
        for doc_id in sorted(idx.index.get(token, [])):

            if doc_id not in doc_ids:
                doc_ids.add(doc_id)
                ordered_ids.append(doc_id)
                if len(ordered_ids) >= limit:
                   break
        if len(ordered_ids) >= limit:
            break

    results = []
    for doc_id in ordered_ids:
        movie = idx.docmap.get(doc_id)
        if movie:
            print(f"{movie['title']} {movie['id']}")
            results.append(movie)
    return results

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float=BM25_K1, b: float=BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25_search_command(term: str, limit=5):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(term, limit)
