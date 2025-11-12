from .text_utils import to_token, to_stem, preprocess_text
from collections import defaultdict
from collections import Counter
import string
import pickle
import os
import math
from .search_utils import BM25_K1, BM25_B, CACHE_DIR
class InvertedIndex:
    def __init__(self, index=None, docmap=None, term_frequencies=None, doc_lengths=None):
        self.index = index if index is not None else defaultdict(set)
        self.docmap = docmap if docmap is not None else {}
        self.term_frequencies = term_frequencies if term_frequencies is not None else defaultdict(Counter)
        self.doc_lengths = doc_lengths if doc_lengths is not None else {}
    def __add_document(self, doc_id, text):
        tokens = to_stem(to_token(preprocess_text(text)))
        tokens_total = len(tokens)
        self.doc_lengths[doc_id] = tokens_total
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        tokens = to_stem(to_token(preprocess_text(term)))
        if not tokens:
            return []
        token = tokens[0]
        doc_ids = self.index.get(token, set())
        return sorted(doc_ids)

    def build(self, movies):
        for movie in movies:
            doc_id = movie.get('id')
            if doc_id is None:
                raise ValueError("Movie missing 'id' field")
            title = movie.get('title', '')
            description = movie.get('description', '')
            if not title and not description:
                continue
            self.docmap[doc_id] = movie 
            text = f"{title} {description}"
            self.__add_document(doc_id, text)

    def save(self):
        if not os.path.exists('cache'):
            os.makedirs('cache', exist_ok=True)
        with open('cache/index.pkl', 'wb') as f:
            pickle.dump(self.index, f)
        with open('cache/docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)
        with open('cache/term_frequencies.pkl', 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open('cache/doc_lengths.pkl', 'wb') as f:
            pickle.dump(self.doc_lengths, f)


    def load(self):
        if not os.path.exists('cache/index.pkl') or not os.path.exists('cache/docmap.pkl') or not os.path.exists('cache/term_frequencies.pkl'):
            raise FileNotFoundError("The specified file was not found.")
        with open('cache/index.pkl', 'rb') as f:
            loaded_index = pickle.load(f)
        with open('cache/docmap.pkl', 'rb') as f:
            loaded_docmap = pickle.load(f)
        with open('cache/term_frequencies.pkl', 'rb') as f:
            loaded_term_frequencies = pickle.load(f)
        with open('cache/doc_lengths.pkl', 'rb') as f:
            loaded_doc_lengths = pickle.load(f)

        self.index = loaded_index
        self.docmap = loaded_docmap
        self.term_frequencies = loaded_term_frequencies
        self.doc_lengths = loaded_doc_lengths

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return avg_doc_length



    def get_tf(self, doc_id, term):
        tokens = to_stem(to_token(preprocess_text(term)))
        if not tokens:
            return 0

        if len(tokens) > 1:
            raise ValueError("Value error")

        token = tokens[0]
        counter = self.term_frequencies.get(doc_id, Counter())
        return counter.get(token, 0)


    def get_bm25_idf(self, term: str) -> float:
        docs_total = len(self.docmap)
        if docs_total == 0:
            raise ValueError("Cannot calculate IDF on an empty index")
        tokens = to_stem(to_token(preprocess_text(term)))
        if not tokens:
            raise ValueError("Value error")

        norm_term = tokens[0]
        doc_count = len(self.index.get(norm_term, set()))
        bm25idf_score = math.log((docs_total - doc_count + 0.5) / (doc_count + 0.5) + 1)
        return bm25idf_score

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        raw_tf = self.get_tf(doc_id, term)
        if raw_tf == 0:
            return 0.0
        avg = self.__get_avg_doc_length()
        if avg == 0:
            return 0.0
        doc_len = self.doc_lengths.get(doc_id, 0)
        if doc_len == 0:
            return 0.0
        length_norm = 1 - b + b * (doc_len / avg)
        norm_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return norm_tf

    def bm25(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        bm25_tf = self.get_bm25_tf(doc_id, term, BM25_K1, BM25_B)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokens = to_stem(to_token(preprocess_text(query)))
        if not tokens:
            return []
        scores = defaultdict(float)
        for token in tokens:
            #print(token, raw_tf, doc_len, avg_len, idf, tf, tf*idf)
            for doc_id in self.index.get(token, set()):
                scores[doc_id] += self.bm25(doc_id, token)

        s = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return s



