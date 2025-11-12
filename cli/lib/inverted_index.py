from .text_utils import to_token, to_stem, preprocess_text
from collections import defaultdict
from collections import Counter
import string
import pickle
import os
import math
from threading import RLock
from .search_utils import BM25_K1, BM25_B, CACHE_DIR


class InvertedIndex:
    """
    Inverted index with BM25 scoring for document search.
    
    Maintains mappings from terms to documents and computes BM25 relevance scores.
    Supports persistence via pickle serialization.
    Thread-safe for concurrent reads.
    """
    
    def __init__(self, index=None, docmap=None, term_frequencies=None, doc_lengths=None):
        """
        Initialize an InvertedIndex.
        
        Args:
            index: Optional pre-built inverted index (dict of term -> set of doc_ids)
            docmap: Optional pre-built document map (dict of doc_id -> document)
            term_frequencies: Optional pre-built term frequencies (dict of doc_id -> Counter)
            doc_lengths: Optional pre-built document lengths (dict of doc_id -> length)
        """
        self.index = index if index is not None else defaultdict(set)
        self.docmap = docmap if docmap is not None else {}
        self.term_frequencies = term_frequencies if term_frequencies is not None else defaultdict(Counter)
        self.doc_lengths = doc_lengths if doc_lengths is not None else {}
        self._lock = RLock()

    def __add_document(self, doc_id, text):
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique identifier for the document
            text: Text content to index
        """
        if doc_id is None:
            raise ValueError("doc_id cannot be None")
        
        tokens = to_stem(to_token(preprocess_text(text)))
        tokens_total = len(tokens)
        
        with self._lock:
            self.doc_lengths[doc_id] = tokens_total
            for token in tokens:
                self.index[token].add(doc_id)
                self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        """
        Find documents containing a term.
        
        Args:
            term: Query term(s)
            
        Returns:
            Sorted list of matching document IDs
        """
        tokens = to_stem(to_token(preprocess_text(term)))
        if not tokens:
            return []
        token = tokens[0]
        doc_ids = self.index.get(token, set())
        return sorted(doc_ids)

    def build(self, movies):
        """
        Build the inverted index from a collection of movie documents.
        
        Args:
            movies: List of movie dictionaries, each with 'id', 'title', and 'description'
            
        Raises:
            ValueError: If any movie is missing required 'id' field
        """
        if not movies:
            return
        
        with self._lock:
            for movie in movies:
                doc_id = movie.get('id')
                if doc_id is None:
                    raise ValueError("Movie missing required 'id' field")
                
                title = movie.get('title', '')
                description = movie.get('description', '')
                
                # Skip movies with no indexable content
                if not title and not description:
                    continue
                
                self.docmap[doc_id] = movie
                text = f"{title} {description}"
                self.__add_document(doc_id, text)

    def save(self):
        """
        Persist the index to disk using pickle.
        Uses relative 'cache/' path for compatibility with existing tests.
        """
        if not os.path.exists('cache'):
            os.makedirs('cache', exist_ok=True)
        
        with self._lock:
            with open('cache/index.pkl', 'wb') as f:
                pickle.dump(self.index, f)
            with open('cache/docmap.pkl', 'wb') as f:
                pickle.dump(self.docmap, f)
            with open('cache/term_frequencies.pkl', 'wb') as f:
                pickle.dump(self.term_frequencies, f)
            with open('cache/doc_lengths.pkl', 'wb') as f:
                pickle.dump(self.doc_lengths, f)

    def load(self):
        """
        Load the persisted index from disk.
        Uses relative 'cache/' path for compatibility with existing tests.
        
        Raises:
            FileNotFoundError: If cache files don't exist
        """
        if not os.path.exists('cache/index.pkl') or not os.path.exists('cache/docmap.pkl') or not os.path.exists('cache/term_frequencies.pkl'):
            raise FileNotFoundError("The specified file was not found.")
        
        with self._lock:
            with open('cache/index.pkl', 'rb') as f:
                self.index = pickle.load(f)
            with open('cache/docmap.pkl', 'rb') as f:
                self.docmap = pickle.load(f)
            with open('cache/term_frequencies.pkl', 'rb') as f:
                self.term_frequencies = pickle.load(f)
            with open('cache/doc_lengths.pkl', 'rb') as f:
                self.doc_lengths = pickle.load(f)

    def __get_avg_doc_length(self) -> float:
        """
        Calculate average document length in the index.
        
        Returns:
            Average length, or 0.0 if index is empty
        """
        if not self.doc_lengths:
            return 0.0
        avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return avg_doc_length

    def get_tf(self, doc_id, term):
        """
        Get term frequency for a term in a document.
        
        Args:
            doc_id: Document ID
            term: Query term
            
        Returns:
            Number of occurrences of term in document
            
        Raises:
            ValueError: If term contains multiple tokens after preprocessing
        """
        tokens = to_stem(to_token(preprocess_text(term)))
        if not tokens:
            return 0

        if len(tokens) > 1:
            raise ValueError(
                f"Term '{term}' resulted in multiple tokens {tokens}. "
                "Please provide a single term."
            )

        token = tokens[0]
        counter = self.term_frequencies.get(doc_id, Counter())
        return counter.get(token, 0)

    def get_bm25_idf(self, term: str) -> float:
        """
        Calculate BM25 inverse document frequency for a term.
        
        Args:
            term: Query term
            
        Returns:
            IDF score
            
        Raises:
            ValueError: If term is empty after preprocessing
        """
        docs_total = len(self.docmap)
        tokens = to_stem(to_token(preprocess_text(term)))
        if not tokens:
            raise ValueError(
                f"Term '{term}' resulted in no tokens after preprocessing. "
                "Invalid search term."
            )

        norm_term = tokens[0]
        doc_count = len(self.index.get(norm_term, set()))
        bm25idf_score = math.log((docs_total - doc_count + 0.5) / (doc_count + 0.5) + 1)
        return bm25idf_score

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        """
        Calculate BM25 term frequency for a term in a document.
        
        Args:
            doc_id: Document ID
            term: Query term
            k1: BM25 k1 parameter (controls term frequency saturation)
            b: BM25 b parameter (controls length normalization)
            
        Returns:
            Normalized term frequency score
        """
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
        """
        Calculate BM25 score for a term in a document.
        
        Args:
            doc_id: Document ID
            term: Query term
            k1: BM25 k1 parameter
            b: BM25 b parameter
            
        Returns:
            BM25 score
        """
        bm25_tf = self.get_bm25_tf(doc_id, term, BM25_K1, BM25_B)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit=10):
        """
        Search documents using BM25 scoring.
        
        Args:
            query: Search query (can be multiple terms)
            limit: Maximum number of results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance (descending)
        """
        if not self.index:
            return []
        
        tokens = to_stem(to_token(preprocess_text(query)))
        if not tokens:
            return []
        
        scores = defaultdict(float)
        
        with self._lock:
            for token in tokens:
                for doc_id in self.index.get(token, set()):
                    scores[doc_id] += self.bm25(doc_id, token)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return ranked
