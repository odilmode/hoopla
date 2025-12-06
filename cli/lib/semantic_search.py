from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .search_utils import CACHE_DIR, load_movies, CHUNK_SIZE, MAX_CHUNK_SIZE, OVERLAP
import re
import json
class SemanticSearch:
    def __init__(self, model=None, embeddings=None, documents=None, document_map=None):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings= embeddings
        self.documents = documents or []
        self.document_map = document_map or {}



    def generate_embedding(self, text):

        if not text or text.isspace():
            raise ValueError("Empty text")
        embeddings = self.model.encode([text])
        embedding = embeddings[0]
        return embedding


    def build_embeddings(self, documents):
        self.documents = documents
        l = []
        for document in documents:
            self.document_map[document['id']] = document
            l.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(l, show_progress_bar=True)
        path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

        np.save(path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document
        if os.path.exists(CACHE_DIR + "/movie_embeddings.npy"):
            path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
            self.embeddings = np.load(path, allow_pickle=False)
            if self.embeddings.shape[0] == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding = self.generate_embedding(query)
        similarity_scores = []

        for i, emb in enumerate(self.embeddings):
            score = cosine_similarity(emb, embedding)
            similarity_scores.append((score, i))
        l = sorted(similarity_scores, reverse=True)[:limit]
        ll = []
        for score, i in l:
            doc = self.documents[i]
            ll.append({
                      "score": float(score),
                      "title": doc["title"],
                      "description": doc["description"],
            })

            

        return ll



class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2")->None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        l = []
        metadatas = []
        for id, doc in enumerate(documents):
            if len(doc["description"]) == 0:
                continue
            chunks = semantic_chunk(doc["description"], max_chunk_s=MAX_CHUNK_SIZE, overlap=OVERLAP)
            l.extend(chunks)
            for idx, c in enumerate(chunks):
                metadatas.append({
                                 "movie_idx": id,
                                 "chunk_idx": idx,
                                 "total_chunks": len(chunks),
                                 })

        self.chunk_embeddings = self.model.encode(l, show_progress_bar=True)
        self.chunk_metadata = metadatas
        path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        np.save(path, self.chunk_embeddings)
        path1 = os.path.join(CACHE_DIR, "chunk_metadata.json")
        with open(path1, 'w') as json_file:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(l)}, json_file, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(CACHE_DIR + "/chunk_embeddings.npy") and os.path.exists(CACHE_DIR + "/chunk_metadata.json"):
            path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
            self.chunk_embeddings = np.load(path, allow_pickle=False)
            path1 = os.path.join(CACHE_DIR, "chunk_metadata.json")
            with open(path1, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)



def embed_chunks():
    data = load_movies()
    chunked_search = ChunkedSemanticSearch()
    build_embs = chunked_search.load_or_create_chunk_embeddings(data)
    print(f"Generated {len(build_embs)} chunked embeddings")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")




def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search(query, limit):
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)
    result = semantic_search.search(query, limit)
    for i, r in enumerate(result, start=1):
        print(f"{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"  {r['description']}\n")



def chunk(text, chunk_s=CHUNK_SIZE, overlap=0):
    if chunk_s <= 0:
        raise ValueError("chunk size must be >=1")

    if overlap < 0 or overlap >= chunk_s:
        raise ValueError("overlap must be >= 0 and chunk size")

    words = text.split()
    chunks = []
    step = chunk_s - overlap
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_s]
        chunks.append(" ".join(chunk))
        i += step
    total_chars = len(text)
    print(f"Chunking {total_chars} characters\n")
    for idx, c in enumerate(chunks, start=1):
        print(f"{idx}. {c}")


def semantic_chunk(text, max_chunk_s=MAX_CHUNK_SIZE, overlap=OVERLAP):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    i = 0
    n_s = len(sentences)
    while i < n_s:
        chunk = sentences[i:i+max_chunk_s]
        if  chunks and len(chunk) <=overlap:
            break
        chunks.append(" ".join(chunk))
        i += max_chunk_s - overlap
    return chunks




