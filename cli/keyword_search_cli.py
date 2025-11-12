#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command, bm25_idf_command, bm25_tf_command, bm25_search_command
from lib.inverted_index import InvertedIndex
from lib.search_utils import load_movies, BM25_K1, BM25_B
from lib.text_utils import to_stem, to_token, preprocess_text
import sys
import math

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")
    tf_parser = subparsers.add_parser("tf", help="Term frequency of the each token")
    tf_parser.add_argument("doc_id", type=int, help="Docs id")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Term Frequency-Inverse Document Frequency")
    tfidf_parser.add_argument("doc_id", type=int, help="Docs id")
    tfidf_parser.add_argument("term", type=str, help="Term")
    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
   "bm25tf", help="Get BM25 TF score for a given document ID and term"
)
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=str, help="Top queries")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f'{i}. {res["title"]}')

        case "build":
            movies = load_movies()
            idx = InvertedIndex()
            idx.build(movies)
            idx.save()

        case "tf":
            doc_id = int(sys.argv[2])
            term = sys.argv[3]
            idx = InvertedIndex()
            idx.load()
            print(idx.get_tf(doc_id, term))


        case "idf":
            term = to_stem(to_token(preprocess_text(sys.argv[2])))
            norm_term = term[0]
            idx = InvertedIndex()
            idx.load()
            docs_total = len(idx.docmap)
            term_doc_count = len(idx.index.get(norm_term, set()))
            val_idf = math.log((docs_total+1)/(term_doc_count+1))
            print(f"Inverse document frequency of '{norm_term}': {val_idf:.2f}")
            print(type(idx.index.get(norm_term)), len(idx.index.get(norm_term, set())))
            print(len(idx.docmap))

        case "tfidf":
            doc_id = int(sys.argv[2])
            term = to_stem(to_token(preprocess_text(sys.argv[3])))
            norm_term = term[0]
            idx = InvertedIndex()
            idx.load()
            docs_total = len(idx.docmap)
            term_doc_count = len(idx.index.get(norm_term, set()))
            val_idf = math.log((docs_total+1)/(term_doc_count+1))
            tf = idx.term_frequencies[doc_id][norm_term]
            tfidf = tf * val_idf
            print(f"TF-IDF score of '{norm_term}' in document '{doc_id}': {tfidf:.2f}")

        case "bm25idf":
            term = sys.argv[2]
            score = bm25_idf_command(term)
            print(f"BM25 IDF score of '{term}': {score:.2f}")

        case "bm25tf":
            doc_id = int(sys.argv[2])
            term = sys.argv[3]
            score = bm25_tf_command(doc_id, term, args.k1, args.b)
            print(f"BM25 TF score of '{term}' in document '{doc_id}': {score:.2f}")

        case "bm25search":
            term = sys.argv[2]
            scores = bm25_search_command(term, limit=5)
            idx = InvertedIndex();
            idx.load()  # or reuse the same idx if bm25_search_command can return it
            for i, (doc_id, score) in enumerate(scores, start=1):
                title = idx.docmap[doc_id]["title"]
                print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
