#!/usr/bin/env python3

import argparse
import sys

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk, semantic_chunk, embed_chunks

from lib.search_utils import CHUNK_SIZE, MAX_CHUNK_SIZE

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verifies model")
    embed_text_parser = subparsers.add_parser("embed_text", help="Generates embeddings")
    embed_text_parser.add_argument("text", type=str, help="Text for embedding")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verifies the embeddings and dimensions")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embeds query")
    embedquery_parser.add_argument("query", type=str, help="User query for embedding")
    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", type=str, help="User query for semantic search")
    search_parser.add_argument("--limit", type=int, default=5, help="Optional limit")

    chunk_parser = subparsers.add_parser("chunk", help="Chunks the documents by size")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Optional chunk size")
    chunk_parser.add_argument("--overlap", type=int, help="Overlaps last n chunks")


    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunks the document by paragraphs")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=MAX_CHUNK_SIZE, help="Optional chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlaps last n chunks")


    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embeds chunks")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            total_chars = len(args.text)
            print(f"Semantically chunking {total_chars} characters\n")
            for idx, c in enumerate(chunks, start=1):
                print(f"{idx}. {c}")
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
