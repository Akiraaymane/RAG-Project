"""
cli.py - CLI Q1 (Ultra-minimal)
"""
from src.indexer import DocumentIndexer

indexer = DocumentIndexer()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cli.py index")
        print("  python cli.py search <query>")
        print("  python cli.py stats")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "index":
        stats = indexer.index(clear=True)
        print(f"Chunks: {stats['total_chunks']}")
        print(f"Sources: {stats['chunks_by_source']}")
    
    elif cmd == "search" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        results = indexer.search(query, top_k=3)
        for doc, score in results:
            print(f"Score: {score:.4f} | {doc.page_content[:80]}")
    
    elif cmd == "stats":
        stats = indexer.get_stats()
        print(stats)