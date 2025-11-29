"""
cli.py 
"""
import sys
from src.indexer import DocumentIndexer
from src.retriever import DocumentRetriever
retriever = DocumentRetriever()
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
        stats = indexer.index(clear=False)
        print(f"Chunks: {stats['total_chunks']}")
        print(f"Sources: {stats['chunks_by_source']}")
    
    elif cmd == "search" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        results = retriever.search(query, top_k=3)
        
        for i, (doc, l2_score) in enumerate(results, 1):
            relevance = retriever.convert_l2_to_relevance(l2_score)
            print(f"\n{i}. Score: {relevance:.4f} (L2: {l2_score:.4f})")
            print(f"   Source: {doc.metadata.get('source', 'N/A')}")
            print(f"   Content: {doc.page_content[:100]}")
    
    elif cmd == "stats":
        stats = indexer.get_stats()
        print(stats)

        
    elif cmd == "chunks" and len(sys.argv) > 2:
        indexer = DocumentIndexer()
        filename = sys.argv[2]
        chunks = indexer.view_chunks(filename)
        
        print(f"Total chunks pour '{filename}': {len(chunks)}\n")
        for chunk in chunks:
            print(f"Chunk #{chunk['chunk_id']} (Page {chunk['page']}):")
            print(f"{chunk['content']}\n")
            print("-" * 80 + "\n")    