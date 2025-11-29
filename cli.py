import argparse
import json

from src.indexer import DocumentIndexer
from src.rag import RAGPipeline
from src.evaluator import evaluate_rag_all


def run_index(config_path: str):
    print("[CLI] Starting indexing...")
    indexer = DocumentIndexer(config_path)
    indexer.index()
    print("[CLI] Indexing completed.")


def run_ask(config_path: str, question: str, k: int):
    print("[CLI] Running RAG query...")
    rag = RAGPipeline(config_path)
    answer = rag.ask(question, k=k)
    print("\n=== ANSWER ===")
    print(answer)
    print("==============")
    return answer


def run_evaluate(config_path: str, k: int, output_path: str):
    print("[CLI] Starting evaluation...")
    evaluate_rag_all(
        config_path=config_path,
        k_retrieval=k,
        output_json_path=output_path,
    )
    print(f"[CLI] Evaluation completed. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAG Project Command-Line Interface")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- index --------
    p_index = subparsers.add_parser("index", help="Index all documents")
    p_index.add_argument("--config", required=True, help="Path to config YAML")

    # -------- ask --------
    p_ask = subparsers.add_parser("ask", help="Ask a question to the RAG system")
    p_ask.add_argument("--config", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--k", type=int, default=4)

    # -------- evaluate --------
    p_eval = subparsers.add_parser("evaluate", help="Evaluate the RAG system")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--k", type=int, default=4)
    p_eval.add_argument("--output", default="results.json")

    args = parser.parse_args()

    if args.command == "index":
        run_index(args.config)

    elif args.command == "ask":
        run_ask(args.config, args.question, args.k)

    elif args.command == "evaluate":
        run_evaluate(args.config, args.k, args.output)


if __name__ == "__main__":
    main()
