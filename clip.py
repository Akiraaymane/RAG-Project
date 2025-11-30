#!/usr/bin/env python3
import argparse, yaml, os
from src.loader import DocumentLoader
from src.indexer import DocumentIndexer
from src.retriever import RetrieverBuilder
from src.llm import LLMBuilder
from src.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    os.environ["GROQ_API_KEY"] = cfg["groq_api_key"]

    # 1. Load and split markdown files
    loader = DocumentLoader(cfg["data_dir"])
    chunks = loader.load()

    # 2. Index documents (or load existing)
    indexer = DocumentIndexer(cfg["db_path"], cfg["embedding_model"])
    if not os.path.exists(cfg["db_path"]):
        db = indexer.build_index(chunks)
    else:
        db = indexer.load_index()

    # 3. Build retriever
    retriever = RetrieverBuilder(db).build(cfg["top_k"])

    # 4. Build LLM
    llm = LLMBuilder(cfg["groq_api_key"], cfg["llm_model"]).build()

    # 5. Build RAG
    rag = RAGPipeline(llm, retriever)

    print(rag.ask(args.query))


if __name__ == "__main__":
    main()
