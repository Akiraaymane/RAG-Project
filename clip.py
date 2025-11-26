from src.loader import DocumentLoader
from src.splitter import DocumentSplitter
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.rag import RAG
from src.utils import load_config

# Load config
config = load_config("config.yaml")

# 1. Load documents
loader = DocumentLoader(config["data_path"])
documents = loader.load_documents()

# 2. Split documents
splitter = DocumentSplitter(config["chunk_size"], config["chunk_overlap"])
chunks = splitter.split_documents(documents)

# 3. Embed chunks
embedder = Embedder(config["embedding_model"])
embedded_chunks = embedder.embed_chunks(chunks)

# 4. Store embeddings
vector_store = VectorStore(config["vector_store_path"])
vector_store.add_embeddings(embedded_chunks)
vector_store.persist()

# 5. Retrieve relevant chunks
retriever = Retriever(vector_store, embedder)
relevant_chunks = retriever.retrieve(config["user_query"])

# 6. Generate answer using Grok
rag = RAG(
    config["llm"]["provider"], 
    config["llm"]["model"], 
    config["llm"]["api_key"]
)
answer = rag.answer_query(relevant_chunks, config["user_query"])

print(answer)
