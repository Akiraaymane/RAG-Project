from src.search import DocumentSearcher
from src.llm_qa_system import GroqLLM
from src.utils.build_rag_prompt import build_rag_prompt  


class RAGPipeline:

    def __init__(self, config_path: str = "config.yaml"):
        # Retriever
        self.searcher = DocumentSearcher(config_path)
        self.searcher.load_vectorstore()

        # LLM via Groq
        self.llm = GroqLLM()

    def ask(self, question: str, k: int = 4) -> str:
        # Récupérer les k meilleurs chunks
        chunks = self.searcher.search(question, k=k)

        # Construire le prompt RAG
        prompt = build_rag_prompt(question, chunks)

        # Générer la réponse via Groq
        answer = self.llm.generate(prompt)
        return answer

