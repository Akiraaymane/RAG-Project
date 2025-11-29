"""
src/retriever.py - Q2: Document Retrieval (Minimal)
"""
from typing import List, Tuple
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class DocumentRetriever:
    """Recherche dans la base vectorielle , vector_db"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_db_path: str = "data/vector_db",
        top_k: int = 3
    ) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.db_path = Path(vector_db_path)
        self.top_k = top_k

    def load_vector_store(self) -> None:
        """Charge la base vectorielle"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Base vectorielle non trouvée: {self.db_path}")
        
        self.vector_store = Chroma(
            persist_directory=str(self.db_path),
            embedding_function=self.embeddings
        )

    def search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """Recherche les documents similaires avec scores"""
        if not self.vector_store:
            self.load_vector_store()
        
        if not query or not query.strip():
            raise ValueError("Requête vide")
        
        k = top_k if top_k else self.top_k
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return results

    def convert_l2_to_relevance(self, l2_distance: float) -> float:
        """Convertit distance L2 (0-2) en score pertinence (0-1)"""
        relevance = max(0, 1 - (l2_distance / 2))
        return round(relevance, 4)