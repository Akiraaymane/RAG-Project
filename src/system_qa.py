"""
src/qa_system.py - Question-Answering System with HuggingFace Inference API
"""
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

from src.retriever import DocumentRetriever


class QASystem:
    """Système de Question-Réponse avec HuggingFace Inference API"""

    def __init__(self, hf_api_key: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.hf_api_key = hf_api_key
        self.model = model
        self.client = InferenceClient(api_key=self.hf_api_key)
        self.retriever = DocumentRetriever()

    def retrieve_context(self, question: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Récupère les documents pertinents"""
        return self.retriever.search(question, top_k=top_k)

    def format_context(self, documents: List[Tuple[Document, float]]) -> str:
        """Formate les documents en contexte"""
        context = "CONTEXTE EXTRAIT DES DOCUMENTS:\n\n"
        for i, (doc, score) in enumerate(documents, 1):
            relevance = self.retriever.convert_l2_to_relevance(score)
            context += f"[Document {i} - Pertinence: {relevance:.2%}]\n"
            context += f"Source: {doc.metadata.get('source', 'N/A')}\n"
            context += f"Contenu: {doc.page_content}\n\n"
        return context

    def create_prompt(self, question: str, context: str) -> str:
        """Crée le prompt optimisé pour HuggingFace  , si on fait réponds uniquement en français , mon système devient si lent"""
        prompt = f"""Tu es un expert en philosophie. Réponds à la question basée UNIQUEMENT sur le contexte fourni.

{context}

QUESTION: {question}

INSTRUCTIONS:
1. Réponds uniquement avec le contexte fourni
2. Si la réponse n'est pas dans le contexte, dis-le clairement
3. Cite les sources quand tu utilises le contexte
4. Sois concis et académique

RÉPONSE:"""
        return prompt

    def query(self, question: str, top_k: int = 3) -> Dict[str, str]:
        """Pose une question et retourne la réponse complète"""
        
        # 1. Récupérer les documents pertinents
        documents = self.retrieve_context(question, top_k=top_k)
        
        # 2. Formater le contexte
        context = self.format_context(documents)
        
        # 3. Créer le prompt
        prompt = self.create_prompt(question, context)
        
        # 4. Appeler HuggingFace API
        response = self._call_hf_api(prompt)
        
        return {
            "question": question,
            "context_documents": len(documents),
            "answer": response,
            "sources": [doc.metadata.get('source', 'N/A') for doc, _ in documents]
        }

    def _call_hf_api(self, prompt: str) -> str:
        """Appelle l'API HuggingFace pour un modèle chat."""
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.3
            )

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Erreur HuggingFace API: {str(e)}"

