import yaml
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document   


class DocumentSearcher:
    """
    Classe responsable de la recherche dans la base vectorielle FAISS.
    - Recharge la base FAISS sauvegardée par l'indexer
    - Fournit des méthodes pour faire des recherches par similarité
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Charger la config (la même que pour l'indexer)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.persist_dir: str = self.config["persist_dir"]
        self.embedding_model_name: str = self.config["embedding_model_name"]

        # Initialiser le modèle d'embedding (même modèle que l'indexer)
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )

        # Placeholder pour le vector store
        self.vectorstore: FAISS | None = None

    # -------------------------
    # 1) Chargement du FAISS
    # -------------------------
    def load_vectorstore(self) -> FAISS:
        """
        Recharge une base FAISS sauvegardée dans persist_dir.
        À utiliser après que l'indexation ait été effectuée.
        """
        print(f"[INFO] Chargement de la base vectorielle FAISS depuis {self.persist_dir} ...")

        self.vectorstore = FAISS.load_local(
            self.persist_dir,
            self.embedding_function,
            allow_dangerous_deserialization=True,  # nécessaire avec les nouvelles versions de langchain
        )

        print("[INFO] Base vectorielle FAISS chargée.")
        return self.vectorstore

    # -------------------------
    # 2) Recherche simple
    # -------------------------
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Recherche les k chunks les plus pertinents pour une requête donnée.
        Retourne une liste de Documents (texte + metadata).
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vector store non chargé. "
                "Appelle d'abord load_vectorstore()."
            )

        results = self.vectorstore.similarity_search(query, k=k)

        return results

    # -------------------------
    # 3) Recherche avec scores
    # -------------------------
    def search_with_scores(self, query: str, k: int = 5):
        """
        Recherche les k chunks les plus pertinents et retourne (Document, score).
        Utile pour l'analyse, l'évaluation et le rapport.
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vector store non chargé. "
                "Appelle d'abord load_vectorstore()."
            )
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

