import os
import yaml
from typing import List, Optional

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document   

class DocumentIndexer:
    """
    Pipeline d'indexation :
    1) Charger les documents (PDF) depuis data_dir
    2) Les découper en chunks
    3) Créer/mettre à jour une base vectorielle FAISS persistée sur disque
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Charger la config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.data_dir: str = self.config["data_dir"]
        self.persist_dir: str = self.config["persist_dir"]
        self.chunk_size: int = self.config["chunk_size"]
        self.chunk_overlap: int = self.config["chunk_overlap"]
        self.embedding_model_name: str = self.config["embedding_model_name"]
        self.max_docs: Optional[int] = self.config.get("max_docs")

        # Créer les dossiers si besoin
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialiser le modèle d’embedding
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )

        # Placeholder pour la base vectorielle
        self.vectorstore: Optional[FAISS] = None

    # --------------------
    # 1) Loading documents
    # --------------------
    def load_documents(self) -> List[Document]:
        """
        Charge tous les PDF dans data_dir et retourne une liste de Documents LangChain.
        """
        loader = PyPDFDirectoryLoader(self.data_dir)
        docs = loader.load()

        if self.max_docs is not None:
            docs = docs[: self.max_docs]

        print(f"[INFO] {len(docs)} documents PDF chargés depuis {self.data_dir}")
        return docs

    # --------------------
    # 2) Splitting
    # --------------------
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Découpe les documents en chunks avec un overlap.
        On conserve les métadonnées (source, page, etc.).
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = splitter.split_documents(docs)
        print(f"[INFO] {len(chunks)} chunks générés (chunk_size={self.chunk_size})")
        return chunks

    # --------------------
    # 3) Construction / mise à jour du vector store
    # --------------------
    def build_vectorstore(self, chunks: List[Document]) -> FAISS:
        """
        Crée une base FAISS à partir des chunks et la sauvegarde sur disque.
        """
        print(f"[INFO] Construction de la base vectorielle FAISS dans {self.persist_dir} ...")

        # Construction de l'index FAISS en mémoire
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_function,
        )

        # Sauvegarde locale (FAISS + index + metadata)
        self.vectorstore.save_local(self.persist_dir)

        print("[INFO] Base vectorielle FAISS construite et sauvegardée.")
        return self.vectorstore


    # --------------------
    # 4) Pipeline complet
    # --------------------
    def index_all(self) -> None:
        """
        Pipeline complet : load → split → build vectorstore.
        """
        docs = self.load_documents()
        if not docs:
            print("[WARN] Aucun document trouvé. Vérifie le dossier data/.")
            return

        chunks = self.split_documents(docs)
        self.build_vectorstore(chunks)
        print("[SUCCESS] Indexation terminée.")


if __name__ == "__main__":
    indexer = DocumentIndexer(config_path="config.yaml")
    indexer.index_all()
