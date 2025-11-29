"""
src/indexer.py : Document Indexing (Minimal)
"""
from typing import List, Dict, Tuple
from pathlib import Path
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class DocumentIndexer:
    

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 14,
        vector_db_path: str = "data/vector_db",
        raw_data_path: str = "data"
    ) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_store = None
        self.db_path = Path(vector_db_path)
        self.raw_data_path = Path(raw_data_path)

    def index(self, clear: bool = True) -> Dict[str, int]:
        """Indexe les documents pdf"""
        if clear and self.db_path.exists():
            shutil.rmtree(self.db_path)
        
        docs = self._load_all_pdfs()
        chunks = self.splitter.split_documents(docs)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.db_path)
        )
        return self.get_stats()

    def _load_all_pdfs(self) -> List[Document]:
        """Charge tous les docs pdf"""
        path = self.raw_data_path
        if not path.exists():
            raise FileNotFoundError(f"Répertoire non trouvé: {path}")
        
        pdfs = list(path.glob("*.pdf"))
        if not pdfs:
            raise ValueError(f"Aucun PDF dans {path}")
        
        docs = []
        for pdf in pdfs:
            docs.extend(PyPDFLoader(str(pdf)).load())
        return docs


    def get_stats(self) -> Dict[str, int]:
        """Statistiques"""
        if not self.vector_store:
            if not self.db_path.exists():
                return {"total_chunks": 0}
            self.vector_store = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=self.embeddings
            )
        
        all_docs = self.vector_store.get()
        sources = {}
        for meta in all_docs["metadatas"]:
            source = meta.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_chunks": len(all_docs["ids"]),
            "chunks_by_source": sources
        }
    def view_chunks(self, filename: str) -> List[Dict]:
        """Affiche les chunks d'un document spécifique"""
        if not self.vector_store:
            if not self.db_path.exists():
                raise FileNotFoundError("Base vectorielle non trouvée")
            self.vector_store = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=self.embeddings
            )
        
        all_docs = self.vector_store.get()
        chunks = []
        
        for i, (doc_id, metadata) in enumerate(zip(all_docs["ids"], all_docs["metadatas"])):
            if filename in metadata.get("source", ""):
                chunks.append({
                    "chunk_id": i,
                    "page": metadata.get("page"),
                    "content": all_docs["documents"][i]
                })
        
        return chunks