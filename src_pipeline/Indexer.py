from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

class DocumentIndexer:
    def __init__(self, db_path: str, embeddings_model: str):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

    def build_index(self, chunks):
        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        db.persist()
        return db

    def load_index(self):
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
