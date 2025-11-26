import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, persist_directory="vector_store"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")
        self.persist_directory = persist_directory

    def add_embeddings(self, embedded_chunks):
        for chunk in embedded_chunks:
            self.collection.add(
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
                embeddings=[chunk["embedding"].tolist()]
            )

    def persist(self):
        self.client.persist(self.persist_directory)

    def query(self, query_vector, n_results=5):
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=n_results
        )
        return results





# Class VectorStore:
#     When you create a VectorStore:
#         Make a new connection to the Chroma database
#         Create a folder or collection called "documents" to store all the embeddings
#         Remember where to save this collection on disk

#     To add embeddings:
#         For each piece of text that has been turned into numbers:
#             Save the text itself
#             Save the information about the text (like filename or path)
#             Save the numbers that represent the text (the embedding)

#     To save the collection:
#         Write all stored data to the folder on disk

#     To search for relevant pieces:
#         Take a query in number form (embedding of a question)
#         Look in the collection and find the top matching pieces of text
#         Return these top pieces along with their information
