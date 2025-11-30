class Retriever:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query, n_results=5):
        query_vector = self.embedder.model.encode(query)
        results = self.vector_store.query(query_vector, n_results=n_results)

        retrieved_chunks = []
        for i, text in enumerate(results["documents"][0]):
            chunk = {
                "text": text,
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i]
            }
            retrieved_chunks.append(chunk)

        return retrieved_chunks








# Class Retriever:
#     When you create a Retriever:
#         Remember which VectorStore to use
#         Remember which Embedder to use

#     To retrieve relevant chunks for a query:
#         Turn the query text into numbers using the Embedder
#         Ask the VectorStore to find the top matching chunks for these numbers
#         For each matching chunk returned:
#             Make a dictionary with:
#                 text -> the chunk text
#                 metadata -> the information about the chunk (like filename)
#                 score -> how close it is to the query
#             Add this dictionary to the list of retrieved chunks
#         Return the list of retrieved chunks
