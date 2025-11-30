from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks):
        embedded_chunks = []

        for chunk in chunks:
            vector = self.model.encode(chunk["text"])
            embedded_chunk = {
                "embedding": vector,
                "metadata": chunk["metadata"],
                "text": chunk["text"]
            }
            embedded_chunks.append(embedded_chunk)

        return embedded_chunks






# Class Embedder:
#     Function __init__(model_name="all-MiniLM-L6-v2"):
#         Load the embedding model with the given model_name

#     Function embed_chunks(chunks):
#         Create an empty list called embedded_chunks

#         For each chunk in chunks:
#             Encode the chunk's text into a vector using the embedding model
#             Create a dictionary embedded_chunk:
#                 embedding -> the vector
#                 metadata -> copy from chunk
#                 text -> original chunk text
#             Add embedded_chunk to embedded_chunks

#         Return the list embedded_chunks
