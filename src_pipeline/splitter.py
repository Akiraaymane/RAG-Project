class DocumentSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        all_chunks = []

        for doc in documents:
            words = doc["text"].split()
            start = 0

            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_text = " ".join(words[start:end])

                chunk = {
                    "text": chunk_text,
                    "metadata": doc["metadata"]
                }
                all_chunks.append(chunk)

                start += self.chunk_size - self.chunk_overlap

        return all_chunks





# Class DocumentSplitter:
#     Function __init__(chunk_size=500, chunk_overlap=50):
#         Store the chunk size
#         Store the chunk overlap

#     Function split_documents(documents):
#         Create an empty list called all_chunks

#         For each document in documents:
#             Split the document's text into words
#             Initialize start index to 0

#             While start is less than total number of words:
#                 Calculate end index = start + chunk_size (or end of words)
#                 Create chunk_text by joining words from start to end

#                 Create a dictionary called chunk:
#                     text -> chunk_text
#                     metadata -> copy metadata from document

#                 Add chunk to all_chunks

#                 Move start forward by (chunk_size - chunk_overlap)

#         Return the list all_chunks
