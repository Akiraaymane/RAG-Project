
class RetrieverBuilder:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def build(self, top_k: int):
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})
