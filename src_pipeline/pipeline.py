from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class RAGPipeline:
    def __init__(self, llm, retriever):
        prompt = ChatPromptTemplate.from_template(
            "Answer ONLY using the context.\n\nContext:\n{context}\n\nQuestion: {input}"
        )

        self.combine = create_stuff_documents_chain(llm, prompt)
        self.chain = create_retrieval_chain(retriever, self.combine)

    def ask(self, query: str):
        response = self.chain.invoke({"input": query})
        return response["answer"]
