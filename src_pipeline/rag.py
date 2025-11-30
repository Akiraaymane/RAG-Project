import openai  # Grok uses the OpenAI-compatible API

class RAG:
    def __init__(self, provider, model, api_key):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        openai.api_key = self.api_key

    def answer_query(self, retrieved_chunks, query):
        context = " ".join([chunk["text"] for chunk in retrieved_chunks])
        prompt = f"Using the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.choices[0].message["content"]
        return answer


