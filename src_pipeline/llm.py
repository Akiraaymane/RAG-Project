from langchain_groq import ChatGroq

class LLMBuilder:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    def build(self):
        return ChatGroq(
            api_key=self.api_key,
            model=self.model_name,
            temperature=0
        )
