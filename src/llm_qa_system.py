import os
from groq import Groq


class GroqLLM:
    """
    LLM wrapper using Groq API (LLaMA 3.1 8B).
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY n'est pas défini dans les variables d'environnement."
            )

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        print(f"[INFO] Groq LLM initialisé avec le modèle: {self.model_name}")

    def generate(self, prompt: str, max_tokens: int = 350) -> str:
        """
        Envoie le prompt au modèle Groq et retourne le texte généré.
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, precise assistant. "
                        "Answer ONLY based on the given context. "
                        "If the answer is not in the context, say it clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response.choices[0].message.content
