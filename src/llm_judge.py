# src/llm_judge.py
import os
import json
from typing import Dict, Any
from groq import Groq

class LLMJudge:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    def faithfulness_score(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        system_prompt = (
            "You are an automatic evaluator for a Retrieval-Augmented Generation (RAG) system. "
            "Your job is to judge how much the answer is supported by the provided context. "
            "Be strict about hallucinations."
        )

        user_prompt = f"""
Question:
{question}

Context:
{context}

Answer:
{answer}

Task:
1. Evaluate how well the answer is supported by the context only.
2. Return a JSON object:
   - "score": float between 0 and 1
   - "explanation": short explanation in English

Return ONLY the JSON.
"""

        raw = self._chat(system_prompt, user_prompt).strip()
        try:
            data = json.loads(raw)
            score = float(data.get("score", 0.0))
            explanation = data.get("explanation", "")
        except Exception:
            score = 0.0
            explanation = f"Parsing error. Raw output: {raw}"
        return {"score": score, "explanation": explanation}

    def answer_relevance_score(self, question: str, answer: str) -> Dict[str, Any]:
        system_prompt = (
            "You are an automatic evaluator for question answering systems. "
            "Your job is to judge how relevant and adequate the answer is to the question."
        )

        user_prompt = f"""
Question:
{question}

Answer:
{answer}

Task:
1. Evaluate how well the answer addresses the question.
2. Return a JSON object:
   - "score": float between 0 and 1
   - "explanation": short explanation in English

Return ONLY the JSON.
"""

        raw = self._chat(system_prompt, user_prompt).strip()
        try:
            data = json.loads(raw)
            score = float(data.get("score", 0.0))
            explanation = data.get("explanation", "")
        except Exception:
            score = 0.0
            explanation = f"Parsing error. Raw output: {raw}"
        return {"score": score, "explanation": explanation}
