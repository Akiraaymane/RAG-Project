"""
src/evaluator.py 

Métriques implémentées:
- Retrieval: Precision@K, Recall@K, MRR, nDCG
- Answer: Faithfulness, Relevance, BLEU-like scores
- Overall: Latency, Success Rate
"""

import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np


@dataclass
class EvaluationSample:
    """Échantillon d'évaluation avec question et vérité de base."""
    question: str
    ground_truth: str
    expected_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Résultat complet d'une évaluation."""
    question: str
    generated_answer: str
    ground_truth: str
    retrieval_metrics: Dict[str, float]
    answer_metrics: Dict[str, float]
    latency_ms: float = 0.0
    sources_retrieved: List[str] = field(default_factory=list)


class RetrievalMetricsCalculator:
    """Calcule les métriques de retrieval."""

    @staticmethod
    def normalize_source(source: str) -> str:
        """Normalise le chemin source."""
        return Path(source).name.lower() if source else ""

    @classmethod
    def precision_at_k(
        cls,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int = 5
    ) -> float:
        """Precision@K: proportion de docs pertinents dans les K premiers."""
        if not retrieved_docs or k == 0:
            return 0.0
        
        top_k_retrieved = retrieved_docs[:k]
        relevant_set = {cls.normalize_source(d) for d in relevant_docs}
        retrieved_set = {cls.normalize_source(d) for d in top_k_retrieved}
        
        overlap = len(relevant_set & retrieved_set)
        return round(overlap / len(top_k_retrieved), 4)

    @classmethod
    def recall_at_k(
        cls,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int = 5
    ) -> float:
        """Recall@K: proportion de docs pertinents trouvés dans les K premiers."""
        if not relevant_docs:
            return 1.0
        
        top_k_retrieved = retrieved_docs[:k]
        relevant_set = {cls.normalize_source(d) for d in relevant_docs}
        retrieved_set = {cls.normalize_source(d) for d in top_k_retrieved}
        
        overlap = len(relevant_set & retrieved_set)
        return round(overlap / len(relevant_set), 4)

    @classmethod
    def mean_reciprocal_rank(
        cls,
        relevant_docs: List[str],
        retrieved_docs: List[str]
    ) -> float:
        """MRR: rang moyen du premier document pertinent."""
        if not relevant_docs:
            return 0.0
        
        relevant_set = {cls.normalize_source(d) for d in relevant_docs}
        
        for rank, doc in enumerate(retrieved_docs, 1):
            if cls.normalize_source(doc) in relevant_set:
                return round(1.0 / rank, 4)
        
        return 0.0

    @classmethod
    def ndcg_at_k(
        cls,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int = 5
    ) -> float:
        """nDCG@K: Normalized Discounted Cumulative Gain."""
        if not relevant_docs:
            return 0.0
        
        relevant_set = {cls.normalize_source(d) for d in relevant_docs}
        top_k = retrieved_docs[:k]
        
        # DCG
        dcg = 0.0
        for i, doc in enumerate(top_k, 1):
            if cls.normalize_source(doc) in relevant_set:
                dcg += 1.0 / np.log2(i + 1)
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant_set), k) + 1))
        
        return round(dcg / idcg, 4) if idcg > 0 else 0.0

    @classmethod
    def calculate(
        cls,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        """Calcule toutes les métriques de retrieval."""
        return {
            'precision_at_k': cls.precision_at_k(relevant_docs, retrieved_docs, k),
            'recall_at_k': cls.recall_at_k(relevant_docs, retrieved_docs, k),
            'mrr': cls.mean_reciprocal_rank(relevant_docs, retrieved_docs),
            'ndcg_at_k': cls.ndcg_at_k(relevant_docs, retrieved_docs, k)
        }


class AnswerMetricsCalculator:
    """Calcule les métriques de qualité des réponses."""

    STOPWORDS = {
        'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'des',
        'et', 'ou', 'mais', 'donc', 'car', 'ni', 'soit',
        'à', 'au', 'aux', 'en', 'dans', 'sur', 'sous', 'entre',
        'est', 'sont', 'être', 'avoir', 'va', 'vont', 'faire',
        'ce', 'cet', 'cette', 'ces', 'ce', 'ceux', 'celle', 'celles',
        'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous',
        'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
        'c\'est', 'ça', 'qui', 'que', 'quoi', 'comment', 'pourquoi'
    }

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalise le texte."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def word_overlap_f1(
        cls,
        generated: str,
        reference: str
    ) -> float:
        """F1 score basé sur le chevauchement de mots."""
        gen_words = set(cls.normalize_text(generated).split())
        ref_words = set(cls.normalize_text(reference).split())
        
        if not gen_words or not ref_words:
            return 0.0
        
        overlap = len(gen_words & ref_words)
        precision = overlap / len(gen_words)
        recall = overlap / len(ref_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return round(f1, 4)

    @classmethod
    def bleu_score(
        cls,
        generated: str,
        reference: str,
        n: int = 2
    ) -> float:
        """Score BLEU simplifié (n-gram overlap)."""
        def get_ngrams(text: str, n: int) -> set:
            words = cls.normalize_text(text).split()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
        
        gen_ngrams = get_ngrams(generated, n)
        ref_ngrams = get_ngrams(reference, n)
        
        if not gen_ngrams or not ref_ngrams:
            return 0.0
        
        overlap = len(gen_ngrams & ref_ngrams)
        return round(overlap / len(gen_ngrams), 4)

    @classmethod
    def faithfulness_score(
        cls,
        answer: str,
        context: str
    ) -> float:
        """Score de fidélité: proportion de l'answer ancrée dans le contexte."""
        answer_words = set(cls.normalize_text(answer).split()) - cls.STOPWORDS
        context_words = set(cls.normalize_text(context).split())
        
        if not answer_words:
            return 1.0
        
        grounded = len(answer_words & context_words)
        return round(grounded / len(answer_words), 4)

    @classmethod
    def answer_relevance(
        cls,
        answer: str,
        question: str
    ) -> float:
        """Score de pertinence: réponse adresse-t-elle la question?"""
        question_words = set(cls.normalize_text(question).split()) - cls.STOPWORDS
        answer_words = set(cls.normalize_text(answer).split())
        
        if not question_words:
            return 1.0
        
        covered = len(question_words & answer_words)
        return round(covered / len(question_words), 4)

    @classmethod
    def calculate(
        cls,
        generated: str,
        reference: str,
        context: str,
        question: str
    ) -> Dict[str, float]:
        """Calcule toutes les métriques de réponse."""
        return {
            'word_overlap_f1': cls.word_overlap_f1(generated, reference),
            'bleu_score': cls.bleu_score(generated, reference),
            'faithfulness': cls.faithfulness_score(generated, context),
            'answer_relevance': cls.answer_relevance(generated, question)
        }


class RAGEvaluator:
    """Évaluateur complet du système RAG."""

    def __init__(self, qa_system, config_path: str = "config.yaml"):
        """Initialise l'évaluateur."""
        self.qa_system = qa_system
        self.retrieval_metrics = RetrievalMetricsCalculator()
        self.answer_metrics = AnswerMetricsCalculator()
        self.results: List[EvaluationResult] = []

    def create_dataset(
        self,
        questions: List[str],
        ground_truths: List[str],
        expected_sources: Optional[List[List[str]]] = None
    ) -> List[EvaluationSample]:
        """Crée un dataset d'évaluation."""
        if len(questions) != len(ground_truths):
            raise ValueError("Questions et ground_truths doivent avoir la même longueur")
        
        if expected_sources is None:
            expected_sources = [[] for _ in questions]
        
        return [
            EvaluationSample(q, gt, es)
            for q, gt, es in zip(questions, ground_truths, expected_sources)
        ]

    def load_dataset(self, filepath: str) -> List[EvaluationSample]:
        """Charge un dataset depuis JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [
            EvaluationSample(
                question=item['question'],
                ground_truth=item['ground_truth'],
                expected_sources=item.get('expected_sources', []),
                metadata=item.get('metadata', {})
            )
            for item in data
        ]

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        k: int = 3
    ) -> EvaluationResult:
        """Évalue un seul échantillon."""
        start_time = time.time()
        
        # Récupérer la réponse
        qa_result = self.qa_system.query(sample.question, top_k=k)
        generated_answer = qa_result['answer']
        sources = qa_result.get('sources', [])
        context = self._get_context(sample.question, k)
        
        # Calculer métriques
        retrieval_metrics = self.retrieval_metrics.calculate(
            sample.expected_sources,
            sources,
            k=k
        )
        
        answer_metrics = self.answer_metrics.calculate(
            generated_answer,
            sample.ground_truth,
            context,
            sample.question
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return EvaluationResult(
            question=sample.question,
            generated_answer=generated_answer,
            ground_truth=sample.ground_truth,
            retrieval_metrics=retrieval_metrics,
            answer_metrics=answer_metrics,
            latency_ms=round(latency_ms, 2),
            sources_retrieved=sources
        )

    def _get_context(self, question: str, k: int) -> str:
        """Récupère le contexte pour une question."""
        try:
            self.qa_system.retriever.load_vector_store()
            results = self.qa_system.retriever.search(question, top_k=k)
            return "\n".join([doc.page_content for doc, _ in results])
        except Exception:
            return ""

    def evaluate_dataset(
        self,
        dataset: List[EvaluationSample],
        k: int = 3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Évalue un dataset entier."""
        self.results = []
        
        for i, sample in enumerate(dataset):
            if verbose:
                print(f"[{i+1}/{len(dataset)}] Évaluation en cours...")
            
            result = self.evaluate_sample(sample, k)
            self.results.append(result)
        
        return self._aggregate_results()

    def _aggregate_results(self) -> Dict[str, Any]:
        """Agrège les résultats."""
        if not self.results:
            return {'error': 'Aucun résultat à agréger'}
        
        n = len(self.results)
        
        # Agrégation retrieval
        retrieval_keys = self.results[0].retrieval_metrics.keys()
        avg_retrieval = {
            key: round(
                sum(r.retrieval_metrics.get(key, 0) for r in self.results) / n,
                4
            )
            for key in retrieval_keys
        }
        
        # Agrégation answer
        answer_keys = self.results[0].answer_metrics.keys()
        avg_answer = {
            key: round(
                sum(r.answer_metrics.get(key, 0) for r in self.results) / n,
                4
            )
            for key in answer_keys
        }
        
        # Stats latency
        latencies = [r.latency_ms for r in self.results]
        
        return {
            'summary': {
                'num_samples': n,
                'avg_latency_ms': round(sum(latencies) / n, 2),
                'min_latency_ms': round(min(latencies), 2),
                'max_latency_ms': round(max(latencies), 2),
                'timestamp': datetime.now().isoformat()
            },
            'retrieval_metrics': avg_retrieval,
            'answer_metrics': avg_answer,
            'details': [
                {
                    'question': r.question,
                    'generated_answer': r.generated_answer,
                    'ground_truth': r.ground_truth,
                    'retrieval_metrics': r.retrieval_metrics,
                    'answer_metrics': r.answer_metrics,
                    'latency_ms': r.latency_ms
                }
                for r in self.results
            ]
        }

    def save_results(self, filepath: str) -> None:
        """Sauvegarde les résultats."""
        results = self._aggregate_results()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def print_summary(self) -> None:
        """Affiche un résumé formaté."""
        results = self._aggregate_results()
        summary = results.get('summary', {})
        
        print("\n" + "="*70)
        print("RAG EVALUATION SUMMARY")
        print("="*70)
        print(f"\nEchantillons: {summary.get('num_samples', 0)}")
        print(f"Latence moyenne: {summary.get('avg_latency_ms', 0):.2f} ms")
        
        print("\n" + "-"*70)
        print("RETRIEVAL METRICS")
        print("-"*70)
        for key, value in results.get('retrieval_metrics', {}).items():
            print(f"  {key:20} {value:.4f}")
        
        print("\n" + "-"*70)
        print("ANSWER QUALITY METRICS")
        print("-"*70)
        for key, value in results.get('answer_metrics', {}).items():
            print(f"  {key:20} {value:.4f}")
        print("\n" + "="*70)