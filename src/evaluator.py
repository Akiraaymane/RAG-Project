import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

from groq import Groq
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

from src.rag import RAGPipeline
from src.search import DocumentSearcher
from src.llm_judge import LLMJudge
# ---------- 1. Gold questions + réponses de référence ----------

@dataclass
class QAExample:
    question: str
    reference_answer: str


def get_gold_qa() -> List[QAExample]:
    """
    Les 15 questions + réponses 'gold' pour évaluer le RAG.
    Tu peux ajuster le wording, mais garde la structure.
    """
    return [
        QAExample(
            question="What is the main goal of feature selection in machine learning?",
            reference_answer=(
                "The main goal of feature selection is to reduce dimensionality by removing "
                "irrelevant and redundant features, in order to improve predictive performance, "
                "speed up learning, and sometimes improve interpretability."
            ),
        ),
        QAExample(
            question="What are the three classical categories of feature selection methods?",
            reference_answer=(
                "The three classical categories are filter methods, wrapper methods, "
                "and embedded methods."
            ),
        ),
        QAExample(
            question="How do filter methods select features, and what is their main advantage?",
            reference_answer=(
                "Filter methods assign a relevance score to each feature using a criterion "
                "independent of the classifier, then select the top-ranked ones. "
                "Their main advantage is low computational cost and good scalability "
                "to high-dimensional data."
            ),
        ),
        QAExample(
            question="What characterizes wrapper methods, and what is their main drawback?",
            reference_answer=(
                "Wrapper methods evaluate subsets of features by training a predictor "
                "and using its performance as the objective function. "
                "They are powerful but their main drawback is the high computational cost "
                "and the risk of overfitting."
            ),
        ),
        QAExample(
            question="What is the key idea behind embedded methods such as mRMR (max-relevancy, min-redundancy)?",
            reference_answer=(
                "Embedded methods integrate feature selection into the training of the model itself. "
                "mRMR in particular selects features that are highly relevant to the target while being "
                "as little redundant as possible with respect to each other, often using mutual information."
            ),
        ),
        QAExample(
            question="What is the Pearson correlation coefficient used for in feature selection?",
            reference_answer=(
                "The Pearson correlation coefficient is used to measure linear correlation between variables, "
                "for example to detect redundant features that are highly correlated and therefore carry "
                "overlapping information."
            ),
        ),
        QAExample(
            question="How is Mutual Information (MI) interpreted in the context of feature selection?",
            reference_answer=(
                "Mutual Information measures how much knowing a feature reduces uncertainty about the class. "
                "In feature selection it is used to quantify feature relevance and, in extended forms, "
                "to control redundancy and interactions among features."
            ),
        ),
        QAExample(
            question="What is Sequential Forward Selection (SFS) and how does it work?",
            reference_answer=(
                "Sequential Forward Selection is a greedy wrapper strategy that starts from an empty set "
                "and iteratively adds the feature whose inclusion yields the largest improvement of the "
                "objective function, until a stopping criterion is met."
            ),
        ),
        QAExample(
            question="What problem does Sequential Floating Forward Selection (SFFS) try to solve compared to naive SFS?",
            reference_answer=(
                "SFFS tries to alleviate the nesting effect of naive SFS, where once a feature is added "
                "it can never be removed. SFFS alternates forward inclusion with conditional backward "
                "elimination to remove previously added features if that improves the objective."
            ),
        ),
        QAExample(
            question="How are supervised, unsupervised, and semi-supervised feature selection defined?",
            reference_answer=(
                "Supervised feature selection uses labeled data to exploit the relationship between features "
                "and class labels. Unsupervised feature selection works on unlabeled data, usually preserving "
                "the intrinsic structure such as clusters. Semi-supervised feature selection uses both a small "
                "labeled set and a larger unlabeled set to guide the selection."
            ),
        ),
        QAExample(
            question="What are hybrid and ensemble feature selection strategies?",
            reference_answer=(
                "Hybrid strategies combine different feature selection methods in multiple stages "
                "(for example filter plus wrapper). Ensemble feature selection combines the outputs of "
                "multiple feature selection algorithms, such as aggregating their rankings or subsets, "
                "to obtain a more robust and accurate feature set."
            ),
        ),
        QAExample(
            question="Why is the stability of feature selection algorithms an important issue?",
            reference_answer=(
                "Stability measures how consistent the selected feature set is under small changes in the data. "
                "It is crucial because unstable feature selection leads to unreliable models and biomarkers, "
                "especially in domains like bioinformatics where interpretability and reproducibility matter."
            ),
        ),
        QAExample(
            question="For the UNSW-NB15 intrusion detection dataset, what is the practical effect of using feature importance?",
            reference_answer=(
                "Using feature importance with a Random Forest reduces the original 41 attributes "
                "to a smaller subset of important features, which improves classification accuracy "
                "and reduces computation and prediction time on the UNSW-NB15 dataset."
            ),
        ),
        QAExample(
            question="Which classifiers are commonly used to evaluate feature selection methods?",
            reference_answer=(
                "Common classifiers include Support Vector Machines, k-Nearest Neighbors, decision trees, "
                "Random Forests and other ensemble methods, which are used to compare the predictive "
                "performance of different feature selection techniques."
            ),
        ),
        QAExample(
            question="Why can filter methods give irregular performance curves as the number of features increases?",
            reference_answer=(
                "Because filter methods rank features individually and ignore interactions and redundancy, "
                "adding more top-ranked features does not guarantee monotonic performance improvements. "
                "This can lead to irregular performance curves as the number of selected features grows."
            ),
        ),
    ]


# ---------- 2. Évaluation complète : ROUGE / Cosine + Faithfulness / Relevance ----------

def evaluate_rag_all(
    config_path: str,
    k_retrieval: int = 4,
    output_json_path: str = "rag_eval_all.json",
) -> None:
    """
    Pipeline complet d'évaluation :
    - Génère les réponses RAG pour les 15 questions
    - Calcule ROUGE-L F1 + cosine similarity (réf vs prédiction)
    - Calcule Faithfulness + Answer Relevance via Groq (LLM-judge)
    - Sauvegarde tout dans un JSON unique
    """

    # RAG pipeline + searcher pour le contexte
    rag = RAGPipeline(config_path)
    searcher = DocumentSearcher(config_path=config_path)
    searcher.load_vectorstore()


    # LLM-judge (Groq)
    judge = LLMJudge()

    # Embeddings + ROUGE pour les métriques classiques
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    qa_list = get_gold_qa()
    results: List[Dict[str, Any]] = []

    sum_rougeL = 0.0
    sum_cosine = 0.0
    sum_faithfulness = 0.0
    sum_relevance = 0.0

    for idx, qa in enumerate(qa_list, start=1):
        question = qa.question
        reference = qa.reference_answer

        print(f"[INFO] Evaluating question {idx}/{len(qa_list)}")

        # 1) Contexte RAG (k chunks) pour Faithfulness
        chunks_with_scores = searcher.search_with_scores(question, k=k_retrieval)
        context_texts = [doc.page_content for doc, _score in chunks_with_scores]
        context = "\n\n---\n\n".join(context_texts)

        # 2) Réponse générée par le RAG
        prediction = rag.ask(question, k=k_retrieval).strip()

        # 3) ROUGE-L F1
        rouge_scores = rouge.score(reference, prediction)
        rougeL_f1 = rouge_scores["rougeL"].fmeasure

        # 4) Similarité cosinus entre embeddings (réf vs prédiction)
        emb_ref = embedder.encode(reference, convert_to_tensor=True)
        emb_pred = embedder.encode(prediction, convert_to_tensor=True)
        cosine_sim = util.cos_sim(emb_ref, emb_pred).item()

        # 5) LLM-judge : Faithfulness + Answer Relevance
        faith = judge.faithfulness_score(question=question, answer=prediction, context=context)
        rel = judge.answer_relevance_score(question=question, answer=prediction)

        # 6) Accumulation des moyennes
        sum_rougeL += rougeL_f1
        sum_cosine += cosine_sim
        sum_faithfulness += faith["score"]
        sum_relevance += rel["score"]

        # 7) Stockage détaillé
        results.append(
            {
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "rougeL_f1": rougeL_f1,
                "cosine_similarity": cosine_sim,
                "faithfulness_score": faith["score"],
                "faithfulness_explanation": faith["explanation"],
                "answer_relevance_score": rel["score"],
                "answer_relevance_explanation": rel["explanation"],
            }
        )

    n = len(qa_list)
    summary = {
        "avg_rougeL_f1": sum_rougeL / n,
        "avg_cosine_similarity": sum_cosine / n,
        "avg_faithfulness": sum_faithfulness / n,
        "avg_answer_relevance": sum_relevance / n,
    }

    output = {
        "results": results,
        "summary": summary,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("[INFO] Evaluation done.")
    print(f"  avg_rougeL_f1:        {summary['avg_rougeL_f1']:.4f}")
    print(f"  avg_cosine_similarity:{summary['avg_cosine_similarity']:.4f}")
    print(f"  avg_faithfulness:     {summary['avg_faithfulness']:.4f}")
    print(f"  avg_answer_relevance: {summary['avg_answer_relevance']:.4f}")
    print(f"[INFO] Saved to {output_json_path}")


# ---------- 3. Main ----------

if __name__ == "__main__":
    evaluate_rag_all(
        config_path="config.yaml",
        k_retrieval=4,
        output_json_path="rag_eval_all.json",
    )
