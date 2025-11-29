📘 RAG Project – Retrieval Augmented Generation (Feature Selection Articles)
🧠 Objectif du projet

Ce projet implémente un pipeline complet de Retrieval-Augmented Generation (RAG) permettant d’interroger efficacement un corpus d’articles scientifiques portant sur le Feature Selection (sélection de variables en Machine Learning).

Nous avons construit :

un pipeline d’indexation avec FAISS

un moteur de recherche sémantique

un module de génération basé sur un LLM

un système d’évaluation avancé (classique + LLM-as-a-Judge)

une interface en ligne de commande (CLI)

Le tout dans une architecture propre et modulaire.

📂 Données utilisées

Nous avons travaillé sur 4 articles scientifiques traitant des techniques de Feature Selection, incluant :

Filter methods

Wrapper methods

Embedded methods

Mutual Information, SFS, SFFS

Hybrid & Ensemble feature selection

Études récentes (Cai et al., Khan et al., etc.)

Ces documents sont placés dans :

data/
    Features_selection_1.pdf
    Features_selection_2.pdf
    Features_selection_3.pdf
    Features_selection_4.pdf

🏗️ Architecture technique
🔹 FAISS

Utilisé comme vector store pour l’indexation efficace des embeddings.
FAISS est rapide, optimisé et standard dans les pipelines RAG.

🔹 LangChain

Framework permettant de gérer :

les loaders

les chunkers

les embeddings

les vector stores

les pipelines de recherche

🔹 Embeddings – all-MiniLM-L6-v2

Nous avons choisi le modèle sentence-transformers/all-MiniLM-L6-v2 pour trois raisons :

Très bonne qualité des représentations sémantiques dans les tâches QA.

Faible coût computationnel (modèle léger → rapide pour FAISS).

Recommandé dans les systèmes RAG pour des documents courts/moyens.

🔹 LLM (Groq API)

Pour la génération finale et l’évaluation LLM-as-a-Judge, nous utilisons un modèle LLama 3.1 via l’API Groq (inférence ultra rapide).

⚙️ Fonctionnalités du projet
✔ 1) Indexation des documents

Extraction PDF (PyPDF)

Chunking (500–800 tokens)

Embedding MiniLM

Stockage FAISS (persistant)

✔ 2) Recherche sémantique

Similarité cosine

Récupération des top-k chunks

Score FAISS + métadonnées (page, source)

✔ 3) Génération de réponse (RAG)

Contexte = top-k chunks

Prompt structuré “question + contexte”

Modèle Groq (LLama 3.1)

✔ 4) Évaluation complète

Nous avons évalué le système sur 15 questions (Human feedback).

🔹 Évaluation du retrieval (retrieval quality)

Sur les 15 questions :

Recall@4 = 0.867
→ dans 86.7% des cas, au moins un chunk pertinent est présent dans les 4 premiers résultats.

Precision@4 = 0.450
→ en moyenne, 45% des chunks retournés sont réellement pertinents.

➡️ Interprétation :
Le pipeline de récupération est très bon (haut recall), mais retourne parfois un peu de bruit (precision moyenne).
Ce comportement est attendu avec MiniLM (embedding léger).

📊 Évaluation des réponses générées

Nous évaluons la qualité des réponses selon 4 métriques :

🔹 1) ROUGE-L : 0.28

ROUGE compare la similarité entre réponse générée et référence humaine.

→ Score modéré : acceptable pour un LLM utilisant un contexte chunké.

🔹 2) Cosine Similarity (embeddings) : 0.70

Similitude sémantique entre la réponse générée et la réponse idéale.
→ 0.70 indique que la réponse est globalement dans le bon sujet.

🧠 Évaluation LLM-as-a-Judge (Groq)

Nous avons ajouté deux métriques avancées, essentielles en RAG :

🔹 3) Faithfulness (Fidélité au contexte) : 0.77

Mesure :

Est-ce que le modèle invente des informations non présentes dans les chunks ?

Calculé avec Llama 3.1 (Groq).
Un score de 0.77 indique très peu d’hallucinations.

🔹 4) Answer Relevance (Pertinence de la réponse) : 0.60

Mesure :

Est-ce que la réponse répond vraiment à la question ?

Score correct, mais montre que certaines réponses sont :

trop générales

trop courtes

ou s’éloignent légèrement de l’intention de la question

🎯 Pourquoi ces métriques ?
Métrique	Pourquoi ?	Rôle
Recall@k	Vérifie si on récupère la bonne info	Qualité du retrieval
Precision@k	Vérifie si le contexte est propre	Bruit dans FAISS
ROUGE-L	Compare réponse vs référence	Surface-level correctness
Cosine similarity	Vérifie la proximité sémantique	Deep meaning correctness
Faithfulness (LLM judge)	Vérifie les hallucinations	Fiabilité
Answer Relevance (LLM judge)	Vérifie l’adéquation	Pertinence réelle

➡️ Ensemble, ces métriques donnent une vision complète du RAG (retrieval + generation).

🖥️ CLI (Command Line Interface)

Un script cli.py permet d'utiliser le système depuis le terminal :

📌 Indexer :
python cli.py index --config config.yaml

📌 Poser une question :
python cli.py ask --config config.yaml --question "What is SFS?"

📌 Évaluer :
python cli.py evaluate --config config.yaml --k 4 --output results.json

📦 Installation
pip install -r requirements.txt


Créer la variable d’environnement :

export GROQ_API_KEY="VOTRE_CLÉ"

🧮 Arborescence du projet
RAG-Project/
    cli.py
    config.yaml
    data/
    storage/
    src/
        indexer.py
        search.py
        rag.py
        evaluator.py
        llm_judge.py
    results/
    README.md

🏁 Conclusion

Ce projet met en place un vrai pipeline RAG complet, évalué, fiable et bien structuré.
Nous avons construit :

🔍 un bon retrieval (R@4 = 0.867)

🧠 un modèle génératif cohérent

📊 une évaluation avancée (classique + LLM judge)

🛠️ un outil CLI professionnel

Tu peux l’utiliser comme base pour :

des projets industriels

des chatbots documentaires

de la recherche appliquée

des systèmes QA avancés
