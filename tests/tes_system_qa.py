"""
tests/test_system_qa.py 
"""
import pytest
import os
from src.system_qa import QASystem


class TestQASystem:

    @pytest.fixture
    def hf_api_key(self):
        """Récupère la clé API depuis les variables d'environnement"""
        return os.getenv("HF_API_KEY", "test-key")

    @pytest.fixture
    def qa_system(self, hf_api_key):
        """Crée une instance de QASystem"""
        return QASystem(hf_api_key=hf_api_key)

    def test_retrieve_context(self, qa_system):
        """Test la récupération du contexte"""
        documents = qa_system.retrieve_context("philosophie", top_k=2)
        
        assert len(documents) > 0
        assert len(documents) <= 2
        assert all(hasattr(doc, 'page_content') for doc, _ in documents)

    def test_format_context(self, qa_system):
        """Test le formatage du contexte"""
        documents = qa_system.retrieve_context("Marc Sautet", top_k=2)
        formatted = qa_system.format_context(documents)
        
        assert "CONTEXTE EXTRAIT" in formatted
        assert "Source:" in formatted
        assert "Pertinence:" in formatted

    def test_create_prompt(self, qa_system):
        """Test la création du prompt"""
        documents = qa_system.retrieve_context("philosophie", top_k=1)
        context = qa_system.format_context(documents)
        prompt = qa_system.create_prompt("Qu'est-ce que la philosophie?", context)
        
        assert "Qu'est-ce que la philosophie?" in prompt
        assert "CONTEXTE EXTRAIT" in prompt
        assert "INSTRUCTIONS:" in prompt

    def test_query_complete(self, qa_system):
        """Test le pipeline complet"""
        result = qa_system.query("Qu'est-ce qu'un café philo?", top_k=2)
        
        assert "question" in result
        assert "answer" in result
        assert "context_documents" in result
        assert "sources" in result
        assert result["question"] == "Qu'est-ce qu'un café philo?"