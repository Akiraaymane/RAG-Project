"""
tests/test_retriever.py - Test retriever
"""
import pytest
from src.retriever import DocumentRetriever


class TestDocumentRetriever:

    def test_search_complete(self):
        """Test la fonction search() avec les pdf du repo data"""
        retriever = DocumentRetriever(vector_db_path="data/vector_db")
        results = retriever.search("philosophie", top_k=3)
        
        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(result[1], float) for result in results)

    def test_search_returns_document_and_score(self):
        """Vérifie que search retourne (Document, score)"""
        retriever = DocumentRetriever(vector_db_path="data/vector_db")
        results = retriever.search("café", top_k=1)
        
        assert len(results) == 1
        doc, score = results[0]
        assert doc.page_content
        assert 0 <= score <= 2

    def test_convert_l2_to_relevance(self):
        """Teste la conversion L2 → pertinence"""
        retriever = DocumentRetriever()
        
        assert retriever.convert_l2_to_relevance(0.0) == 1.0
        assert retriever.convert_l2_to_relevance(1.0) == 0.5
        assert retriever.convert_l2_to_relevance(2.0) == 0.0