"""
tests/test_indexer.py - Test simple Q1
"""
import pytest
from src.indexer import DocumentIndexer


class TestDocumentIndexer:

    def test_index_complete(self):
        """Test la fonction index() avec les PDFs réels"""
        indexer = DocumentIndexer(raw_data_path="data")
        stats = indexer.index(clear=True)
        
        assert stats["total_chunks"] > 0
        assert "chunks_by_source" in stats
        assert len(stats["chunks_by_source"]) > 0