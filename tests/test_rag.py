import pytest
from src.rag.chunking import TextChunk, ArticleChunker

def test_text_chunk():
    chunk = TextChunk("Test metni", {"pmid": "123"})
    assert chunk.text == "Test metni"
    assert chunk.metadata["pmid"] == "123"
    assert len(chunk) == 10

def test_article_chunker():
    chunker = ArticleChunker(chunk_size=100, chunk_overlap=20)
    article = {
        "pmid": "123",
        "title": "Melanoma",
        "abstract": "This is a short abstract about melanoma diagnosis.",
        "category": "melanoma"
    }
    
    chunks = chunker.chunk_article(article)
    assert len(chunks) > 0
    assert chunks[0].metadata["pmid"] == "123"
    assert chunks[0].metadata["source_section"] == "abstract"
