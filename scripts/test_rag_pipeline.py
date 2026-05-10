"""
Dermato-RAG RAG Pipeline Test Scripti.

Bilgi tabanini olusturur ve ornek sorgularla test eder.

Kullanim:
    python scripts/test_rag_pipeline.py
    python scripts/test_rag_pipeline.py --build    # sadece build
    python scripts/test_rag_pipeline.py --search   # sadece search
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("test_rag")

# Ornek sorgular
TEST_QUERIES = [
    "What are the dermoscopic features of melanoma?",
    "How to differentiate basal cell carcinoma from squamous cell carcinoma?",
    "Actinic keratosis treatment options and diagnosis",
    "Deep learning methods for skin lesion classification",
    "Seborrheic keratosis vs melanocytic nevus dermoscopy",
]


def build_knowledge_base():
    """Bilgi tabanini olusturur."""
    from src.rag.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    articles_dir = PROJECT_ROOT / "data" / "knowledge_base" / "raw_docs"

    result = kb.build_from_articles(str(articles_dir))
    logger.info(f"\nSonuc: {result['status']}")
    logger.info(f"Toplam chunk: {result.get('total_chunks', 0)}")
    logger.info(f"Embedding dim: {result.get('embedding_dim', 0)}")
    logger.info(f"Model: {result.get('model', '')}")

    return kb


def test_search(kb=None):
    """Ornek sorgularla arama testi."""
    if kb is None:
        from src.rag.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()

    logger.info("\n" + "=" * 60)
    logger.info("ARAMA TESTI")
    logger.info("=" * 60)

    for query in TEST_QUERIES:
        logger.info(f"\nSorgu: {query}")
        logger.info("-" * 50)

        results = kb.search(query, top_k=3)

        for i, r in enumerate(results):
            score = r.get("score", 0)
            title = r["metadata"].get("title", "")[:80]
            category = r["metadata"].get("category", "")
            section = r["metadata"].get("source_section", "")
            text_preview = r["text"][:150].replace("\n", " ")

            logger.info(f"  [{i+1}] Skor: {score:.4f} | {category} | {section}")
            logger.info(f"      Baslik: {title}")
            logger.info(f"      Metin: {text_preview}...")


def test_hybrid_retriever(kb=None):
    """Hibrit retriever testi."""
    if kb is None:
        from src.rag.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()

    from src.rag.retriever import HybridRetriever

    logger.info("\n" + "=" * 60)
    logger.info("HIBRIT RETRIEVER TESTI")
    logger.info("=" * 60)

    retriever = HybridRetriever(knowledge_base=kb)

    query = "melanoma dermoscopic features diagnosis"
    logger.info(f"\nSorgu: {query}")

    for mode in ["dense", "sparse", "hybrid"]:
        logger.info(f"\n--- {mode.upper()} ---")
        results = retriever.retrieve(query, top_k=3, mode=mode)

        for i, r in enumerate(results):
            score = r.get("score", 0)
            title = r["metadata"].get("title", "")[:60]
            logger.info(f"  [{i+1}] Skor: {score:.4f} | {title}")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Test")
    parser.add_argument("--build", action="store_true", help="Sadece build")
    parser.add_argument("--search", action="store_true", help="Sadece search")
    args = parser.parse_args()

    start = time.time()

    if args.search:
        test_search()
        test_hybrid_retriever()
    elif args.build:
        build_knowledge_base()
    else:
        # Full pipeline
        kb = build_knowledge_base()
        test_search(kb)
        test_hybrid_retriever(kb)

    elapsed = time.time() - start
    logger.info(f"\nToplam sure: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
