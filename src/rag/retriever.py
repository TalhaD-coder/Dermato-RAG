"""
Dermato-RAG Retriever Modulu.

Hibrit arama: Dense (embedding) + Sparse (BM25) birlestirme.
ChromaDB'den semantik + BM25 ile keyword arama yapar ve
sonuclari birlestirerek en ilgili dokumanlari getirir.

Kullanim:
    from src.rag.retriever import HybridRetriever
    retriever = HybridRetriever(knowledge_base=kb)
    results = retriever.retrieve("melanoma dermoscopy", top_k=10)
"""

from typing import Any, Dict, List, Optional

import numpy as np

from src.rag.knowledge_base import KnowledgeBase
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hibrit retriever: Dense + Sparse birlestirme.

    Dense arama: ChromaDB uzerinden cosine similarity.
    Sparse arama: BM25 ile keyword matching.
    Sonuclar Reciprocal Rank Fusion (RRF) ile birlestirilir.

    Args:
        knowledge_base: KnowledgeBase instance.
        dense_weight: Dense sonuclarin agirligi (0-1).
        sparse_weight: Sparse sonuclarin agirligi (0-1).
        rrf_k: RRF parametresi (tipik: 60).
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> None:
        self.kb = knowledge_base
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

        # BM25 index (lazy build)
        self._bm25 = None
        self._bm25_corpus_ids = None
        self._bm25_corpus_docs = None
        self._bm25_corpus_metas = None

    def _build_bm25_index(self) -> None:
        """BM25 indeksini olusturur."""
        from rank_bm25 import BM25Okapi

        logger.info("BM25 indeksi olusturuluyor...")

        # ChromaDB'den tum dokumanlari al
        collection = self.kb.collection
        count = collection.count()

        if count == 0:
            logger.warning("Koleksiyon bos, BM25 olusturulamadi!")
            return

        # Tum dokumanlari cek (batch halinde)
        all_ids = []
        all_docs = []
        all_metas = []
        batch_size = 500

        for offset in range(0, count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            all_ids.extend(result["ids"])
            all_docs.extend(result["documents"])
            all_metas.extend(result["metadatas"])

        # Tokenize (basit whitespace + lowercase)
        tokenized = [doc.lower().split() for doc in all_docs]

        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus_ids = all_ids
        self._bm25_corpus_docs = all_docs
        self._bm25_corpus_metas = all_metas

        logger.info(f"BM25 indeksi hazir: {len(all_docs)} dokuman")

    def _dense_search(
        self,
        query: str,
        top_k: int,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """ChromaDB uzerinden dense arama."""
        return self.kb.search(
            query=query,
            top_k=top_k,
            category_filter=category_filter,
        )

    def _sparse_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """BM25 ile sparse arama."""
        if self._bm25 is None:
            self._build_bm25_index()

        if self._bm25 is None:
            return []

        # Query tokenize
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # Top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self._bm25_corpus_docs[idx],
                    "metadata": self._bm25_corpus_metas[idx],
                    "score": float(scores[idx]),
                    "chunk_id": self._bm25_corpus_ids[idx],
                })

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        """
        Hibrit arama ile en ilgili dokumanlari getirir.

        Args:
            query: Arama sorgusu.
            top_k: Dondurulecek sonuc sayisi.
            category_filter: Kategori filtresi.
            mode: "hybrid", "dense" veya "sparse".

        Returns:
            Sonuc listesi [{text, metadata, score}, ...].
        """
        if mode == "dense":
            return self._dense_search(query, top_k, category_filter)

        if mode == "sparse":
            return self._sparse_search(query, top_k)

        # Hybrid: Dense + Sparse + RRF
        # Daha fazla sonuc cek ve birlestir
        fetch_k = top_k * 3

        dense_results = self._dense_search(query, fetch_k, category_filter)
        sparse_results = self._sparse_search(query, fetch_k)

        # Reciprocal Rank Fusion (RRF)
        fused = self._rrf_fusion(dense_results, sparse_results, top_k)

        return fused

    def _rrf_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion ile sonuclari birlestir.

        RRF skoru = sum(1 / (k + rank)) her kaynak icin.
        """
        scores = {}  # text_hash -> {score, result}

        # Dense sonuclari
        for rank, result in enumerate(dense_results):
            key = result["text"][:200]  # Ilk 200 karakter = unique key
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)

            if key not in scores:
                scores[key] = {"score": 0, "result": result}
            scores[key]["score"] += rrf_score

        # Sparse sonuclari
        for rank, result in enumerate(sparse_results):
            key = result["text"][:200]
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)

            if key not in scores:
                scores[key] = {"score": 0, "result": result}
            scores[key]["score"] += rrf_score

        # Sirala ve dondur
        sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

        final = []
        for item in sorted_results[:top_k]:
            result = item["result"]
            result["score"] = item["score"]
            final.append(result)

        return final
