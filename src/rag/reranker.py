"""
Dermato-RAG Re-Ranker Modulu.

Cross-encoder tabanli yeniden siralama.
Retriever'dan gelen sonuclari query-document ciftleri
olarak degerlendirip daha isabetli siralama yapar.

Kullanim:
    from src.rag.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, results, top_k=5)
"""

from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cross-encoder modelleri
DEFAULT_RERANKER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",    # Hizli, iyi performans
    "cross-encoder/ms-marco-MiniLM-L-12-v2",    # Daha buyuk, daha iyi
]


class CrossEncoderReranker:
    """
    Cross-encoder tabanli re-ranker.

    Retriever sonuclarini query-document cifti olarak
    cross-encoder'a gonderip daha isabetli skorlar uretir.

    Args:
        model_name: Cross-encoder model adi.
        device: "cpu", "cuda" veya "auto".
        max_length: Maksimum token uzunlugu.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name or DEFAULT_RERANKER_MODELS[0]
        self.max_length = max_length
        self.model = None

        self._load_model(device)

    def _load_model(self, device: str) -> None:
        """Cross-encoder modelini yukler."""
        try:
            from sentence_transformers import CrossEncoder

            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Re-ranker yukleniyor: {self.model_name} ({device})")
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=device,
            )
            logger.info("Re-ranker hazir.")

        except Exception as e:
            logger.warning(f"Re-ranker yuklenemedi: {e}")
            logger.warning("Re-ranking devre disi, orijinal siralama kullanilacak.")
            self.model = None

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retriever sonuclarini yeniden siralar.

        Args:
            query: Arama sorgusu.
            results: Retriever sonuclari [{text, metadata, score}, ...].
            top_k: Dondurulecek sonuc sayisi.

        Returns:
            Yeniden siralanmis sonuc listesi.
        """
        if not results:
            return []

        if self.model is None:
            logger.debug("Re-ranker mevcut degil, orijinal siralama donduruldu.")
            return results[:top_k]

        # Query-document ciftleri olustur
        pairs = [[query, r["text"]] for r in results]

        # Cross-encoder skorlari
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Sonuclara skor ekle ve sirala
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
                result["original_score"] = result.get("score", 0)

            reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logger.warning(f"Re-ranking hatasi: {e}")
            return results[:top_k]

    def score_pair(self, query: str, document: str) -> float:
        """Tek bir query-document cifti icin skor dondurur."""
        if self.model is None:
            return 0.0
        try:
            return float(self.model.predict([[query, document]])[0])
        except Exception:
            return 0.0
