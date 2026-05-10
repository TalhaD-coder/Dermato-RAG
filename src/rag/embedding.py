"""
Dermato-RAG Embedding Modulu.

Metin chunk'larini vektor temsillerine donusturur.
PubMedBERT tabanli sentence-transformers kullanir.

Kullanim:
    from src.rag.embedding import EmbeddingModel
    model = EmbeddingModel()
    vectors = model.encode_texts(["melanoma diagnosis", "skin lesion"])
"""

from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Biyomedikal embedding modelleri (oncelik sirasina gore)
DEFAULT_MODELS = [
    "pritamdeka/S-PubMedBert-MS-MARCO",       # PubMed + MS-MARCO fine-tuned
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",  # PubMedBERT
    "all-MiniLM-L6-v2",                        # Genel amacli (fallback)
]


class EmbeddingModel:
    """
    Metin embedding modeli.

    PubMedBERT veya benzeri biyomedikal modeller ile
    metin chunk'larini vektor temsillerine donusturur.

    Args:
        model_name: HuggingFace model adi.
        device: "cpu", "cuda" veya "auto".
        batch_size: Toplu isleme boyutu.
        normalize: Vektorleri L2 normalize et.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = None
        self.model_name = model_name or DEFAULT_MODELS[0]
        self.dimension = 0

        self._load_model(device)

    def _load_model(self, device: str) -> None:
        """Modeli yukler."""
        try:
            from sentence_transformers import SentenceTransformer

            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Embedding modeli yukleniyor: {self.model_name} ({device})")
            self.model = SentenceTransformer(self.model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model hadir. Boyut: {self.dimension}")

        except Exception as e:
            logger.error(f"Model yuklenemedi ({self.model_name}): {e}")

            # Fallback modeli dene
            for fallback in DEFAULT_MODELS[1:]:
                if fallback == self.model_name:
                    continue
                try:
                    logger.info(f"Fallback deneniyor: {fallback}")
                    self.model = SentenceTransformer(fallback, device=device)
                    self.model_name = fallback
                    self.dimension = self.model.get_sentence_embedding_dimension()
                    logger.info(f"Fallback basarili: {fallback} (dim={self.dimension})")
                    return
                except Exception:
                    continue

            raise RuntimeError("Hicbir embedding modeli yuklenemedi!")

    def encode_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Metin listesini vektor temsillerine donusturur.

        Args:
            texts: Metin listesi.
            show_progress: Ilerleme cubugu goster.

        Returns:
            (N, D) boyutunda numpy array.
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
        )

        return np.array(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Tek bir sorguyu encode eder."""
        return self.encode_texts([query], show_progress=False)[0]

    def similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Cosine benzerlik hesaplar."""
        if self.normalize:
            # Normalizeli vektorlerde dot product = cosine similarity
            return np.dot(doc_vecs, query_vec)
        else:
            # Manuel cosine similarity
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
            return np.dot(doc_norms, query_norm)

    def get_info(self) -> Dict[str, Any]:
        """Model bilgilerini dondurur."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
        }
