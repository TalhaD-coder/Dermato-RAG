"""
Dermato-RAG Ana Pipeline Modülü.

Bu modül projenin tüm bileşenlerini (Vision Model, RAG, LLM) 
bir araya getirerek uçtan uca tanı desteği akışını yönetir.

Akış:
    Görüntü → Vision Encoder → RAG Retriever → LLM → Tanı Raporu

Kullanım:
    from src.pipeline import DermatoRAGPipeline

    pipeline = DermatoRAGPipeline()
    result = pipeline.analyze(
        image_path="lesion.jpg",
        clinical_info="45 yaş, sırt bölgesi, 3 aydır büyüyen lezyon"
    )
    print(result["diagnosis"])
"""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

from src.models.vision_encoder import DermatoVisionEncoder
from src.llm.generator import DiagnosticGenerator
from src.llm.confidence import ConfidenceEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 9 sınıfın tam isimleri (RAG araması için kullanılır)
CLASS_LABELS = {
    0: "actinic_keratosis",
    1: "basal_cell_carcinoma",
    2: "benign_keratosis",
    3: "dermatofibroma",
    4: "melanoma",
    5: "nevus",
    6: "seborrheic_keratosis",
    7: "squamous_cell_carcinoma",
    8: "vascular_lesion",
}

CLASS_DISPLAY_NAMES = {
    "actinic_keratosis": "Aktinik Keratoz",
    "basal_cell_carcinoma": "Bazal Hücreli Karsinom",
    "benign_keratosis": "Benign Keratoz",
    "dermatofibroma": "Dermatofibrom",
    "melanoma": "Melanom",
    "nevus": "Nevus (Ben)",
    "seborrheic_keratosis": "Seboreli Keratoz",
    "squamous_cell_carcinoma": "Skuamöz Hücreli Karsinom",
    "vascular_lesion": "Vasküler Lezyon",
}


class DermatoRAGPipeline:
    """
    Dermato-RAG uçtan uca tanı desteği pipeline'ı.

    Bileşenler:
    1. Vision Encoder: Görüntüyü sınıflandırır (BiomedCLIP fine-tuned)
    2. RAG Retriever: ChromaDB'den ilgili makaleleri çeker
    3. LLM Generator: Tanı raporunu üretir (Gemini 1.5 Pro)
    4. Confidence Evaluator: Hallüsinasyon kontrolü yapar
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-1.5-pro",
        top_k_articles: int = 5,
        device: Optional[str] = None,
    ) -> None:
        """
        Pipeline'ı başlatır ve tüm bileşenleri yükler.

        Args:
            model_path: Vision model checkpoint yolu. None ise varsayılan konum kullanılır.
            llm_provider: "gemini" veya "openai"
            llm_model: Kullanılacak LLM model adı
            top_k_articles: RAG'dan çekilecek makale sayısı
            device: "cuda", "cpu" veya None (otomatik)
        """
        self.top_k = top_k_articles
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"DermatoRAGPipeline başlatılıyor (device={self.device})")

        # 1. Vision Model'i yükle
        self._load_vision_model(model_path)

        # 2. RAG Retriever'ı yükle
        self._load_rag_retriever()

        # 3. LLM'i başlat
        self._load_llm(llm_provider, llm_model)

        logger.info("Pipeline hazır ✅")

    def _load_vision_model(self, model_path: Optional[str]) -> None:
        """Vision encoder'ı checkpoint'tan yükler."""
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "best_model.pt")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {model_path}\n"
                "Lütfen best_model.pt dosyasını models/ klasörüne koyun."
            )

        logger.info(f"Vision model yükleniyor: {model_path}")
        self.vision_model = DermatoVisionEncoder.load_checkpoint(
            model_path, device=self.device
        )
        self.vision_model.eval()
        logger.info("Vision model yüklendi ✅")

    def _load_rag_retriever(self) -> None:
        """ChromaDB tabanlı RAG retriever'ı yükler."""
        try:
            import chromadb

            db_path = str(PROJECT_ROOT / "data" / "embeddings" / "chroma_db")
            logger.info(f"ChromaDB yükleniyor: {db_path}")

            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name="dermato_rag",
                metadata={"hnsw:space": "cosine"},
            )
            doc_count = self.collection.count()
            logger.info(f"RAG veritabanı yüklendi: {doc_count} doküman ✅")
        except Exception as e:
            logger.warning(f"RAG yüklenemedi: {e}. Literatür desteği olmadan devam edilecek.")
            self.collection = None

    def _load_llm(self, provider: str, model_name: str) -> None:
        """LLM generator ve confidence evaluator'ı başlatır."""
        logger.info(f"LLM başlatılıyor: {provider} / {model_name}")
        self.generator = DiagnosticGenerator(
            provider=provider,
            model_name=model_name,
            temperature=0.1,
        )
        self.evaluator = ConfidenceEvaluator(llm=self.generator.llm)
        logger.info("LLM hazır ✅")

    def _predict_image(self, image_path: str) -> Dict:
        """
        Görüntüyü vision model ile analiz eder.

        Returns:
            {
                "top_class": "melanoma",
                "top_class_display": "Melanom",
                "confidence": 0.82,
                "top3": [("melanoma", 0.82), ("nevus", 0.11), ...],
                "all_probs": {...}
            }
        """
        # Görüntüyü yükle ve ön işle
        image = Image.open(image_path).convert("RGB")

        # BiomedCLIP'in kendi preprocess fonksiyonu varsa kullan
        if hasattr(self.vision_model, '_preprocess') and self.vision_model._preprocess:
            tensor = self.vision_model._preprocess(image).unsqueeze(0).to(self.device)
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = transform(image).unsqueeze(0).to(self.device)

        # Tahmin
        with torch.no_grad():
            logits = self.vision_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # Sonuçları formatla
        prob_dict = {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        top_class = sorted_probs[0][0]
        top_confidence = sorted_probs[0][1]

        return {
            "top_class": top_class,
            "top_class_display": CLASS_DISPLAY_NAMES.get(top_class, top_class),
            "confidence": top_confidence,
            "top3": sorted_probs[:3],
            "all_probs": prob_dict,
        }

    def _retrieve_articles(self, disease_class: str, query: str = "") -> list:
        """
        ChromaDB'den ilgili makaleleri getirir.

        Args:
            disease_class: Sınıf adı (örn: "melanoma")
            query: Ek arama sorgusu

        Returns:
            Makale listesi [{"text": ..., "metadata": {...}}, ...]
        """
        if self.collection is None:
            return []

        try:
            search_query = query or f"{disease_class} diagnosis dermoscopy clinical features"

            results = self.collection.query(
                query_texts=[search_query],
                n_results=self.top_k,
                where={"category": disease_class} if disease_class else None,
            )

            articles = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                    articles.append({"text": doc, "metadata": meta})

            logger.info(f"RAG: {len(articles)} makale getirildi (sınıf: {disease_class})")
            return articles

        except Exception as e:
            logger.warning(f"RAG sorgusu başarısız: {e}")
            return []

    def analyze(
        self,
        image_path: str,
        clinical_info: str = "",
        run_faithfulness_check: bool = True,
    ) -> Dict:
        """
        Dermatolojik görüntüyü analiz eder ve tanı raporu üretir.

        Args:
            image_path: Analiz edilecek görüntünün dosya yolu
            clinical_info: Hasta bilgileri (yaş, semptomlar, lezyon bölgesi vb.)
            run_faithfulness_check: LLM çıktısının hallüsinasyon kontrolü yapılsın mı?

        Returns:
            {
                "vision": {...},       # Vision model sonuçları
                "articles": [...],     # RAG makaleleri
                "diagnosis": "...",    # LLM tanı raporu
                "faithfulness": "..."  # Hallüsinasyon kontrol sonucu (opsiyonel)
            }
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")

        logger.info(f"Analiz başlıyor: {image_path}")

        # ── Adım 1: Vision Model ──────────────────────────────
        logger.info("Adım 1/3: Vision Model tahmini yapılıyor...")
        vision_result = self._predict_image(image_path)
        logger.info(
            f"  → Tahmin: {vision_result['top_class_display']} "
            f"(%{vision_result['confidence']*100:.1f} güven)"
        )

        # ── Adım 2: RAG ───────────────────────────────────────
        logger.info("Adım 2/3: Tıbbi literatür aranıyor...")
        articles = self._retrieve_articles(
            disease_class=vision_result["top_class"],
            query=f"{vision_result['top_class']} {clinical_info}",
        )

        # ── Adım 3: LLM ───────────────────────────────────────
        logger.info("Adım 3/3: LLM tanı raporu üretiliyor...")

        # Vision model çıktısını LLM'e uygun formata çevir
        top3_text = "\n".join([
            f"  - {CLASS_DISPLAY_NAMES.get(cls, cls)}: %{prob*100:.1f}"
            for cls, prob in vision_result["top3"]
        ])
        vision_prediction_text = (
            f"En yüksek olasılıklı tanı: {vision_result['top_class_display']} "
            f"(%{vision_result['confidence']*100:.1f} güven)\n"
            f"İlk 3 tahmin:\n{top3_text}"
        )
        vision_features_text = (
            f"Model: BiomedCLIP fine-tuned (Val Acc: %73.48, Test Acc: %73.00)\n"
            f"Sınıflandırma boyutu: 9 sınıf (ISIC + PAD-UFES veri seti)"
        )

        diagnosis = self.generator.generate_diagnosis(
            clinical_info=clinical_info or "Hasta bilgisi sağlanmadı.",
            vision_prediction=vision_prediction_text,
            vision_features=vision_features_text,
            rag_results=articles,
        )

        result = {
            "vision": vision_result,
            "articles": articles,
            "diagnosis": diagnosis,
        }

        # ── Adım 4: Faithfulness Kontrolü (opsiyonel) ─────────
        if run_faithfulness_check and articles:
            logger.info("Hallüsinasyon kontrolü yapılıyor...")
            faithfulness = self.evaluator.check_faithfulness(
                generated_response=diagnosis,
                rag_results=articles,
            )
            result["faithfulness"] = faithfulness

        logger.info("Analiz tamamlandı ✅")
        return result
