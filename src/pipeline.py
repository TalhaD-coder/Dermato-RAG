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

import math
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
    "seborrheic_keratosis": "Seboreik Keratoz",
    "squamous_cell_carcinoma": "Skuamöz Hücreli Karsinom",
    "vascular_lesion": "Vasküler Lezyon",
}

CLASS_DISPLAY_NAMES_EN = {
    "actinic_keratosis": "Actinic Keratosis",
    "basal_cell_carcinoma": "Basal Cell Carcinoma",
    "benign_keratosis": "Benign Keratosis",
    "dermatofibroma": "Dermatofibroma",
    "melanoma": "Melanoma",
    "nevus": "Nevus (Mole)",
    "seborrheic_keratosis": "Seborrheic Keratosis",
    "squamous_cell_carcinoma": "Squamous Cell Carcinoma",
    "vascular_lesion": "Vascular Lesion",
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

    # RAG kategorisinde yeterli unique makale yoksa devreye giren yedek kategori
    # (PubMed makaleleri indirildikten sonra direkt aynı kategori kullanılır.
    #  Bu yalnız incremental veri eksikliği durumunda kullanılır.)
    RAG_CATEGORY_FALLBACK = {
        "seborrheic_keratosis": "benign_keratosis",  # Yedek olarak benign keratosis ailesi
    }
    # Yedek kategori devreye girmesi için minimum unique makale eşiği
    MIN_UNIQUE_ARTICLES = 3

    def __init__(
        self,
        model_path: Optional[str] = None,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-2.5-flash",
        top_k_articles: int = 5,
        device: Optional[str] = None,
        retrieval_mode: str = "dense",       # "dense" | "hybrid"
        use_reranker: bool = False,
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
        self.retrieval_mode = retrieval_mode
        self.use_reranker = use_reranker
        self.retriever = None  # Lazy-set if hybrid mode
        self.reranker = None   # Lazy-set if reranker enabled

        logger.info(
            f"DermatoRAGPipeline başlatılıyor "
            f"(device={self.device}, retrieval={self.retrieval_mode}, reranker={self.use_reranker})"
        )

        # 1. Vision Model'i yükle
        self._load_vision_model(model_path)

        # 2. RAG Retriever'ı yükle
        self._load_rag_retriever()

        # 3. LLM'i başlat
        self._load_llm(llm_provider, llm_model)

        # 4. CLIP zero-shot OOD dedektörü (cilt mi / değil mi)
        self._load_ood_detector()

        logger.info("Pipeline hazir")

    def _load_ood_detector(self) -> None:
        """
        BiomedCLIP zero-shot ile cilt/dermatoloji içerip içermediğini
        kontrol eden OOD dedektörü. Kedi, ağaç, manzara gibi alakasız
        görüntüleri reddetmek için kullanılır.
        """
        self._ood_clip_model = None
        self._ood_clip_preprocess = None
        self._ood_text_features = None
        try:
            import open_clip

            logger.info("OOD dedektörü (BiomedCLIP zero-shot) yükleniyor...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            tokenizer = open_clip.get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            model = model.to(self.device).eval()

            # Referans prompt'lar — ilk 4 = cilt/dermatoloji, son 5 = alakasız
            self._ood_text_prompts = [
                "a dermatoscopic image of a skin lesion",
                "a clinical close-up photograph of human skin",
                "a photograph of a mole or skin growth",
                "a photograph of human skin with a lesion or rash",
                "a photograph of a cat or other animal",
                "a photograph of a tree, plant, or flower",
                "a photograph of an outdoor landscape or scene",
                "a photograph of a car, building, or man-made object",
                "a photograph of food or everyday objects",
            ]
            self._ood_skin_indices = {0, 1, 2, 3}  # ilk 4 cilt

            text_tokens = tokenizer(self._ood_text_prompts).to(self.device)
            with torch.no_grad():
                tf = model.encode_text(text_tokens)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                self._ood_text_features = tf

            self._ood_clip_model = model
            self._ood_clip_preprocess = preprocess
            logger.info(
                f"OOD dedektörü hazır ({len(self._ood_text_prompts)} prompt, "
                f"{len(self._ood_skin_indices)} cilt sınıfı)"
            )
        except Exception as e:
            logger.warning(
                f"OOD dedektörü yüklenemedi, basit eşik fallback kullanılacak: {e}"
            )
            self._ood_clip_model = None

    def _clip_is_dermatologic(self, image: Image.Image) -> tuple:
        """
        Görüntünün cilt/dermatoloji içerip içermediğini CLIP zero-shot ile
        kontrol eder. Geri dönüş: (is_skin, best_prompt, best_score).
        Hata olursa (True, "", 0.0) döner (güvenli fallback).
        """
        if self._ood_clip_model is None or self._ood_text_features is None:
            return (True, "", 0.0)
        try:
            img_tensor = self._ood_clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feat = self._ood_clip_model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                # Cosine similarity → softmax
                sims = (100.0 * img_feat @ self._ood_text_features.T).softmax(dim=-1)
                best_idx = int(sims.argmax(dim=-1).item())
                best_score = float(sims[0, best_idx].item())
            is_skin = best_idx in self._ood_skin_indices
            best_prompt = self._ood_text_prompts[best_idx]
            logger.info(
                f"CLIP-OOD check: best='{best_prompt}' ({best_score:.3f}) "
                f"is_skin={is_skin}"
            )
            return (is_skin, best_prompt, best_score)
        except Exception as e:
            logger.warning(f"CLIP-OOD check hatası: {e} — kabul ediliyor")
            return (True, "", 0.0)

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
        logger.info("Vision model yuklendi")

    def _load_rag_retriever(self) -> None:
        """ChromaDB tabanlı RAG retriever'ı (ve istenmişse hybrid + reranker) yükler."""
        try:
            from src.rag.knowledge_base import KnowledgeBase

            db_path = str(PROJECT_ROOT / "data" / "embeddings" / "chromadb")
            logger.info(f"RAG Knowledge Base yükleniyor: {db_path}")

            self.kb = KnowledgeBase(
                collection_name="dermato_kb",
                persist_dir=db_path,
                embedding_model="pritamdeka/S-PubMedBert-MS-MARCO"
            )
            doc_count = self.kb.collection.count()
            logger.info(f"RAG veritabanı yüklendi: {doc_count} doküman")

            # Hybrid retriever (opsiyonel)
            if self.retrieval_mode == "hybrid":
                try:
                    from src.rag.retriever import HybridRetriever
                    self.retriever = HybridRetriever(knowledge_base=self.kb)
                    logger.info("HybridRetriever (dense + BM25 + RRF) aktif")
                except Exception as e:
                    logger.warning(f"HybridRetriever yüklenemedi, dense'e düşülüyor: {e}")
                    self.retriever = None
                    self.retrieval_mode = "dense"

            # Cross-encoder reranker (opsiyonel)
            if self.use_reranker:
                try:
                    from src.rag.reranker import CrossEncoderReranker
                    self.reranker = CrossEncoderReranker()
                    logger.info("CrossEncoderReranker aktif")
                except Exception as e:
                    logger.warning(f"Reranker yüklenemedi: {e}")
                    self.reranker = None
                    self.use_reranker = False

        except Exception as e:
            logger.warning(f"RAG yüklenemedi: {e}. Literatür desteği olmadan devam edilecek.")
            self.kb = None

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
        Görüntüyü vision model ile analiz eder (TTA — 6 augmentation ortalaması).

        Returns:
            {
                "top_class": "melanoma",
                "top_class_display": "Melanom",
                "confidence": 0.82,
                "top3": [("melanoma", 0.82), ("nevus", 0.11), ...],
                "all_probs": {...}
            }
        """
        from PIL import ImageOps

        image = Image.open(image_path).convert("RGB")

        # Preprocess fonksiyonunu belirle
        if hasattr(self.vision_model, '_preprocess') and self.vision_model._preprocess:
            preprocess = self.vision_model._preprocess
        else:
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        # TTA: 6 augmented versiyon
        tta_images = [
            image,
            ImageOps.mirror(image),
            ImageOps.flip(image),
            ImageOps.mirror(ImageOps.flip(image)),
            image.rotate(90),
            image.rotate(270),
        ]

        all_probs_tta = []
        with torch.no_grad():
            for img in tta_images:
                tensor = preprocess(img).unsqueeze(0).to(self.device)
                logits = self.vision_model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                all_probs_tta.append(probs)

        # TTA olasılıklarını ortala
        avg_probs = torch.stack(all_probs_tta).mean(dim=0)

        prob_dict = {CLASS_LABELS[i]: float(avg_probs[i]) for i in range(len(CLASS_LABELS))}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        top_class = sorted_probs[0][0]
        top_confidence = sorted_probs[0][1]

        # ============================================================
        # OOD tespiti — iki katmanlı
        # ============================================================
        # 1) CLIP zero-shot: görüntü cilt/dermatoloji içeriyor mu?
        #    (Kedi, ağaç, manzara, eşya vb. burada yakalanır.)
        # 2) Confidence/entropy: model çok kararsız mı?
        # ============================================================
        is_skin, _ood_best_prompt, _ood_best_score = self._clip_is_dermatologic(image)

        # Confidence/entropy fallback kriteri
        _entropy_norm = 0.0
        try:
            _eps = 1e-10
            _probs_list = [float(prob_dict[c]) for c in CLASS_LABELS]
            _entropy = -sum(p * math.log(p + _eps) for p in _probs_list)
            _max_entropy = math.log(max(len(CLASS_LABELS), 2))
            _entropy_norm = _entropy / _max_entropy if _max_entropy > 0 else 0.0
        except Exception as _ood_err:
            logger.warning(f"OOD entropi hesabi basarisiz: {_ood_err}")

        is_ood = (
            (not is_skin)                       # CLIP "cilt değil" diyor
            or (top_confidence < 0.30)          # Çok düşük güven
            or (_entropy_norm > 0.92)           # Aşırı dağıtık olasılıklar
        )

        if is_ood:
            logger.info(
                f"OOD tespit edildi | is_skin={is_skin} "
                f"(clip='{_ood_best_prompt}' score={_ood_best_score:.3f}) "
                f"conf={top_confidence:.3f} entropy_norm={_entropy_norm:.3f}"
            )

        return {
            "top_class": top_class,
            "top_class_display": CLASS_DISPLAY_NAMES.get(top_class, top_class),
            "confidence": top_confidence,
            "top3": sorted_probs[:3],
            "all_probs": prob_dict,
            "is_ood": is_ood,
        }

    def _retrieve_articles(self, disease_class: str, query: str = "") -> list:
        """
        ChromaDB'den ilgili makaleleri getirir.

        Özellikler:
        - PMID-bazlı dedup: aynı makaleden yalnız 1 chunk'ı top-K'ya alınır
        - Kategori fallback: RAG'da yetersiz makale olan sınıflar için
          (örn. seborrheic_keratosis → benign_keratosis)
        - Hybrid mod: dense + BM25 + RRF (retrieval_mode="hybrid")
        - Reranker: cross-encoder rerank (use_reranker=True)

        Args:
            disease_class: Vision modelin tahmin ettiği sınıf adı
            query: Klinik bilgilerle zenginleştirilmiş ek arama sorgusu

        Returns:
            Makale listesi (en fazla self.top_k unique makale).
        """
        if getattr(self, "kb", None) is None:
            return []

        # 1) Önce orijinal sınıfı dene; yetersiz unique makale gelirse aşağıda yedek kategoriye geçilecek
        effective_category = disease_class

        # Klinik query'yi de kullan, aksi halde varsayılan template
        base_query = (query or "").strip()
        if not base_query:
            base_query = f"Dermoscopic features and clinical management of {effective_category}."
        else:
            # Klinik bilgiyi sınıf adıyla birleştir
            base_query = f"{effective_category} {base_query} dermoscopic features management"

        # 2) Daha geniş bir aday havuzu çek (dedup sonrası top_k kalsın diye)
        # Aynı makalenin chunk'ları çakıştığı için 3-5x oversample yapıyoruz.
        oversample_k = max(self.top_k * 4, 20)
        try:
            if self.retrieval_mode == "hybrid" and self.retriever is not None:
                raw_results = self.retriever.retrieve(
                    query=base_query,
                    top_k=oversample_k,
                    category_filter=effective_category,
                    mode="hybrid",
                )
            else:
                raw_results = self.kb.search(
                    query=base_query,
                    top_k=oversample_k,
                    category_filter=effective_category,
                )
        except Exception as e:
            logger.warning(f"RAG sorgusu başarısız: {e}")
            return []

        # 3) İsteğe bağlı reranker
        if self.use_reranker and self.reranker is not None and raw_results:
            try:
                raw_results = self.reranker.rerank(base_query, raw_results, top_k=oversample_k)
            except Exception as e:
                logger.warning(f"Reranker hatası: {e}")

        # 4) PMID-bazlı dedup: aynı makaleden yalnız tek chunk al, en yüksek skorlu olanı
        seen_pmids = set()
        articles = []
        for res in raw_results:
            meta = res.get("metadata", {}) or {}
            doc = res.get("text", "") or ""
            pmid = str(meta.get("pmid", "")).strip()

            # Dedup anahtarı: PMID varsa PMID, yoksa title fallback
            dedup_key = pmid if pmid else (str(meta.get("title", ""))[:120] or doc[:120])
            if not dedup_key or dedup_key in seen_pmids:
                continue
            seen_pmids.add(dedup_key)

            articles.append({
                "title": meta.get("title", "Unknown Title"),
                "journal": meta.get("journal", "Unknown Journal"),
                "year": meta.get("pub_date", "")[:4] if meta.get("pub_date") else "",
                "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#",
                "snippet": doc[:400] + "..." if len(doc) > 400 else doc,
                "pmid": pmid,
                "score": res.get("score", 0.0),
            })
            if len(articles) >= self.top_k:
                break

        # 5a) Az unique makale: önce yedek kategoriden tamamla
        backup_category = self.RAG_CATEGORY_FALLBACK.get(disease_class)
        if (
            len(articles) < self.MIN_UNIQUE_ARTICLES
            and backup_category
            and backup_category != effective_category
        ):
            try:
                logger.info(
                    f"Yetersiz unique makale ({len(articles)}); "
                    f"yedek kategoriden ({backup_category}) ek arama yapılıyor"
                )
                backup_results = self.kb.search(
                    query=base_query,
                    top_k=oversample_k,
                    category_filter=backup_category,
                )
                for res in backup_results:
                    if len(articles) >= self.top_k:
                        break
                    meta = res.get("metadata", {}) or {}
                    doc = res.get("text", "") or ""
                    pmid = str(meta.get("pmid", "")).strip()
                    dedup_key = pmid if pmid else (str(meta.get("title", ""))[:120] or doc[:120])
                    if not dedup_key or dedup_key in seen_pmids:
                        continue
                    seen_pmids.add(dedup_key)
                    articles.append({
                        "title": meta.get("title", "Unknown Title"),
                        "journal": meta.get("journal", "Unknown Journal"),
                        "year": meta.get("pub_date", "")[:4] if meta.get("pub_date") else "",
                        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#",
                        "snippet": doc[:400] + "..." if len(doc) > 400 else doc,
                        "pmid": pmid,
                        "score": res.get("score", 0.0),
                    })
            except Exception as e:
                logger.debug(f"Yedek kategori sorgusu hata: {e}")

        # 5b) Hâlâ az ise, kategori filtresi olmadan tüm KB'de tara
        if len(articles) < self.top_k:
            try:
                logger.info(
                    f"Yetersiz unique makale ({len(articles)}), "
                    f"kategori filtresiz son fallback"
                )
                extra = self.kb.search(
                    query=base_query,
                    top_k=oversample_k,
                    category_filter=None,
                )
                for res in extra:
                    if len(articles) >= self.top_k:
                        break
                    meta = res.get("metadata", {}) or {}
                    doc = res.get("text", "") or ""
                    pmid = str(meta.get("pmid", "")).strip()
                    dedup_key = pmid if pmid else (str(meta.get("title", ""))[:120] or doc[:120])
                    if not dedup_key or dedup_key in seen_pmids:
                        continue
                    seen_pmids.add(dedup_key)
                    articles.append({
                        "title": meta.get("title", "Unknown Title"),
                        "journal": meta.get("journal", "Unknown Journal"),
                        "year": meta.get("pub_date", "")[:4] if meta.get("pub_date") else "",
                        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#",
                        "snippet": doc[:400] + "..." if len(doc) > 400 else doc,
                        "pmid": pmid,
                        "score": res.get("score", 0.0),
                    })
            except Exception as e:
                logger.debug(f"Fallback retrieval da boş döndü: {e}")

        logger.info(
            f"RAG: {len(articles)} unique makale | "
            f"sınıf={disease_class}, kategori={effective_category}, mod={self.retrieval_mode}"
        )
        return articles

    def analyze(
        self,
        image_path: str,
        clinical_info: str = "",
        run_faithfulness_check: bool = True,
        language: str = "tr",
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

        # OOD Kontrolü
        if vision_result.get("is_ood"):
            ood_msg = (
                "Sisteme yüklenen görüntü, eğitilen 9 cilt lezyonu sınıfından hiçbirine "
                "yeterli benzerlik göstermediği için tanı üretilememiştir. "
                "Lütfen fotoğrafın net, iyi aydınlatılmış ve bir cilt lezyonuna ait olduğundan emin olun."
                if language == "tr" else
                "The uploaded image does not match any of the 9 trained skin lesion classes "
                "(confidence below threshold). Please ensure the image is clear, well-lit, "
                "and shows a skin lesion."
            )
            return {
                "vision": vision_result,
                "articles": [],
                "diagnosis": ood_msg,
            }

        # Vision model çıktısını LLM'e uygun formata çevir (dile göre)
        _names = CLASS_DISPLAY_NAMES if language == "tr" else CLASS_DISPLAY_NAMES_EN
        _top_display = _names.get(vision_result["top_class"], vision_result["top_class"])
        top3_text = "\n".join([
            f"  - {_names.get(cls, cls)}: %{prob*100:.1f}"
            for cls, prob in vision_result["top3"]
        ])
        if language == "tr":
            vision_prediction_text = (
                f"En yüksek olasılıklı tanı: {_top_display} "
                f"(%{vision_result['confidence']*100:.1f} güven)\n"
                f"İlk 3 tahmin (TTA ortalaması):\n{top3_text}"
            )
            vision_features_text = (
                f"Model: BiomedCLIP fine-tuned + TTA (Top-1: %77.78, Top-3: %95.19, Top-5: %98.52)\n"
                f"Sınıflandırma boyutu: 9 sınıf (ISIC + PAD-UFES veri seti)"
            )
        else:
            vision_prediction_text = (
                f"Primary diagnosis: {_top_display} "
                f"({vision_result['confidence']*100:.1f}% confidence)\n"
                f"Top 3 predictions (TTA averaged):\n{top3_text}"
            )
            vision_features_text = (
                f"Model: BiomedCLIP fine-tuned + TTA (Top-1: 77.78%, Top-3: 95.19%, Top-5: 98.52%)\n"
                f"Classification scope: 9 classes (ISIC + PAD-UFES dataset)"
            )

        # Klinik bilgi fallback'i de dile uysun
        clinical_fallback = (
            "No patient information was provided."
            if language != "tr"
            else "Hasta bilgisi sağlanmadı."
        )
        diagnosis = self.generator.generate_diagnosis(
            clinical_info=clinical_info or clinical_fallback,
            vision_prediction=vision_prediction_text,
            vision_features=vision_features_text,
            rag_results=articles,
            language=language,
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
                language=language,
            )
            result["faithfulness"] = faithfulness

        logger.info("Analiz tamamlandi")
        return result
