"""
Dermato-RAG LLM Generator Modülü.

Bu modül, LangChain kullanarak OpenAI (GPT-4) veya Google Gemini
modelleri ile iletişim kurar. Vision modelinden ve RAG retriever'dan
gelen verileri prompt şablonlarıyla birleştirip son tanıyı üretir.
"""

from typing import Any, Dict, List, Optional
import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm.prompt_templates import (
    SYSTEM_PROMPT,                # geriye uyumluluk alias (TR)
    DIAGNOSIS_PROMPT_TEMPLATE,    # geriye uyumluluk alias (TR)
    get_system_prompt,
    get_diagnosis_template,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiagnosticGenerator:
    """
    Klinik verileri, vision tahminlerini ve RAG kanıtlarını
    birleştirerek ayırıcı tanı üreten LLM yöneticisi.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str = "gemini-pro",
        temperature: float = 0.1,
        max_tokens: int = 8192
    ) -> None:
        """
        LLM bağlantısını başlatır.
        
        Args:
            provider: "openai" veya "gemini"
            model_name: Kullanılacak modelin tam adı (örn: "gpt-4o", "gemini-1.5-pro")
            temperature: Üretim sıcaklığı (tıbbi işler için düşük tutulur, 0.1)
            max_tokens: Maksimum çıktı uzunluğu
        """
        self.provider = provider.lower()
        self.temperature = temperature
        
        logger.info(f"LLM Generator başlatılıyor... ({self.provider} - {model_name})")
        
        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY ortam değişkeni bulunamadı!")
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
        elif self.provider == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                logger.warning("GOOGLE_API_KEY ortam değişkeni bulunamadı!")
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,  # Gemini için doğru parametre adı
            )
        else:
            raise ValueError(f"Desteklenmeyen LLM sağlayıcısı: {provider}")

    def _format_rag_context(
        self,
        rag_results: List[Dict[str, Any]],
        language: str = "tr",
    ) -> str:
        """RAG'dan dönen sözlük listesini okunabilir metne dönüştürür."""
        is_en = str(language).lower().startswith("en")
        if not rag_results:
            return ("No relevant literature was retrieved."
                    if is_en
                    else "Literatürden ilgili bir kaynak bulunamadı.")

        # Dil bazlı etiketler
        src_lbl     = "Source"           if is_en else "Kaynak"
        no_journal  = "Unknown Journal"  if is_en else "Bilinmeyen Kaynak"
        no_title    = "Untitled article" if is_en else "Başlıksız Makale"
        excerpt_lbl = "Abstract/Excerpt" if is_en else "Özet/Alıntı"

        formatted_text = ""
        for i, res in enumerate(rag_results, 1):
            source = res.get("journal", no_journal) or no_journal
            title  = res.get("title", no_title) or no_title
            text   = (res.get("snippet", "") or "").strip()
            formatted_text += f"[{src_lbl} {i}]: {title} ({source})\n"
            formatted_text += f"{excerpt_lbl}: {text}\n\n"
        return formatted_text

    def generate_diagnosis(
        self,
        clinical_info: str,
        vision_prediction: str,
        vision_features: str,
        rag_results: List[Dict[str, Any]],
        language: str = "tr",
    ) -> str:
        """
        Girdileri birleştirip LLM'e göndererek tanıyı oluşturur.
        
        Args:
            clinical_info: Hastanın yaşı, lezyon bölgesi, semptomları vb.
            vision_prediction: Vision encoder'ın sınıflandırma sonucu
            vision_features: Modele dair ek özellikler/skorlar
            rag_results: Retriever'dan gelen makale listesi
            
        Returns:
            LLM'in ürettiği metin yanıtı
        """
        # RAG verilerini formatla (dile göre)
        formatted_rag_context = self._format_rag_context(rag_results, language=language)

        # Dile göre prompt fallback'leri
        is_en = str(language).lower().startswith("en")
        fb_clinical = "No patient information was provided." if is_en else "Bilgi sağlanmadı."
        fb_vision   = "Vision model prediction unavailable." if is_en else "Model tahmini yapılamadı."
        fb_features = "No feature extraction available."     if is_en else "Özellik çıkarımı yapılamadı."

        template = get_diagnosis_template(language)
        user_prompt = template.format(
            clinical_info=clinical_info or fb_clinical,
            vision_prediction=vision_prediction or fb_vision,
            vision_features=vision_features or fb_features,
            rag_context=formatted_rag_context,
        )

        # Sistem promptunu da dile göre seç
        system_prompt_text = get_system_prompt(language)
        messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=user_prompt),
        ]
        
        logger.info("LLM'den tanı önerisi isteniyor...")
        
        try:
            response = self.llm.invoke(messages)
            logger.info("LLM yanıtı başarıyla alındı.")
            return response.content
        except Exception as e:
            err = str(e)
            logger.error(f"LLM yanıt üretirken hata: {err[:200]}")
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                return "LLM_ERROR:QUOTA"
            elif "API_KEY" in err or "INVALID" in err or "401" in err:
                return "LLM_ERROR:AUTH"
            elif "timeout" in err.lower() or "deadline" in err.lower():
                return "LLM_ERROR:TIMEOUT"
            else:
                return "LLM_ERROR:UNKNOWN"
