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

from src.llm.prompt_templates import SYSTEM_PROMPT, DIAGNOSIS_PROMPT_TEMPLATE
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
        max_tokens: int = 2048
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
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Desteklenmeyen LLM sağlayıcısı: {provider}")

    def _format_rag_context(self, rag_results: List[Dict[str, Any]]) -> str:
        """RAG'dan dönen sözlük listesini okunabilir metne dönüştürür."""
        if not rag_results:
            return "Literatürden ilgili bir kaynak bulunamadı."
            
        formatted_text = ""
        for i, res in enumerate(rag_results, 1):
            source = res.get("metadata", {}).get("source", "Bilinmeyen Kaynak")
            title = res.get("metadata", {}).get("title", "Başlıksız Makale")
            text = res.get("text", "").strip()
            
            formatted_text += f"[Kaynak {i}]: {title} ({source})\n"
            formatted_text += f"Özet/Alıntı: {text}\n\n"
            
        return formatted_text

    def generate_diagnosis(
        self,
        clinical_info: str,
        vision_prediction: str,
        vision_features: str,
        rag_results: List[Dict[str, Any]]
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
        # RAG verilerini formatla
        formatted_rag_context = self._format_rag_context(rag_results)
        
        # Kullanıcı promptunu doldur
        user_prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(
            clinical_info=clinical_info or "Bilgi sağlanmadı.",
            vision_prediction=vision_prediction or "Model tahmini yapılamadı.",
            vision_features=vision_features or "Özellik çıkarımı yapılamadı.",
            rag_context=formatted_rag_context
        )
        
        # Mesajları hazırla
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        logger.info("LLM'den tanı önerisi isteniyor...")
        
        try:
            response = self.llm.invoke(messages)
            logger.info("LLM yanıtı başarıyla alındı.")
            return response.content
        except Exception as e:
            logger.error(f"LLM yanıt üretirken hata oluştu: {e}")
            return f"HATA: Tanı üretilemedi. Detay: {str(e)}"
