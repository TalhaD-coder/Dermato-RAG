"""
Dermato-RAG Güven ve Hallüsinasyon Kontrolü Modülü.

Bu modül, LLM tarafından üretilen yanıtın, verilen RAG
kaynaklarına (tıbbi literatüre) ne ölçüde sadık kaldığını
(faithfulness) değerlendirir.

`language` parametresi ile TR/EN promptları arasında seçim yapar.
"""

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from src.llm.prompt_templates import (
    FAITHFULNESS_CHECK_PROMPT,  # geriye uyumluluk alias (TR)
    get_faithfulness_prompt,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceEvaluator:
    """
    Üretilen tanının literatürle uyumunu (Faithfulness) ve
    hallüsinasyon içerip içermediğini kontrol eden sınıf.
    """

    def __init__(self, llm) -> None:
        """
        Args:
            llm: LangChain ChatModel instance (DiagnosticGenerator içindeki llm)
        """
        self.llm = llm

    def _format_rag_context(
        self,
        rag_results: List[Dict[str, Any]],
        language: str = "tr",
    ) -> str:
        """
        Pipeline article'ları (snippet/title/journal) ve raw KB sonuçları
        (text/metadata) — her iki şekli de destekler.
        """
        is_en = str(language).lower().startswith("en")
        if not rag_results:
            return "No sources." if is_en else "Kaynak yok."

        src_lbl = "Source" if is_en else "Kaynak"
        parts = []
        for i, res in enumerate(rag_results, 1):
            # 1) raw KB sonucu
            text = res.get("text", "")
            if not text:
                # 2) pipeline article'ı (snippet alanı)
                text = res.get("snippet", "")
            title = ""
            meta = res.get("metadata") or {}
            if isinstance(meta, dict):
                title = meta.get("title", "")
            if not title:
                title = res.get("title", "")
            if title:
                parts.append(f"[{src_lbl} {i}] {title}\n{text}")
            else:
                parts.append(f"[{src_lbl} {i}]\n{text}")
        return "\n\n".join(parts)

    def check_faithfulness(
        self,
        generated_response: str,
        rag_results: List[Dict[str, Any]],
        language: str = "tr",
    ) -> str:
        """
        LLM'in verdiği yanıtı, kaynak makalelerle karşılaştırarak
        kendi kendini düzeltmesini veya uyarı vermesini sağlar.

        Returns:
            Değerlendirme sonucu (hallüsinasyon uyarısı veya onay).
        """
        is_en = str(language).lower().startswith("en")
        formatted_rag = self._format_rag_context(rag_results, language=language)

        template = get_faithfulness_prompt(language)
        prompt = template.format(
            rag_context=formatted_rag,
            generated_response=generated_response,
        )

        logger.info("Faithfulness (kaynağa sadakat) kontrolü yapılıyor...")

        try:
            messages = [HumanMessage(content=prompt)]
            evaluation = self.llm.invoke(messages)
            return evaluation.content
        except Exception as e:
            logger.error(f"Güvenilirlik kontrolü sırasında hata: {e}")
            return ("Check could not be completed."
                    if is_en
                    else "Kontrol tamamlanamadı.")
