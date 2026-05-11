"""
Dermato-RAG Güven ve Hallüsinasyon Kontrolü Modülü.

Bu modül, LLM tarafından üretilen yanıtın, verilen RAG 
kaynaklarına (tıbbi literatüre) ne ölçüde sadık kaldığını 
(faithfulness) değerlendirir.
"""

from typing import Any, Dict, List
from langchain_core.messages import HumanMessage
from src.llm.prompt_templates import FAITHFULNESS_CHECK_PROMPT
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
        
    def _format_rag_context(self, rag_results: List[Dict[str, Any]]) -> str:
        if not rag_results:
            return "Kaynak yok."
        
        return "\n\n".join([res.get("text", "") for res in rag_results])
        
    def check_faithfulness(self, generated_response: str, rag_results: List[Dict[str, Any]]) -> str:
        """
        LLM'in verdiği yanıtı, kaynak makalelerle karşılaştırarak
        kendi kendini düzeltmesini veya uyarı vermesini sağlar.
        
        Returns:
            Değerlendirme sonucu (Hallüsinasyon uyarısı veya onay)
        """
        formatted_rag = self._format_rag_context(rag_results)
        
        prompt = FAITHFULNESS_CHECK_PROMPT.format(
            rag_context=formatted_rag,
            generated_response=generated_response
        )
        
        logger.info("Faithfulness (kaynağa sadakat) kontrolü yapılıyor...")
        
        try:
            messages = [HumanMessage(content=prompt)]
            evaluation = self.llm.invoke(messages)
            return evaluation.content
        except Exception as e:
            logger.error(f"Güvenilirlik kontrolü sırasında hata: {e}")
            return "Kontrol tamamlanamadı."
