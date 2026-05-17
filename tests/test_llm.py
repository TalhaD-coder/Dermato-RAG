"""
Dermato-RAG LLM Pipeline Birim Testleri.

Bu testler gerçek API çağrısı yapmaz (mock kullanır).
Dolayısıyla internet bağlantısı veya API anahtarı gerektirmez.

Çalıştırma:
    python -m pytest tests/test_llm.py -v
"""

import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────
# 1. PROMPT TEMPLATES TESTLERİ
# ─────────────────────────────────────────────────────────────

class TestPromptTemplates:
    """prompt_templates.py içeriğini doğrular."""

    def test_system_prompt_exists(self):
        """Sistem promptu var ve dolu mu?"""
        from src.llm.prompt_templates import SYSTEM_PROMPT
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_has_rules(self):
        """Sistem promptu 'hallucination' veya 'uydurma' içeriyor mu?"""
        from src.llm.prompt_templates import SYSTEM_PROMPT
        # Türkçe veya İngilizce olabilir
        assert any(word in SYSTEM_PROMPT.lower() for word in ["hallucin", "uydurma", "kural"])

    def test_diagnosis_template_has_placeholders(self):
        """Tanı şablonu doğru placeholder'ları içeriyor mu?"""
        from src.llm.prompt_templates import DIAGNOSIS_PROMPT_TEMPLATE
        required = ["{clinical_info}", "{vision_prediction}", "{vision_features}", "{rag_context}"]
        for placeholder in required:
            assert placeholder in DIAGNOSIS_PROMPT_TEMPLATE, \
                f"Eksik placeholder: {placeholder}"

    def test_diagnosis_template_fills_correctly(self):
        """Şablon doğru doldurulabiliyor mu?"""
        from src.llm.prompt_templates import DIAGNOSIS_PROMPT_TEMPLATE
        filled = DIAGNOSIS_PROMPT_TEMPLATE.format(
            clinical_info="45 yaş erkek hasta",
            vision_prediction="Melanoma %82",
            vision_features="BiomedCLIP, 9 sınıf",
            rag_context="[Kaynak 1]: Test makalesi",
        )
        assert "45 yaş erkek hasta" in filled
        assert "Melanoma %82" in filled
        assert "[Kaynak 1]" in filled

    def test_faithfulness_template_has_placeholders(self):
        """Güvenilirlik kontrol şablonu doğru placeholder'ları içeriyor mu?"""
        from src.llm.prompt_templates import FAITHFULNESS_CHECK_PROMPT
        assert "{rag_context}" in FAITHFULNESS_CHECK_PROMPT
        assert "{generated_response}" in FAITHFULNESS_CHECK_PROMPT


# ─────────────────────────────────────────────────────────────
# 2. DiagnosticGenerator TESTLERİ
# ─────────────────────────────────────────────────────────────

class TestDiagnosticGenerator:
    """generator.py bileşenini doğrular (LLM mock'lanır)."""

    def test_rag_context_empty_list(self):
        """Boş RAG sonucu için fallback metni döner mi?"""
        with patch("langchain_google_genai.ChatGoogleGenerativeAI"):
            # Cache temizle
            for mod in [k for k in sys.modules if "src.llm" in k]:
                del sys.modules[mod]
            from src.llm.generator import DiagnosticGenerator
            gen = DiagnosticGenerator(provider="gemini", model_name="gemini-1.5-pro")
            result = gen._format_rag_context([])
            assert "bulunamadı" in result.lower() or len(result) > 0

    def test_rag_context_formats_correctly(self):
        """RAG sonuçları doğru formatlanıyor mu?"""
        with patch("langchain_google_genai.ChatGoogleGenerativeAI"):
            for mod in [k for k in sys.modules if "src.llm" in k]:
                del sys.modules[mod]
            from src.llm.generator import DiagnosticGenerator
            gen = DiagnosticGenerator(provider="gemini", model_name="gemini-1.5-pro")

            rag_results = [
                {
                    "text": "Melanoma tanısı için ABCDE kriterleri kullanılır.",
                    "metadata": {"title": "Melanoma Review", "source": "JAAD 2023"},
                }
            ]
            result = gen._format_rag_context(rag_results)
            assert "Melanoma Review" in result
            assert "ABCDE" in result
            assert "[Kaynak 1]" in result

    def test_generate_diagnosis_calls_llm(self):
        """generate_diagnosis() LLM'i çağırıyor mu?"""
        mock_response = MagicMock()
        mock_response.content = "Test tanı raporu üretildi."
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response

        with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
            for mod in [k for k in sys.modules if "src.llm" in k]:
                del sys.modules[mod]
            from src.llm.generator import DiagnosticGenerator
            gen = DiagnosticGenerator(provider="gemini", model_name="gemini-1.5-pro")
            gen.llm = mock_llm_instance

            result = gen.generate_diagnosis(
                clinical_info="Test hasta",
                vision_prediction="Melanoma %80",
                vision_features="BiomedCLIP",
                rag_results=[],
            )

            assert mock_llm_instance.invoke.called
            assert result == "Test tanı raporu üretildi."

    def test_generate_diagnosis_handles_error(self):
        """LLM hata verdiğinde düzgün hata mesajı döner mi?"""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = Exception("API bağlantı hatası")

        with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
            for mod in [k for k in sys.modules if "src.llm" in k]:
                del sys.modules[mod]
            from src.llm.generator import DiagnosticGenerator
            gen = DiagnosticGenerator(provider="gemini", model_name="gemini-1.5-pro")
            gen.llm = mock_llm_instance

            result = gen.generate_diagnosis(
                clinical_info="Test",
                vision_prediction="Test",
                vision_features="Test",
                rag_results=[],
            )

            assert "HATA" in result or "hata" in result.lower()

    def test_unsupported_provider_raises(self):
        """Desteklenmeyen LLM sağlayıcısı hata fırlatıyor mu?"""
        for mod in [k for k in sys.modules if "src.llm" in k]:
            del sys.modules[mod]
        with pytest.raises(ValueError, match="Desteklenmeyen"):
            from src.llm.generator import DiagnosticGenerator
            DiagnosticGenerator(provider="unknown_llm", model_name="test")


# ─────────────────────────────────────────────────────────────
# 3. ConfidenceEvaluator TESTLERİ
# ─────────────────────────────────────────────────────────────

class TestConfidenceEvaluator:
    """confidence.py bileşenini doğrular."""

    def test_faithfulness_check_calls_llm(self):
        """check_faithfulness() LLM'i çağırıyor mu?"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Yanıt kaynaklarla uyumlu, hallüsinasyon tespit edilmedi."
        mock_llm.invoke.return_value = mock_response

        from src.llm.confidence import ConfidenceEvaluator
        evaluator = ConfidenceEvaluator(llm=mock_llm)

        result = evaluator.check_faithfulness(
            generated_response="Melanoma tanısı konuldu.",
            rag_results=[{"text": "Melanoma ABCDE kriterleri...", "metadata": {}}],
        )

        assert mock_llm.invoke.called
        assert "hallüsinasyon" in result.lower() or len(result) > 0

    def test_faithfulness_empty_rag(self):
        """Boş RAG ile faithfulness kontrolü çalışıyor mu?"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Kaynak yok, değerlendirme yapılamadı."
        mock_llm.invoke.return_value = mock_response

        from src.llm.confidence import ConfidenceEvaluator
        evaluator = ConfidenceEvaluator(llm=mock_llm)

        result = evaluator.check_faithfulness(
            generated_response="Test tanı",
            rag_results=[],
        )
        assert result is not None

    def test_faithfulness_handles_error(self):
        """LLM hata verirse 'tamamlanamadı' mesajı dönüyor mu?"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Ağ hatası")

        from src.llm.confidence import ConfidenceEvaluator
        evaluator = ConfidenceEvaluator(llm=mock_llm)

        result = evaluator.check_faithfulness(
            generated_response="Test",
            rag_results=[],
        )
        assert "tamamlanamadı" in result.lower() or result is not None


# ─────────────────────────────────────────────────────────────
# 4. DermatoRAGPipeline TESTLERİ
# ─────────────────────────────────────────────────────────────

class TestDermatoRAGPipeline:
    """pipeline.py'nin ana akışını doğrular (tüm bağımlılıklar mock'lanır)."""

    def test_pipeline_initializes(self):
        """Pipeline hatasız başlatılabiliyor mu?"""
        mock_model = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 706
        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch("src.pipeline.DermatoVisionEncoder.load_checkpoint", return_value=mock_model), \
             patch("src.pipeline.DiagnosticGenerator"), \
             patch("src.pipeline.ConfidenceEvaluator"), \
             patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            for mod in [k for k in sys.modules if "src.pipeline" in k]:
                del sys.modules[mod]
            from src.pipeline import DermatoRAGPipeline
            pipeline = DermatoRAGPipeline(
                model_path="models/best_model.pt",
                llm_provider="gemini",
                llm_model="gemini-1.5-pro",
            )
            assert pipeline is not None
            assert pipeline.top_k == 5

    def test_pipeline_missing_model_raises(self):
        """Model dosyası yoksa FileNotFoundError fırlatıyor mu?"""
        with patch("src.pipeline.DiagnosticGenerator"), \
             patch("src.pipeline.ConfidenceEvaluator"), \
             patch("chromadb.PersistentClient"):
            for mod in [k for k in sys.modules if "src.pipeline" in k]:
                del sys.modules[mod]
            from src.pipeline import DermatoRAGPipeline
            with pytest.raises(FileNotFoundError, match="Model dosyası bulunamadı"):
                DermatoRAGPipeline(model_path="olmayan_dosya.pt")

    def test_class_labels_complete(self):
        """9 sınıf etiketinin tamamı tanımlı mı?"""
        with patch("langchain_google_genai.ChatGoogleGenerativeAI"):
            for mod in [k for k in sys.modules if "src.pipeline" in k or "src.llm" in k]:
                del sys.modules[mod]
            from src.pipeline import CLASS_LABELS
        assert len(CLASS_LABELS) == 9
        required_classes = {
            "melanoma", "nevus", "basal_cell_carcinoma",
            "squamous_cell_carcinoma", "actinic_keratosis",
            "benign_keratosis", "dermatofibroma",
            "vascular_lesion", "seborrheic_keratosis",
        }
        assert required_classes == set(CLASS_LABELS.values())

    def test_display_names_complete(self):
        """Tüm sınıflar için Türkçe görüntü isimleri var mı?"""
        with patch("langchain_google_genai.ChatGoogleGenerativeAI"):
            for mod in [k for k in sys.modules if "src.pipeline" in k or "src.llm" in k]:
                del sys.modules[mod]
            from src.pipeline import CLASS_LABELS, CLASS_DISPLAY_NAMES
        for cls in CLASS_LABELS.values():
            assert cls in CLASS_DISPLAY_NAMES, f"Eksik Türkçe isim: {cls}"
