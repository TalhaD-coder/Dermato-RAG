"""
Dermato-RAG Vision Model Testleri.

DermatoVisionEncoder ve VisionTrainer için birim testleri.
Modelin doğru şekilde yüklendiğini, forward pass'in çalıştığını
ve checkpoint mekanizmasının düzgün olduğunu doğrular.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.models.vision_encoder import (
    CLASS_NAMES,
    DEFAULT_NUM_CLASSES,
    EMBEDDING_DIM,
    DermatoVisionEncoder,
)


class TestDermatoVisionEncoder:
    """DermatoVisionEncoder birim testleri."""

    def test_class_names_count(self):
        """Sınıf ismi sayısı doğru mu?"""
        assert len(CLASS_NAMES) == DEFAULT_NUM_CLASSES

    def test_model_creation_classify(self):
        """Sınıflandırma modunda model oluşturulabiliyor mu?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="classify", pretrained=False
        )
        assert model is not None
        assert model.mode == "classify"
        assert model.num_classes == 9
        assert model.classifier is not None

    def test_model_creation_extract(self):
        """Feature extraction modunda model oluşturulabiliyor mu?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="extract", pretrained=False
        )
        assert model is not None
        assert model.mode == "extract"
        assert model.classifier is None

    def test_invalid_mode(self):
        """Geçersiz mod hata veriyor mu?"""
        with pytest.raises(AssertionError):
            DermatoVisionEncoder(mode="invalid", pretrained=False)

    def test_forward_classify(self):
        """Classify modu forward pass çıktısı doğru boyutta mı?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="classify", pretrained=False
        )
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 9), f"Beklenen (2,9), gelen {output.shape}"

    def test_forward_extract(self):
        """Extract modu forward pass çıktısı doğru boyutta mı?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="extract", pretrained=False
        )
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, EMBEDDING_DIM), (
            f"Beklenen (2,{EMBEDDING_DIM}), gelen {output.shape}"
        )

    def test_extract_features(self):
        """extract_features metodu çalışıyor mu?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="classify", pretrained=False
        )
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model.extract_features(dummy_input)

        assert features.shape == (1, EMBEDDING_DIM)

    def test_freeze_backbone(self):
        """Backbone dondurma çalışıyor mu?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="classify",
            pretrained=False, freeze_backbone=True
        )

        # Backbone parametreleri dondurulmuş olmalı
        for param in model.backbone.parameters():
            assert not param.requires_grad

        # Classifier parametreleri eğitilebilir olmalı
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self):
        """Backbone açma çalışıyor mu?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="classify",
            pretrained=False, freeze_backbone=True
        )
        model.unfreeze_backbone()

        # Artık tüm parametreler eğitilebilir olmalı
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_get_trainable_params(self):
        """Parametre sayısı hesaplaması çalışıyor mu?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="classify", pretrained=False
        )
        trainable, total = model.get_trainable_params()

        assert trainable > 0
        assert total > 0
        assert trainable <= total

    def test_save_load_checkpoint(self):
        """Checkpoint kaydetme/yükleme çalışıyor mu?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="classify", pretrained=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = str(Path(tmpdir) / "test_ckpt.pt")

            # Kaydet
            model.save_checkpoint(ckpt_path, epoch=5, metrics={"val_acc": 0.85})

            # Yükle
            loaded = DermatoVisionEncoder.load_checkpoint(ckpt_path)

            assert loaded.num_classes == 9
            assert loaded.mode == "classify"

    def test_summary(self):
        """Model özeti doğru döndürülüyor mu?"""
        model = DermatoVisionEncoder(
            num_classes=9, mode="classify", pretrained=False
        )
        summary = model.summary()

        assert "total_params" in summary
        assert "trainable_params" in summary
        assert summary["num_classes"] == 9
        assert summary["mode"] == "classify"

    def test_frozen_model_has_fewer_trainable_params(self):
        """Dondurulmuş model daha az eğitilebilir parametreye sahip mi?"""
        model_full = DermatoVisionEncoder(
            num_classes=9, mode="classify",
            pretrained=False, freeze_backbone=False
        )
        model_frozen = DermatoVisionEncoder(
            num_classes=9, mode="classify",
            pretrained=False, freeze_backbone=True
        )

        trainable_full, _ = model_full.get_trainable_params()
        trainable_frozen, _ = model_frozen.get_trainable_params()

        assert trainable_frozen < trainable_full
