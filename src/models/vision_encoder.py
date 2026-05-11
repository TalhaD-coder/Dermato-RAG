"""
Dermato-RAG Vision Encoder Modülü.

BiomedCLIP tabanlı görüntü encoder. Dermatolojik lezyonların
sınıflandırılması ve özellik çıkarımı (feature extraction) için
fine-tune edilebilir mimari.

Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- Tıbbi görüntüler için önceden eğitilmiş CLIP modeli
- ViT-B/16 mimarisi (224x224 girdi)
- 768 boyutlu embedding çıktısı

Kullanım:
    from src.models.vision_encoder import DermatoVisionEncoder

    # Sınıflandırma modu
    model = DermatoVisionEncoder(num_classes=9, mode="classify")
    logits = model(images)

    # Feature extraction modu
    model = DermatoVisionEncoder(mode="extract")
    features = model(images)  # (batch, 768)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Model sabitleri
BIOMEDCLIP_MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBEDDING_DIM = 512
DEFAULT_NUM_CLASSES = 9  # ISIC + PAD-UFES birleşik sınıf sayısı

# Sınıf isimleri (preprocessing.py ile uyumlu)
CLASS_NAMES = [
    "akiec",   # Actinic Keratosis / Bowen's
    "bcc",     # Basal Cell Carcinoma
    "bkl",     # Benign Keratosis
    "df",      # Dermatofibroma
    "mel",     # Melanoma
    "nv",      # Melanocytic Nevus
    "vasc",    # Vascular Lesion
    "scc",     # Squamous Cell Carcinoma (PAD-UFES)
    "ack",     # Actinic Keratosis (PAD-UFES)
]


class DermatoVisionEncoder(nn.Module):
    """
    BiomedCLIP tabanlı dermatoloji vision encoder.

    İki çalışma modu:
    - 'classify': Sınıflandırma head'i ile lezyon tipi tahmini.
    - 'extract': 768 boyutlu özellik vektörü çıkarır (RAG pipeline için).

    Args:
        num_classes: Sınıf sayısı (varsayılan: 9).
        mode: 'classify' veya 'extract'.
        pretrained: Önceden eğitilmiş ağırlıkları yükle.
        freeze_backbone: Backbone'u dondur (sadece head eğitilsin).
        dropout_rate: Sınıflandırma head'indeki dropout oranı.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        mode: str = "classify",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        assert mode in ("classify", "extract"), f"Geçersiz mod: {mode}"

        self.num_classes = num_classes
        self.mode = mode
        self.freeze_backbone = freeze_backbone
        self.embedding_dim = EMBEDDING_DIM

        # Backbone yükle
        self.backbone = self._load_backbone(pretrained)

        # Backbone dondurma (opsiyonel)
        if freeze_backbone:
            self._freeze_backbone()

        # Sınıflandırma head'i (classify modu için)
        if mode == "classify":
            self.classifier = nn.Sequential(
                nn.LayerNorm(EMBEDDING_DIM),
                nn.Dropout(dropout_rate),
                nn.Linear(EMBEDDING_DIM, 256),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes),
            )
        else:
            self.classifier = None

        logger.info(
            f"DermatoVisionEncoder oluşturuldu: "
            f"mode={mode}, classes={num_classes}, "
            f"freeze={freeze_backbone}, dropout={dropout_rate}"
        )

    def _load_backbone(self, pretrained: bool) -> nn.Module:
        """
        BiomedCLIP vision backbone'unu yükler.

        open_clip kütüphanesi ile BiomedCLIP'in ViT kısmını yükler.
        Eğer open_clip yoksa, timm üzerinden ViT-B/16 yükler.
        """
        try:
            # Yöntem 1: open_clip ile BiomedCLIP
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            # Sadece visual encoder'ı al
            backbone = model.visual
            logger.info("BiomedCLIP (open_clip) başarıyla yüklendi")
            self._preprocess = preprocess
            return backbone

        except (ImportError, Exception) as e:
            logger.warning(f"open_clip yüklenemedi: {e}. timm ile ViT yükleniyor...")

            # Yöntem 2: timm ile ViT-B/16
            import timm

            backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=0,  # Head'siz (feature extractor)
            )
            logger.info("ViT-B/16 (timm, ImageNet) yüklendi")
            self._preprocess = None
            return backbone

    def _freeze_backbone(self) -> None:
        """Backbone parametrelerini dondurur."""
        frozen_count = 0
        for param in self.backbone.parameters():
            param.requires_grad = False
            frozen_count += 1
        logger.info(f"Backbone donduruldu: {frozen_count} parametre")

    def unfreeze_backbone(self, unfreeze_last_n: int = 0) -> None:
        """
        Backbone'un tamamını veya son N katmanını açar.

        Args:
            unfreeze_last_n: Son kaç katmanı açacak (0 = tümü).
        """
        if unfreeze_last_n == 0:
            # Tümünü aç
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Backbone tamamen açıldı (unfreeze)")
        else:
            # Son N katmanı aç (gradual unfreezing)
            params = list(self.backbone.parameters())
            for param in params[-unfreeze_last_n:]:
                param.requires_grad = True
            logger.info(f"Son {unfreeze_last_n} parametre açıldı")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Görüntülerden özellik vektörü çıkarır.

        Args:
            x: Görüntü tensörü (batch, 3, 224, 224).

        Returns:
            Özellik vektörü (batch, 768).
        """
        features = self.backbone(x)

        # Bazı modeller farklı formatta çıktı verir
        if isinstance(features, tuple):
            features = features[0]

        # 2D'den fazla boyut varsa (örn. ViT cls token)
        if features.dim() > 2:
            features = features[:, 0]  # CLS token

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Görüntü tensörü (batch, 3, 224, 224).

        Returns:
            - classify modu: logits (batch, num_classes)
            - extract modu: features (batch, 768)
        """
        features = self.extract_features(x)

        if self.mode == "classify" and self.classifier is not None:
            return self.classifier(features)

        return features

    def get_trainable_params(self) -> Tuple[int, int]:
        """Eğitilebilir ve toplam parametre sayılarını döndürür."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def save_checkpoint(self, path: str, epoch: int = 0, metrics: Optional[Dict] = None) -> None:
        """
        Model checkpoint'ını kaydeder.

        Args:
            path: Kayıt yolu.
            epoch: Epoch numarası.
            metrics: Metrik değerleri.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "num_classes": self.num_classes,
            "mode": self.mode,
            "embedding_dim": self.embedding_dim,
            "metrics": metrics or {},
        }
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint kaydedildi: {save_path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "DermatoVisionEncoder":
        """
        Checkpoint'tan model yükler.

        Args:
            path: Checkpoint dosya yolu.
            device: Yükleme cihazı.

        Returns:
            Yüklenmiş DermatoVisionEncoder instance.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = cls(
            num_classes=checkpoint["num_classes"],
            mode=checkpoint["mode"],
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        logger.info(
            f"Checkpoint yüklendi: epoch={checkpoint['epoch']}, "
            f"metrics={checkpoint.get('metrics', {})}"
        )
        return model

    def summary(self) -> Dict:
        """Model özetini döndürür."""
        trainable, total = self.get_trainable_params()
        return {
            "model_name": BIOMEDCLIP_MODEL_NAME,
            "mode": self.mode,
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "total_params": f"{total:,}",
            "trainable_params": f"{trainable:,}",
            "frozen_params": f"{total - trainable:,}",
            "freeze_backbone": self.freeze_backbone,
        }
