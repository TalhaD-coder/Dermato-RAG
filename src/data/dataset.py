"""
Dermato-RAG Veri Seti Modülü.

PyTorch Dataset sınıfları ve veri yükleyicileri.
İşlenmiş birleşik veri setini yükler ve eğitim/değerlendirme
için hazır hale getirir.

Kullanım:
    from src.data.dataset import DermatoDataset, get_dataloaders

    # Dataset oluştur
    train_dataset = DermatoDataset(split="train")

    # DataLoader'lar al
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DermatoDataset(Dataset):
    """
    Dermato-RAG birleşik veri seti.

    İşlenmiş ISIC 2019 + PAD-UFES-20 verilerini yükler.
    Train/Val/Test split'lerine göre filtreleme yapar.

    Args:
        split: "train", "val" veya "test".
        processed_dir: İşlenmiş veri dizini.
        transform: torchvision transform pipeline.
        return_metadata: True ise metadata dict de döndürür.
    """

    def __init__(
        self,
        split: str = "train",
        processed_dir: Optional[str] = None,
        transform: Optional[Any] = None,
        return_metadata: bool = False,
    ) -> None:
        assert split in ("train", "val", "test"), f"Geçersiz split: {split}"

        self.split = split
        self.transform = transform
        self.return_metadata = return_metadata

        # Dizin belirleme
        if processed_dir is None:
            config = Config()
            self.processed_dir = PROJECT_ROOT / config.get(
                "paths.processed_data_dir", "data/processed"
            )
        else:
            self.processed_dir = Path(processed_dir)

        # Metadata yükle
        meta_path = self.processed_dir / f"{split}_metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata dosyası bulunamadı: {meta_path}\n"
                f"Önce 'python scripts/process_data.py' komutunu çalıştırın."
            )

        self.metadata = pd.read_csv(meta_path)
        self.num_classes = self.metadata["label_id"].nunique()

        logger.info(
            f"DermatoDataset ({split}): {len(self.metadata)} görüntü, "
            f"{self.num_classes} sınıf yüklendi"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Any, int] | Tuple[Any, int, Dict]:
        """
        Tek bir örneği döndürür.

        Args:
            idx: Örnek indeksi.

        Returns:
            (image, label) veya (image, label, metadata) tuple.
        """
        row = self.metadata.iloc[idx]

        # Görüntü yükle
        img_path = self.processed_dir / row["relative_path"]
        image = Image.open(img_path).convert("RGB")

        # Label
        label = int(row["label_id"])

        # Transform uygula
        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            meta = {
                "image_id": row["image_id"],
                "source": row["source"],
                "label_name": row["label"],
                "category": row["category"],
                "image_type": row["image_type"],
            }
            return image, label, meta

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Sınıf ağırlıklarını hesaplar (imbalanced data için).

        Returns:
            Her sınıf için ağırlık tensor'ü.
        """
        class_counts = self.metadata["label_id"].value_counts().sort_index()
        total = len(self.metadata)
        n_classes = len(class_counts)
        weights = []
        for i in range(n_classes):
            count = class_counts.get(i, 1)
            weights.append(total / (n_classes * count))
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Her örnek için ağırlık hesaplar (WeightedRandomSampler için).

        Returns:
            Her örnek için ağırlık tensor'ü.
        """
        class_weights = self.get_class_weights()
        sample_weights = [
            class_weights[label_id].item()
            for label_id in self.metadata["label_id"].values
        ]
        return torch.tensor(sample_weights, dtype=torch.float64)

    @property
    def class_names(self) -> List[str]:
        """Sınıf adlarını döndürür (label_id sırasıyla)."""
        from src.data.preprocessing import ID_TO_LABEL
        return [ID_TO_LABEL[i] for i in range(self.num_classes)]

    def get_stats(self) -> Dict[str, Any]:
        """Veri seti istatistiklerini döndürür."""
        return {
            "split": self.split,
            "total_samples": len(self.metadata),
            "num_classes": self.num_classes,
            "class_distribution": self.metadata["label"].value_counts().to_dict(),
            "source_distribution": self.metadata["source"].value_counts().to_dict(),
            "image_type_distribution": self.metadata["image_type"].value_counts().to_dict(),
        }


def get_transforms(
    split: str = "train",
    image_size: int = 224,
) -> Any:
    """
    Split'e göre uygun transform pipeline döndürür.

    Train: Augmentation + normalizasyon
    Val/Test: Sadece normalizasyon

    Args:
        split: "train", "val" veya "test".
        image_size: Görüntü boyutu.

    Returns:
        torchvision.transforms.Compose instance.
    """
    try:
        from torchvision import transforms
    except ImportError:
        logger.warning("torchvision yüklü değil, transform döndürülemiyor")
        return None

    # ImageNet normalizasyon değerleri
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    processed_dir: Optional[str] = None,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train, val, test DataLoader'larını oluşturur.

    Args:
        batch_size: Batch boyutu.
        image_size: Görüntü boyutu.
        num_workers: Veri yükleme worker sayısı.
        processed_dir: İşlenmiş veri dizini.
        use_weighted_sampler: Sınıf dengesizliği için ağırlıklı örnekleme.
        pin_memory: GPU transferi optimizasyonu.

    Returns:
        (train_loader, val_loader, test_loader) tuple.
    """
    # Dataset'ler
    train_dataset = DermatoDataset(
        split="train",
        processed_dir=processed_dir,
        transform=get_transforms("train", image_size),
    )
    val_dataset = DermatoDataset(
        split="val",
        processed_dir=processed_dir,
        transform=get_transforms("val", image_size),
    )
    test_dataset = DermatoDataset(
        split="test",
        processed_dir=processed_dir,
        transform=get_transforms("test", image_size),
    )

    # Train sampler (sınıf dengesizliği için)
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False  # Sampler ile shuffle kullanılmaz

    # DataLoader'lar
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"DataLoader'lar oluşturuldu (batch_size={batch_size}):")
    logger.info(f"  Train: {len(train_dataset)} örnek, {len(train_loader)} batch")
    logger.info(f"  Val:   {len(val_dataset)} örnek, {len(val_loader)} batch")
    logger.info(f"  Test:  {len(test_dataset)} örnek, {len(test_loader)} batch")

    return train_loader, val_loader, test_loader
