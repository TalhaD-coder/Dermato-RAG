"""
Dermato-RAG - Veri Modulu Testleri.

Preprocessing, dataset ve augmentation modullerinin birim testleri.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


# Proje kok dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================
# Preprocessing Testleri
# ============================================

class TestLabelMappings:
    """Etiket eslestirme tablosu testleri."""

    def test_isic_label_map_has_8_classes(self):
        """ISIC 2019 etiket haritasi 8 sinif icermeli."""
        from src.data.preprocessing import ISIC_LABEL_MAP
        assert len(ISIC_LABEL_MAP) == 8

    def test_pad_label_map_has_6_classes(self):
        """PAD-UFES-20 etiket haritasi 6 sinif icermeli."""
        from src.data.preprocessing import PAD_LABEL_MAP
        assert len(PAD_LABEL_MAP) == 6

    def test_label_to_id_has_9_classes(self):
        """Birlesik etiket haritasi 9 sinif icermeli."""
        from src.data.preprocessing import LABEL_TO_ID
        assert len(LABEL_TO_ID) == 9

    def test_id_to_label_roundtrip(self):
        """ID -> Label -> ID donusumu tutarli olmali."""
        from src.data.preprocessing import LABEL_TO_ID, ID_TO_LABEL
        for label, lid in LABEL_TO_ID.items():
            assert ID_TO_LABEL[lid] == label

    def test_all_isic_labels_mapped(self):
        """Tum ISIC etiketleri standart etiketlere eslenebilmeli."""
        from src.data.preprocessing import ISIC_LABEL_MAP, LABEL_TO_ID
        for _, std_label in ISIC_LABEL_MAP.items():
            assert std_label in LABEL_TO_ID, f"{std_label} LABEL_TO_ID'de yok"

    def test_all_pad_labels_mapped(self):
        """Tum PAD etiketleri standart etiketlere eslenebilmeli."""
        from src.data.preprocessing import PAD_LABEL_MAP, LABEL_TO_ID
        for _, std_label in PAD_LABEL_MAP.items():
            assert std_label in LABEL_TO_ID, f"{std_label} LABEL_TO_ID'de yok"

    def test_label_to_category_covers_all(self):
        """Her standart etiketin bir ust kategorisi olmali."""
        from src.data.preprocessing import LABEL_TO_ID, LABEL_TO_CATEGORY
        for label in LABEL_TO_ID:
            assert label in LABEL_TO_CATEGORY, f"{label} kategori eslesmesi yok"

    def test_categories_are_valid(self):
        """Kategoriler benign, malignant veya precancerous olmali."""
        from src.data.preprocessing import LABEL_TO_CATEGORY
        valid = {"benign", "malignant", "precancerous"}
        for label, cat in LABEL_TO_CATEGORY.items():
            assert cat in valid, f"{label} icin gecersiz kategori: {cat}"


class TestProcessSingleImage:
    """Tek goruntu isleme testleri."""

    def test_process_rgb_image(self, tmp_path):
        """RGB goruntu islenmeli."""
        from src.data.preprocessing import process_single_image

        # Test goruntusu olustur
        img = Image.new("RGB", (600, 450), color=(128, 64, 32))
        src = tmp_path / "test.jpg"
        img.save(src)

        dst = tmp_path / "out" / "test.jpg"
        result = process_single_image(src, dst, target_size=(224, 224))

        assert result is True
        assert dst.exists()
        out_img = Image.open(dst)
        assert out_img.size == (224, 224)
        assert out_img.mode == "RGB"

    def test_process_rgba_image(self, tmp_path):
        """RGBA goruntu RGB'ye donusturulmeli."""
        from src.data.preprocessing import process_single_image

        img = Image.new("RGBA", (400, 400), color=(128, 64, 32, 200))
        src = tmp_path / "test.png"
        img.save(src)

        dst = tmp_path / "out" / "test.jpg"
        result = process_single_image(src, dst, target_size=(224, 224))

        assert result is True
        out_img = Image.open(dst)
        assert out_img.mode == "RGB"
        assert out_img.size == (224, 224)

    def test_process_grayscale_image(self, tmp_path):
        """Grayscale goruntu RGB'ye donusturulmeli."""
        from src.data.preprocessing import process_single_image

        img = Image.new("L", (300, 300), color=128)
        src = tmp_path / "test.png"
        img.save(src)

        dst = tmp_path / "out" / "test.jpg"
        result = process_single_image(src, dst, target_size=(224, 224))

        assert result is True
        out_img = Image.open(dst)
        assert out_img.mode == "RGB"

    def test_process_nonexistent_image(self, tmp_path):
        """Olmayan dosya False dondurmeli."""
        from src.data.preprocessing import process_single_image

        src = tmp_path / "nonexistent.jpg"
        dst = tmp_path / "out.jpg"
        result = process_single_image(src, dst)
        assert result is False

    def test_custom_target_size(self, tmp_path):
        """Farkli hedef boyut desteklenmeli."""
        from src.data.preprocessing import process_single_image

        img = Image.new("RGB", (600, 450), color=(100, 100, 100))
        src = tmp_path / "test.jpg"
        img.save(src)

        dst = tmp_path / "out.jpg"
        result = process_single_image(src, dst, target_size=(128, 128))

        assert result is True
        out_img = Image.open(dst)
        assert out_img.size == (128, 128)


# ============================================
# Processed Data Integrity Testleri
# ============================================

class TestProcessedData:
    """Islenmis veri butunlugu testleri."""

    @pytest.fixture(autouse=True)
    def check_data_exists(self):
        """Islenmis veri varsa testleri calistir."""
        if not PROCESSED_DIR.exists() or not (PROCESSED_DIR / "unified_metadata.csv").exists():
            pytest.skip("Islenmis veri bulunamadi, once process_data.py calistirin")

    def test_unified_metadata_exists(self):
        """unified_metadata.csv mevcut olmali."""
        assert (PROCESSED_DIR / "unified_metadata.csv").exists()

    def test_split_metadata_files_exist(self):
        """Train/val/test metadata dosyalari mevcut olmali."""
        for split in ["train", "val", "test"]:
            path = PROCESSED_DIR / f"{split}_metadata.csv"
            assert path.exists(), f"{split}_metadata.csv bulunamadi"

    def test_class_weights_exist(self):
        """class_weights.csv mevcut olmali."""
        assert (PROCESSED_DIR / "class_weights.csv").exists()

    def test_dataset_stats_exist(self):
        """dataset_stats.json mevcut olmali."""
        assert (PROCESSED_DIR / "dataset_stats.json").exists()

    def test_total_image_count(self):
        """Toplam goruntu sayisi 27629 olmali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        assert len(df) == 27629

    def test_split_ratios(self):
        """Split oranlari yaklasik 70/15/15 olmali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        total = len(df)
        train_count = len(df[df["split"] == "train"])
        val_count = len(df[df["split"] == "val"])
        test_count = len(df[df["split"] == "test"])

        # Toplam tutarli olmali
        assert train_count + val_count + test_count == total

        # Oranlar yaklasik dogru olmali (+-2%)
        assert abs(train_count / total - 0.70) < 0.02
        assert abs(val_count / total - 0.15) < 0.02
        assert abs(test_count / total - 0.15) < 0.02

    def test_9_classes_present(self):
        """9 farkli sinif olmali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        assert df["label"].nunique() == 9

    def test_label_ids_sequential(self):
        """Label ID'ler 0-8 arasinda olmali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        label_ids = sorted(df["label_id"].unique())
        assert label_ids == list(range(9))

    def test_no_missing_labels(self):
        """Label sutununda eksik deger olmamali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        assert df["label"].isna().sum() == 0
        assert df["label_id"].isna().sum() == 0

    def test_sources_correct(self):
        """Kaynak isic2019 ve pad_ufes_20 olmali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        sources = set(df["source"].unique())
        assert sources == {"isic2019", "pad_ufes_20"}

    def test_required_columns_present(self):
        """Gerekli sutunlar mevcut olmali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        required = [
            "image_id", "filename", "relative_path", "source",
            "original_label", "label", "label_id", "category",
            "image_type", "split",
        ]
        for col in required:
            assert col in df.columns, f"Sutun eksik: {col}"

    def test_image_types_correct(self):
        """Goruntu tipleri dermoscopic ve smartphone olmali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        types = set(df["image_type"].unique())
        assert types == {"dermoscopic", "smartphone"}

    def test_split_metadata_consistent(self):
        """Split CSV'leri unified ile tutarli olmali."""
        unified = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        for split_name in ["train", "val", "test"]:
            split_df = pd.read_csv(PROCESSED_DIR / f"{split_name}_metadata.csv")
            unified_split = unified[unified["split"] == split_name]
            assert len(split_df) == len(unified_split)

    def test_dataset_stats_valid(self):
        """dataset_stats.json gecerli ve tutarli olmali."""
        with open(PROCESSED_DIR / "dataset_stats.json") as f:
            stats = json.load(f)

        assert stats["total_images"] == 27629
        assert stats["num_classes"] == 9
        assert len(stats["classes"]) == 9
        assert sum(stats["split_counts"].values()) == 27629

    def test_class_weights_valid(self):
        """Sinif agirliklari pozitif olmali."""
        weights_df = pd.read_csv(PROCESSED_DIR / "class_weights.csv")
        assert len(weights_df) == 9
        assert (weights_df["weight"] > 0).all()
        assert (weights_df["count"] > 0).all()

    def test_sample_images_exist(self):
        """Rastgele 10 goruntu dosyasi mevcut olmali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        sample = df.sample(n=min(10, len(df)), random_state=42)
        for _, row in sample.iterrows():
            img_path = PROCESSED_DIR / row["relative_path"]
            assert img_path.exists(), f"Goruntu bulunamadi: {img_path}"

    def test_sample_images_valid(self):
        """Rastgele 5 goruntu acilabilir ve 224x224 olmali."""
        df = pd.read_csv(PROCESSED_DIR / "unified_metadata.csv")
        sample = df.sample(n=min(5, len(df)), random_state=42)
        for _, row in sample.iterrows():
            img_path = PROCESSED_DIR / row["relative_path"]
            img = Image.open(img_path)
            assert img.size == (224, 224), f"Boyut hatasi: {img.size}"
            assert img.mode == "RGB", f"Mode hatasi: {img.mode}"


# ============================================
# Augmentation Testleri
# ============================================

class TestAugmentationConfig:
    """Augmentation konfigurasyon testleri."""

    def test_get_light_config(self):
        from src.data.augmentation import get_augmentation_config
        config = get_augmentation_config("light")
        assert config["horizontal_flip_p"] == 0.5
        assert config["vertical_flip_p"] == 0.0

    def test_get_medium_config(self):
        from src.data.augmentation import get_augmentation_config
        config = get_augmentation_config("medium")
        assert config["rotation_degrees"] == 20
        assert config["random_erasing_p"] == 0.1

    def test_get_heavy_config(self):
        from src.data.augmentation import get_augmentation_config
        config = get_augmentation_config("heavy")
        assert config["rotation_degrees"] == 30
        assert config["random_erasing_p"] == 0.2

    def test_invalid_level_raises(self):
        from src.data.augmentation import get_augmentation_config
        with pytest.raises(ValueError):
            get_augmentation_config("invalid")

    def test_config_is_copy(self):
        """Config degistirildiginde orijinal etkilenmemeli."""
        from src.data.augmentation import get_augmentation_config
        config1 = get_augmentation_config("medium")
        config1["rotation_degrees"] = 999
        config2 = get_augmentation_config("medium")
        assert config2["rotation_degrees"] == 20


class TestHairAugmentation:
    """Sac artefakti augmentation testleri."""

    def test_hair_augmentation_returns_image(self):
        from src.data.augmentation import HairAugmentation
        aug = HairAugmentation(p=1.0)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = aug(img)
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)
        assert result.mode == "RGB"

    def test_hair_augmentation_modifies_image(self):
        from src.data.augmentation import HairAugmentation
        np.random.seed(42)
        aug = HairAugmentation(p=1.0)
        img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        result = aug(img)
        # Sac eklendiyse piksel degerleri degismeli
        orig_arr = np.array(img)
        result_arr = np.array(result)
        assert not np.array_equal(orig_arr, result_arr)

    def test_hair_augmentation_skip(self):
        from src.data.augmentation import HairAugmentation
        aug = HairAugmentation(p=0.0)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = aug(img)
        assert np.array_equal(np.array(img), np.array(result))

    def test_hair_augmentation_repr(self):
        from src.data.augmentation import HairAugmentation
        aug = HairAugmentation(p=0.5)
        repr_str = repr(aug)
        assert "HairAugmentation" in repr_str
        assert "0.5" in repr_str


class TestMicroscopeAugmentation:
    """Dermoskop cerceve augmentation testleri."""

    def test_microscope_returns_image(self):
        from src.data.augmentation import MicroscopeAugmentation
        aug = MicroscopeAugmentation(p=1.0)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = aug(img)
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)

    def test_microscope_creates_dark_edges(self):
        from src.data.augmentation import MicroscopeAugmentation
        np.random.seed(42)
        aug = MicroscopeAugmentation(p=1.0)
        img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        result = aug(img)
        result_arr = np.array(result)
        # Koseler siyah olmali (vignette efekti)
        corner_val = result_arr[0, 0].sum()
        center_val = result_arr[112, 112].sum()
        assert corner_val < center_val

    def test_microscope_skip(self):
        from src.data.augmentation import MicroscopeAugmentation
        aug = MicroscopeAugmentation(p=0.0)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = aug(img)
        assert np.array_equal(np.array(img), np.array(result))


class TestImageNetConstants:
    """Normalizasyon sabitleri testleri."""

    def test_imagenet_mean_values(self):
        from src.data.augmentation import IMAGENET_MEAN
        assert len(IMAGENET_MEAN) == 3
        assert all(0 < v < 1 for v in IMAGENET_MEAN)

    def test_imagenet_std_values(self):
        from src.data.augmentation import IMAGENET_STD
        assert len(IMAGENET_STD) == 3
        assert all(0 < v < 1 for v in IMAGENET_STD)
