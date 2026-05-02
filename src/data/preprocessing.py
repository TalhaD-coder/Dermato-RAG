"""
Dermato-RAG Görüntü Ön İşleme Modülü.

Dermatolojik görüntüler için ön işleme pipeline'ı:
- Yeniden boyutlandırma (224×224)
- RGBA → RGB dönüşümü
- Normalizasyon
- Birleşik metadata oluşturma

Desteklenen veri setleri:
- ISIC 2019
- PAD-UFES-20
- Fitzpatrick17k
"""

import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================
# Birleşik Etiket Eşleştirme Tablosu
# ============================================
# ISIC 2019 ve PAD-UFES-20 arasındaki etiket uyumu.
# Ortak bir standart oluşturuyoruz.
# ============================================

# ISIC 2019 sınıfları → standart etiket
ISIC_LABEL_MAP: Dict[str, str] = {
    "MEL": "melanoma",
    "NV": "nevus",
    "BCC": "basal_cell_carcinoma",
    "AK": "actinic_keratosis",
    "BKL": "benign_keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular_lesion",
    "SCC": "squamous_cell_carcinoma",
}

# PAD-UFES-20 sınıfları → standart etiket
PAD_LABEL_MAP: Dict[str, str] = {
    "MEL": "melanoma",
    "NEV": "nevus",
    "BCC": "basal_cell_carcinoma",
    "ACK": "actinic_keratosis",
    "SEK": "seborrheic_keratosis",
    "SCC": "squamous_cell_carcinoma",
}

# Tüm standart etiketler → sayısal ID
LABEL_TO_ID: Dict[str, int] = {
    "actinic_keratosis": 0,
    "basal_cell_carcinoma": 1,
    "benign_keratosis": 2,
    "dermatofibroma": 3,
    "melanoma": 4,
    "nevus": 5,
    "seborrheic_keratosis": 6,
    "squamous_cell_carcinoma": 7,
    "vascular_lesion": 8,
}

ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}

# Üst kategori eşleştirmesi (makale için)
LABEL_TO_CATEGORY: Dict[str, str] = {
    "actinic_keratosis": "precancerous",
    "basal_cell_carcinoma": "malignant",
    "benign_keratosis": "benign",
    "dermatofibroma": "benign",
    "melanoma": "malignant",
    "nevus": "benign",
    "seborrheic_keratosis": "benign",
    "squamous_cell_carcinoma": "malignant",
    "vascular_lesion": "benign",
}


def process_single_image(
    src_path: Path,
    dst_path: Path,
    target_size: Tuple[int, int] = (224, 224),
    quality: int = 95,
) -> bool:
    """
    Tek bir görüntüyü işler ve kaydeder.

    İşlemler:
    1. RGBA → RGB dönüşümü (gerekirse)
    2. Yeniden boyutlandırma (LANCZOS ile yüksek kalite)
    3. JPEG olarak kaydetme

    Args:
        src_path: Kaynak görüntü yolu.
        dst_path: Hedef kayıt yolu.
        target_size: Hedef boyut (width, height).
        quality: JPEG kalitesi (1-100).

    Returns:
        True ise başarılı, False ise hata.
    """
    try:
        img = Image.open(src_path)

        # RGBA → RGB dönüşümü (PAD-UFES-20 PNG dosyaları için)
        if img.mode == "RGBA":
            # Beyaz arka plan üzerine yapıştır
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Alpha channel as mask
            img = background
        elif img.mode == "L":
            # Grayscale → RGB
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Yeniden boyutlandırma (LANCZOS = en yüksek kalite)
        img = img.resize(target_size, Image.LANCZOS)

        # Hedef dizini oluştur
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # JPEG olarak kaydet
        img.save(dst_path, format="JPEG", quality=quality)
        return True

    except Exception as e:
        logger.warning(f"Görüntü işlenemedi: {src_path} → {e}")
        return False


def process_isic2019(
    raw_dir: Path,
    processed_dir: Path,
    target_size: Tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """
    ISIC 2019 veri setini işler.

    Args:
        raw_dir: Ham veri dizini (data/raw/ISIC 2019).
        processed_dir: İşlenmiş veri hedef dizini.
        target_size: Hedef görüntü boyutu.

    Returns:
        İşlenmiş metadata DataFrame.
    """
    logger.info("=" * 60)
    logger.info("ISIC 2019 veri seti işleniyor...")
    logger.info("=" * 60)

    # Ground truth ve metadata yükle
    gt_path = raw_dir / "ISIC_2019_Training_GroundTruth.csv"
    meta_path = raw_dir / "ISIC_2019_Training_Metadata.csv"

    gt_df = pd.read_csv(gt_path)
    meta_df = pd.read_csv(meta_path)

    # One-hot → tek etiket dönüşümü
    label_columns = [c for c in gt_df.columns if c != "image"]
    gt_df["original_label"] = gt_df[label_columns].idxmax(axis=1)
    gt_df["label"] = gt_df["original_label"].map(ISIC_LABEL_MAP)
    gt_df["label_id"] = gt_df["label"].map(LABEL_TO_ID)
    gt_df["category"] = gt_df["label"].map(LABEL_TO_CATEGORY)

    # Metadata ile birleştir
    merged = gt_df.merge(meta_df, on="image", how="left")

    # Görüntü dizini (iç içe klasör yapısını kontrol et)
    img_dir = raw_dir / "ISIC_2019_Training_Input"
    if (img_dir / "ISIC_2019_Training_Input").is_dir():
        img_dir = img_dir / "ISIC_2019_Training_Input"

    # İşlenmiş dizin
    output_img_dir = processed_dir / "images" / "isic2019"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    # Görüntüleri işle
    records = []
    success_count = 0
    fail_count = 0

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="ISIC 2019"):
        image_id = row["image"]
        src_path = img_dir / f"{image_id}.jpg"

        if not src_path.exists():
            fail_count += 1
            continue

        dst_filename = f"{image_id}.jpg"
        dst_path = output_img_dir / dst_filename

        if process_single_image(src_path, dst_path, target_size):
            records.append({
                "image_id": image_id,
                "filename": dst_filename,
                "relative_path": f"images/isic2019/{dst_filename}",
                "source": "isic2019",
                "original_label": row["original_label"],
                "label": row["label"],
                "label_id": row["label_id"],
                "category": row["category"],
                "age": row.get("age_approx", None),
                "sex": row.get("sex", None),
                "localization": row.get("anatom_site_general", None),
                "fitzpatrick": None,  # ISIC 2019'da yok
                "image_type": "dermoscopic",
            })
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"ISIC 2019: {success_count} başarılı, {fail_count} başarısız")
    return pd.DataFrame(records)


def process_pad_ufes(
    raw_dir: Path,
    processed_dir: Path,
    target_size: Tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """
    PAD-UFES-20 veri setini işler.

    Args:
        raw_dir: Ham veri dizini (data/raw/pad_ufes_20).
        processed_dir: İşlenmiş veri hedef dizini.
        target_size: Hedef görüntü boyutu.

    Returns:
        İşlenmiş metadata DataFrame.
    """
    logger.info("=" * 60)
    logger.info("PAD-UFES-20 veri seti işleniyor...")
    logger.info("=" * 60)

    meta_df = pd.read_csv(raw_dir / "metadata.csv")

    # Etiket dönüşümü
    meta_df["label"] = meta_df["diagnostic"].map(PAD_LABEL_MAP)
    meta_df["label_id"] = meta_df["label"].map(LABEL_TO_ID)
    meta_df["category"] = meta_df["label"].map(LABEL_TO_CATEGORY)

    # İşlenmiş dizin
    output_img_dir = processed_dir / "images" / "pad_ufes"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    # Görüntü parçaları dizinleri
    img_dirs = [
        raw_dir / "imgs_part_1",
        raw_dir / "imgs_part_2",
        raw_dir / "imgs_part_3",
    ]

    # Tüm görüntü yollarını indexle
    img_path_map: Dict[str, Path] = {}
    for d in img_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    img_path_map[f.name] = f

    # Görüntüleri işle
    records = []
    success_count = 0
    fail_count = 0

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="PAD-UFES-20"):
        img_id = row["img_id"]
        src_path = img_path_map.get(img_id)

        if src_path is None or not src_path.exists():
            fail_count += 1
            continue

        # PNG → JPG dönüşümü (dosya adı değişir)
        dst_filename = img_id.replace(".png", ".jpg").replace(".PNG", ".jpg")
        dst_path = output_img_dir / dst_filename

        if process_single_image(src_path, dst_path, target_size):
            records.append({
                "image_id": img_id.rsplit(".", 1)[0],
                "filename": dst_filename,
                "relative_path": f"images/pad_ufes/{dst_filename}",
                "source": "pad_ufes_20",
                "original_label": row["diagnostic"],
                "label": row["label"],
                "label_id": row["label_id"],
                "category": row["category"],
                "age": row.get("age", None),
                "sex": row.get("gender", None),
                "localization": row.get("region", None),
                "fitzpatrick": row.get("fitspatrick", None),
                "image_type": "smartphone",
            })
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"PAD-UFES-20: {success_count} başarılı, {fail_count} başarısız")
    return pd.DataFrame(records)


def create_unified_dataset(
    processed_dir: Path,
    isic_df: pd.DataFrame,
    pad_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Birleşik veri seti oluşturur ve train/val/test split yapar.

    Stratified split: Her sınıftan orantılı örnekleme yapılır.

    Args:
        processed_dir: İşlenmiş veri dizini.
        isic_df: ISIC 2019 işlenmiş metadata.
        pad_df: PAD-UFES-20 işlenmiş metadata.
        train_ratio: Eğitim oranı.
        val_ratio: Doğrulama oranı.
        test_ratio: Test oranı.
        random_state: Random seed.

    Returns:
        Split bilgisi eklenmiş birleşik DataFrame.
    """
    from sklearn.model_selection import train_test_split

    logger.info("=" * 60)
    logger.info("Birleşik veri seti oluşturuluyor...")
    logger.info("=" * 60)

    # Birleştir
    unified_df = pd.concat([isic_df, pad_df], ignore_index=True)
    logger.info(f"Toplam görüntü: {len(unified_df)}")

    # Kaynak dağılımı
    logger.info(f"  ISIC 2019: {len(isic_df)}")
    logger.info(f"  PAD-UFES-20: {len(pad_df)}")

    # Sınıf dağılımı
    logger.info("Sınıf dağılımı:")
    for label, count in unified_df["label"].value_counts().items():
        pct = count / len(unified_df) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    # Stratified Train / (Val+Test) split
    train_df, temp_df = train_test_split(
        unified_df,
        test_size=(val_ratio + test_ratio),
        stratify=unified_df["label"],
        random_state=random_state,
    )

    # Stratified Val / Test split
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    # Split etiketleri ekle
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Tekrar birleştir
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    logger.info(f"\nSplit dağılımı:")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(final_df)*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/len(final_df)*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/len(final_df)*100:.1f}%)")

    # CSV olarak kaydet
    output_path = processed_dir / "unified_metadata.csv"
    final_df.to_csv(output_path, index=False)
    logger.info(f"\nBirleşik metadata kaydedildi: {output_path}")

    # Split bazlı CSV'ler
    for split_name in ["train", "val", "test"]:
        split_df = final_df[final_df["split"] == split_name]
        split_path = processed_dir / f"{split_name}_metadata.csv"
        split_df.to_csv(split_path, index=False)
        logger.info(f"  {split_name}: {split_path}")

    # Sınıf ağırlıkları hesapla (class imbalance için)
    class_counts = final_df[final_df["split"] == "train"]["label_id"].value_counts().sort_index()
    total_train = len(train_df)
    n_classes = len(class_counts)
    class_weights = {}
    for label_id, count in class_counts.items():
        weight = total_train / (n_classes * count)
        class_weights[int(label_id)] = round(weight, 4)

    # Sınıf ağırlıklarını kaydet
    weights_df = pd.DataFrame([
        {"label_id": lid, "label": ID_TO_LABEL[lid], "count": int(class_counts[lid]), "weight": w}
        for lid, w in class_weights.items()
    ])
    weights_path = processed_dir / "class_weights.csv"
    weights_df.to_csv(weights_path, index=False)
    logger.info(f"\nSınıf ağırlıkları kaydedildi: {weights_path}")

    # İstatistik özeti kaydet
    _save_dataset_stats(final_df, processed_dir)

    return final_df


def _save_dataset_stats(df: pd.DataFrame, processed_dir: Path) -> None:
    """Veri seti istatistiklerini kaydet."""
    stats = {
        "total_images": len(df),
        "num_classes": df["label"].nunique(),
        "classes": sorted(df["label"].unique().tolist()),
        "split_counts": df["split"].value_counts().to_dict(),
        "source_counts": df["source"].value_counts().to_dict(),
        "label_counts": df["label"].value_counts().to_dict(),
        "image_type_counts": df["image_type"].value_counts().to_dict(),
    }

    import json
    stats_path = processed_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"İstatistikler kaydedildi: {stats_path}")
