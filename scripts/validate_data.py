"""
Dermato-RAG Veri Dogrulama Scripti.

Islenmis veri setinin butunlugunu ve kalitesini dogrular.
Tum kontrolleri otomatik yapar ve rapor uretir.

Kullanim:
    python scripts/validate_data.py
    python scripts/validate_data.py --check-images
    python scripts/validate_data.py --sample-size 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Proje kokunu ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("validate_data")


def validate_metadata(processed_dir: Path) -> dict:
    """Metadata dosyalarini dogrula."""
    results = {"passed": 0, "failed": 0, "errors": []}

    # 1. Dosya varlik kontrolleri
    required_files = [
        "unified_metadata.csv",
        "train_metadata.csv",
        "val_metadata.csv",
        "test_metadata.csv",
        "class_weights.csv",
        "dataset_stats.json",
    ]

    for fname in required_files:
        path = processed_dir / fname
        if path.exists():
            results["passed"] += 1
            logger.info(f"  [OK] {fname} mevcut")
        else:
            results["failed"] += 1
            results["errors"].append(f"Dosya bulunamadi: {fname}")
            logger.error(f"  [FAIL] {fname} bulunamadi!")

    if results["failed"] > 0:
        return results

    # 2. Unified metadata kontrolleri
    df = pd.read_csv(processed_dir / "unified_metadata.csv")

    # Toplam goruntu sayisi
    if len(df) == 27629:
        results["passed"] += 1
        logger.info(f"  [OK] Toplam goruntu: {len(df)}")
    else:
        results["failed"] += 1
        results["errors"].append(f"Goruntu sayisi hatasi: {len(df)} (beklenen: 27629)")
        logger.error(f"  [FAIL] Goruntu sayisi: {len(df)} (beklenen: 27629)")

    # Sinif sayisi
    n_classes = df["label"].nunique()
    if n_classes == 9:
        results["passed"] += 1
        logger.info(f"  [OK] Sinif sayisi: {n_classes}")
    else:
        results["failed"] += 1
        results["errors"].append(f"Sinif sayisi hatasi: {n_classes} (beklenen: 9)")

    # Split kontrolleri
    split_counts = df["split"].value_counts()
    total_from_splits = split_counts.sum()
    if total_from_splits == len(df):
        results["passed"] += 1
        logger.info(f"  [OK] Split toplamlar tutarli")
    else:
        results["failed"] += 1
        results["errors"].append("Split toplamlari tutarsiz")

    # Eksik deger kontrolleri
    critical_cols = ["image_id", "filename", "relative_path", "source", "label", "label_id", "split"]
    for col in critical_cols:
        missing = df[col].isna().sum()
        if missing == 0:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(f"Sutun '{col}' icinde {missing} eksik deger")
            logger.error(f"  [FAIL] '{col}' sutununda {missing} eksik deger")

    # Label ID aralik kontrolu
    label_ids = sorted(df["label_id"].unique())
    if label_ids == list(range(9)):
        results["passed"] += 1
        logger.info(f"  [OK] Label ID'ler 0-8 arasinda")
    else:
        results["failed"] += 1
        results["errors"].append(f"Label ID araligi hatali: {label_ids}")

    # Split CSV tutarliligi
    for split_name in ["train", "val", "test"]:
        split_df = pd.read_csv(processed_dir / f"{split_name}_metadata.csv")
        unified_count = len(df[df["split"] == split_name])
        if len(split_df) == unified_count:
            results["passed"] += 1
            logger.info(f"  [OK] {split_name}_metadata.csv tutarli ({len(split_df)} kayit)")
        else:
            results["failed"] += 1
            results["errors"].append(
                f"{split_name}_metadata.csv tutarsiz: {len(split_df)} vs {unified_count}"
            )

    # Stats JSON
    with open(processed_dir / "dataset_stats.json") as f:
        stats = json.load(f)
    if stats["total_images"] == len(df):
        results["passed"] += 1
        logger.info(f"  [OK] dataset_stats.json tutarli")
    else:
        results["failed"] += 1
        results["errors"].append("dataset_stats.json tutarsiz")

    # Class weights
    weights_df = pd.read_csv(processed_dir / "class_weights.csv")
    if len(weights_df) == 9 and (weights_df["weight"] > 0).all():
        results["passed"] += 1
        logger.info(f"  [OK] Sinif agirliklari gecerli")
    else:
        results["failed"] += 1
        results["errors"].append("Sinif agirliklari gecersiz")

    return results


def validate_images(
    processed_dir: Path,
    sample_size: int = 0,
    check_all: bool = False,
) -> dict:
    """Goruntu dosyalarini dogrula."""
    results = {"passed": 0, "failed": 0, "errors": [], "checked": 0}

    df = pd.read_csv(processed_dir / "unified_metadata.csv")

    if check_all:
        sample = df
    elif sample_size > 0:
        sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        # Varsayilan: her siniftan 5 ornek
        sample = df.groupby("label").apply(
            lambda x: x.sample(n=min(5, len(x)), random_state=42)
        ).reset_index(drop=True)

    logger.info(f"\nGoruntu dogrulama: {len(sample)} goruntu kontrol edilecek")

    missing = 0
    corrupted = 0
    wrong_size = 0
    wrong_mode = 0

    for idx, row in sample.iterrows():
        img_path = processed_dir / row["relative_path"]
        results["checked"] += 1

        if not img_path.exists():
            missing += 1
            continue

        try:
            img = Image.open(img_path)
            img.load()  # Lazy loading'i zorla

            if img.size != (224, 224):
                wrong_size += 1

            if img.mode != "RGB":
                wrong_mode += 1

        except Exception as e:
            corrupted += 1

    if missing == 0:
        results["passed"] += 1
        logger.info(f"  [OK] Eksik goruntu: 0")
    else:
        results["failed"] += 1
        results["errors"].append(f"{missing} goruntu dosyasi bulunamadi")
        logger.error(f"  [FAIL] {missing} goruntu eksik!")

    if corrupted == 0:
        results["passed"] += 1
        logger.info(f"  [OK] Bozuk goruntu: 0")
    else:
        results["failed"] += 1
        results["errors"].append(f"{corrupted} goruntu bozuk")

    if wrong_size == 0:
        results["passed"] += 1
        logger.info(f"  [OK] Boyut tutarli: hepsi 224x224")
    else:
        results["failed"] += 1
        results["errors"].append(f"{wrong_size} goruntu yanlis boyutta")

    if wrong_mode == 0:
        results["passed"] += 1
        logger.info(f"  [OK] Mode tutarli: hepsi RGB")
    else:
        results["failed"] += 1
        results["errors"].append(f"{wrong_mode} goruntu yanlis modda")

    return results


def main():
    parser = argparse.ArgumentParser(description="Dermato-RAG Veri Dogrulama")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Islenmis veri dizini",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Tum goruntuler kontrol edilsin (yavas)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Kontrol edilecek goruntu sayisi (0 = sinif basi 5)",
    )
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / args.processed_dir

    logger.info("=" * 60)
    logger.info("Dermato-RAG Veri Dogrulama Raporu")
    logger.info("=" * 60)

    start_time = time.time()

    # 1. Metadata dogrulama
    logger.info("\n[1/2] Metadata dogrulama...")
    meta_results = validate_metadata(processed_dir)

    # 2. Goruntu dogrulama
    logger.info("\n[2/2] Goruntu dogrulama...")
    img_results = validate_images(
        processed_dir,
        sample_size=args.sample_size,
        check_all=args.check_images,
    )

    elapsed = time.time() - start_time

    # Sonuc raporu
    total_passed = meta_results["passed"] + img_results["passed"]
    total_failed = meta_results["failed"] + img_results["failed"]
    all_errors = meta_results["errors"] + img_results["errors"]

    logger.info("\n" + "=" * 60)
    logger.info("DOGRULAMA SONUCU")
    logger.info("=" * 60)
    logger.info(f"Gecen: {total_passed} | Basarisiz: {total_failed} | Sure: {elapsed:.1f}s")

    if total_failed == 0:
        logger.info("\n[BASARILI] Tum kontroller gecti!")
        return 0
    else:
        logger.error(f"\n[BASARISIZ] {total_failed} kontrol basarisiz:")
        for err in all_errors:
            logger.error(f"  - {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
