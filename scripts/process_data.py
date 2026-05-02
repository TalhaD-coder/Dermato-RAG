"""
Dermato-RAG Veri İşleme Scripti.

Ham veri setlerini (ISIC 2019, PAD-UFES-20) işler ve
birleşik bir veri seti oluşturur.

Kullanım:
    python scripts/process_data.py
    python scripts/process_data.py --target-size 224
    python scripts/process_data.py --skip-isic    # Sadece PAD-UFES-20
"""

import argparse
import sys
import time
from pathlib import Path

# Proje kökünü Python path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    create_unified_dataset,
    process_isic2019,
    process_pad_ufes,
)
from src.utils.helpers import set_seed, timer
from src.utils.logger import get_logger, log_separator

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dermato-RAG veri işleme pipeline'ı"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Hedef görüntü boyutu (varsayılan: 224)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Eğitim seti oranı (varsayılan: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Doğrulama seti oranı (varsayılan: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test seti oranı (varsayılan: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (varsayılan: 42)",
    )
    parser.add_argument(
        "--skip-isic",
        action="store_true",
        help="ISIC 2019 işlemeyi atla",
    )
    parser.add_argument(
        "--skip-pad",
        action="store_true",
        help="PAD-UFES-20 işlemeyi atla",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_size = (args.target_size, args.target_size)

    log_separator(logger, "DERMATO-RAG VERİ İŞLEME")
    logger.info(f"Hedef boyut: {target_size}")
    logger.info(f"Split: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    logger.info(f"Seed: {args.seed}")

    set_seed(args.seed)

    # Dizinler
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataframes = []

    # ---- ISIC 2019 ----
    if not args.skip_isic:
        isic_dir = raw_dir / "ISIC 2019"
        if isic_dir.exists():
            with timer("ISIC 2019 işleme"):
                isic_df = process_isic2019(isic_dir, processed_dir, target_size)
                dataframes.append(isic_df)
                logger.info(f"ISIC 2019: {len(isic_df)} görüntü işlendi")
        else:
            logger.warning(f"ISIC 2019 dizini bulunamadı: {isic_dir}")
    else:
        logger.info("ISIC 2019 atlandı (--skip-isic)")
        # Önceden işlenmiş veriyi yükle
        prev_path = processed_dir / "unified_metadata.csv"
        if prev_path.exists():
            prev_df = pd.read_csv(prev_path)
            isic_prev = prev_df[prev_df["source"] == "isic2019"]
            if len(isic_prev) > 0:
                dataframes.append(isic_prev)
                logger.info(f"Önceki ISIC 2019 verisi yüklendi: {len(isic_prev)}")

    # ---- PAD-UFES-20 ----
    if not args.skip_pad:
        pad_dir = raw_dir / "pad_ufes_20"
        if pad_dir.exists():
            with timer("PAD-UFES-20 işleme"):
                pad_df = process_pad_ufes(pad_dir, processed_dir, target_size)
                dataframes.append(pad_df)
                logger.info(f"PAD-UFES-20: {len(pad_df)} görüntü işlendi")
        else:
            logger.warning(f"PAD-UFES-20 dizini bulunamadı: {pad_dir}")
    else:
        logger.info("PAD-UFES-20 atlandı (--skip-pad)")

    # ---- Birleşik Veri Seti Oluştur ----
    if len(dataframes) == 0:
        logger.error("Hiçbir veri seti işlenemedi!")
        sys.exit(1)

    # Kaç DataFrame var buna göre birleştir
    if len(dataframes) == 1:
        isic_part = dataframes[0] if not args.skip_isic else pd.DataFrame()
        pad_part = dataframes[0] if args.skip_isic else pd.DataFrame()
    else:
        isic_part = dataframes[0]
        pad_part = dataframes[1]

    with timer("Birleşik veri seti oluşturma"):
        unified_df = create_unified_dataset(
            processed_dir=processed_dir,
            isic_df=isic_part,
            pad_df=pad_part,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.seed,
        )

    log_separator(logger, "İŞLEM TAMAMLANDI")
    logger.info(f"Toplam: {len(unified_df)} görüntü")
    logger.info(f"Çıktı dizini: {processed_dir}")
    logger.info("Dosyalar:")
    logger.info(f"  - unified_metadata.csv")
    logger.info(f"  - train_metadata.csv")
    logger.info(f"  - val_metadata.csv")
    logger.info(f"  - test_metadata.csv")
    logger.info(f"  - class_weights.csv")
    logger.info(f"  - dataset_stats.json")


if __name__ == "__main__":
    import pandas as pd
    main()
