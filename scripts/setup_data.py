"""
Dermato-RAG Veri Seti Indirme ve Kurulum Rehberi.

Bu script, arkadaşınızın (veya yeni bir geliştiricinin)
projeyi GitHub'dan klonladıktan sonra veri setlerini
kurmasına yardımcı olur.

Kullanim:
    python scripts/setup_data.py --check
    python scripts/setup_data.py --download-links
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DATA_SOURCES = {
    "ISIC 2019": {
        "dizin": "data/raw/ISIC 2019/",
        "indirme": "https://www.kaggle.com/datasets/andrewmvd/isic-2019",
        "boyut": "~9.1 GB",
        "dosyalar": [
            "ISIC_2019_Training_GroundTruth.csv",
            "ISIC_2019_Training_Metadata.csv",
            "ISIC_2019_Training_Input/ (25,331 .jpg goruntu)",
        ],
        "aciklama": "Dermoskopik cilt lezyonu goruntuleri (8 sinif)",
    },
    "PAD-UFES-20": {
        "dizin": "data/raw/pad_ufes_20/",
        "indirme": "https://data.mendeley.com/datasets/zr7vgbcyr2",
        "boyut": "~3.4 GB",
        "dosyalar": [
            "metadata.csv",
            "imgs_part_1/ (goruntu dosyalari)",
            "imgs_part_2/ (goruntu dosyalari)",
            "imgs_part_3/ (goruntu dosyalari)",
        ],
        "aciklama": "Akilli telefon cilt lezyonu goruntuleri (6 sinif)",
    },
    "Fitzpatrick17k": {
        "dizin": "data/raw/fitzpatrick17k/",
        "indirme": "https://doi.org/10.5281/ZENODO.11101337",
        "boyut": "~1.1 GB",
        "dosyalar": [
            "Fitzpatrick17k-C.csv",
            "Fitzpatrick17k_DiagnosisMapping.xlsx",
            "dermamnist_corrected_224.npz",
        ],
        "aciklama": "Cilt tonu cesitliligi ile etiketli metadata",
    },
}

SHARED_DRIVE_INFO = """
============================================================
GOOGLE DRIVE / ONEDRIVE PAYLASIM YONTEMI
============================================================

Projeyi yukarida belirtilen kaynaklardan veri setlerini 
indirmek yerine, takim arkadaşınızla paylaşılan bir 
bulut klasorundan de alabilirsiniz.

ADIMLAR:
1. Paylaşılan linkten tum data/ klasorunu indirin
2. Indirdiginiz data/ klasorunu proje kokune koyun:
   C:\\Dermato-RAG\\data\\
3. Kontrol edin:
   python scripts/setup_data.py --check
4. Eger sadece raw veri aldıysanız, isleyin:
   python scripts/process_data.py
============================================================
"""


def check_data(verbose: bool = True):
    """Veri setlerinin mevcut durumunu kontrol eder."""
    all_ok = True

    if verbose:
        print("=" * 60)
        print("VERI SETI DURUM KONTROLU")
        print("=" * 60)

    # Raw data
    for name, info in DATA_SOURCES.items():
        path = PROJECT_ROOT / info["dizin"]
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        status = "[OK]" if exists else "[EKSIK]"
        if not exists:
            all_ok = False
        if verbose:
            print(f"  {status} {name}: {info['dizin']}")

    # Processed data
    processed = PROJECT_ROOT / "data" / "processed"
    unified = processed / "unified_metadata.csv"
    proc_ok = unified.exists()
    if not proc_ok:
        all_ok = False
    if verbose:
        status = "[OK]" if proc_ok else "[EKSIK]"
        print(f"  {status} Islenmis veri: data/processed/")

    if verbose:
        print()
        if all_ok:
            print("[TAMAM] Tum veri setleri mevcut!")
        else:
            print("[EKSIK] Bazi veri setleri eksik. Asagidaki komutlari calistirin:")
            print("  python scripts/setup_data.py --download-links")

    return all_ok


def show_download_links():
    """Indirme linklerini ve talimatlari gosterir."""
    print("=" * 60)
    print("VERI SETI INDIRME REHBERI")
    print("=" * 60)

    for name, info in DATA_SOURCES.items():
        print(f"\n--- {name} ---")
        print(f"  Aciklama : {info['aciklama']}")
        print(f"  Boyut    : {info['boyut']}")
        print(f"  Indirme  : {info['indirme']}")
        print(f"  Hedef    : {info['dizin']}")
        print(f"  Dosyalar :")
        for f in info["dosyalar"]:
            print(f"    - {f}")

    print(SHARED_DRIVE_INFO)

    print("\nINDIRDIKTEN SONRA:")
    print("  1. python scripts/setup_data.py --check")
    print("  2. python scripts/process_data.py")
    print("  3. python scripts/validate_data.py")


def main():
    parser = argparse.ArgumentParser(description="Dermato-RAG Veri Kurulum")
    parser.add_argument("--check", action="store_true", help="Veri durumunu kontrol et")
    parser.add_argument("--download-links", action="store_true", help="Indirme linklerini goster")
    args = parser.parse_args()

    if args.check:
        ok = check_data()
        sys.exit(0 if ok else 1)
    elif args.download_links:
        show_download_links()
    else:
        check_data()
        print()
        show_download_links()


if __name__ == "__main__":
    main()
