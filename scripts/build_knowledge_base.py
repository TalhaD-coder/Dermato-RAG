"""
Dermato-RAG Bilgi Tabani Olusturma Scripti.

PubMed/PMC'den dermatoloji makalelerini ceker ve
data/knowledge_base/raw_docs/ altina kaydeder.

Kullanim:
    python scripts/build_knowledge_base.py
    python scripts/build_knowledge_base.py --max-articles 500
    python scripts/build_knowledge_base.py --full-text-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree

import pandas as pd
from dotenv import load_dotenv

# Proje kokunu ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from Bio import Entrez, Medline
from src.utils.logger import get_logger

logger = get_logger("build_kb")

# ============================================
# PubMed Arama Sorgulari
# ============================================
# Projemizdeki 9 sinifa yonelik spesifik sorgular
# + genel dermatoloji sorgusu.
# Her sorgu icin belirli sayida makale cekilecek.
# ============================================

SEARCH_QUERIES = {
    # Genel dermatoloji tani ve siniflandirma
    "general_dermatology": (
        "(dermatology[MeSH] OR skin diseases[MeSH]) "
        "AND (diagnosis[MeSH] OR classification) "
        "AND (dermoscopy OR dermatoscopy) "
        "AND english[la] AND hasabstract[text]"
    ),
    # Melanom
    "melanoma": (
        "(melanoma[MeSH] OR malignant melanoma) "
        "AND (dermoscopy OR dermatoscopy OR skin lesion) "
        "AND (diagnosis OR detection OR classification) "
        "AND english[la] AND hasabstract[text]"
    ),
    # Bazal hucreli karsinom
    "basal_cell_carcinoma": (
        "(basal cell carcinoma[MeSH] OR basal cell carcinoma) "
        "AND (dermoscopy OR diagnosis OR skin) "
        "AND english[la] AND hasabstract[text]"
    ),
    # Skuamoz hucreli karsinom
    "squamous_cell_carcinoma": (
        "(squamous cell carcinoma[MeSH]) "
        "AND (skin OR cutaneous) "
        "AND (diagnosis OR dermoscopy) "
        "AND english[la] AND hasabstract[text]"
    ),
    # Aktinik keratoz
    "actinic_keratosis": (
        "(actinic keratosis[MeSH] OR actinic keratoses) "
        "AND (diagnosis OR treatment OR dermoscopy) "
        "AND english[la] AND hasabstract[text]"
    ),
    # Nevus (ben)
    "nevus": (
        "(nevus[MeSH] OR melanocytic nevus OR nevi) "
        "AND (dermoscopy OR diagnosis OR classification) "
        "AND english[la] AND hasabstract[text]"
    ),
    # Benign keratoz
    "benign_keratosis": (
        "(seborrheic keratosis[MeSH] OR benign keratosis) "
        "AND (diagnosis OR dermoscopy) "
        "AND english[la] AND hasabstract[text]"
    ),
    # Seboreik keratoz (Faz 6.5 — pipeline sınıfıyla bire bir eşleşmesi için ayrı kategori)
    "seborrheic_keratosis": (
        "(seborrheic keratosis[MeSH] OR seborrhoeic keratosis OR seborrheic keratoses) "
        "AND (dermoscopy OR dermatoscopy OR diagnosis OR clinical features OR pathology) "
        "AND english[la] AND hasabstract[text]"
    ),
    # Vaskuler lezyon
    "vascular_lesion": (
        "(hemangioma[MeSH] OR vascular skin lesion OR angioma) "
        "AND (dermoscopy OR diagnosis) "
        "AND english[la] AND hasabstract[text]"
    ),
    # Dermatofibrom
    "dermatofibroma": (
        "(dermatofibroma[MeSH] OR histiocytoma, benign fibrous[MeSH]) "
        "AND (dermoscopy OR diagnosis OR skin) "
        "AND english[la] AND hasabstract[text]"
    ),
    # AI + Dermatoloji (makale icin onemli)
    "ai_dermatology": (
        "(artificial intelligence OR deep learning OR machine learning) "
        "AND (dermatology OR skin lesion OR dermoscopy) "
        "AND (diagnosis OR classification) "
        "AND english[la] AND hasabstract[text]"
    ),
}

# Her sorgu basina cekilecek varsayilan makale sayisi
DEFAULT_PER_QUERY = 80


def setup_entrez():
    """Entrez API konfigurasyonu."""
    email = os.getenv("NCBI_EMAIL")
    api_key = os.getenv("NCBI_API_KEY")

    if not email or not api_key or "your_" in api_key:
        logger.error("NCBI_EMAIL ve NCBI_API_KEY .env dosyasinda tanimlanmali!")
        sys.exit(1)

    Entrez.email = email
    Entrez.api_key = api_key
    logger.info(f"Entrez API yapilandirildi (email: {email})")


def search_pubmed(query: str, max_results: int = 100) -> List[str]:
    """
    PubMed'de arama yapip makale ID'lerini dondurur.

    Args:
        query: PubMed arama sorgusu.
        max_results: Maksimum sonuc sayisi.

    Returns:
        PubMed ID listesi (PMID).
    """
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            usehistory="y",
        )
        results = Entrez.read(handle)
        handle.close()

        pmids = results.get("IdList", [])
        total = results.get("Count", "0")
        logger.info(f"  Bulunan: {total} | Cekilecek: {len(pmids)}")
        return pmids

    except Exception as e:
        logger.error(f"  PubMed arama hatasi: {e}")
        return []


def fetch_article_details(pmids: List[str]) -> List[Dict]:
    """
    PMID listesi icin makale detaylarini ceker.

    Args:
        pmids: PubMed ID listesi.

    Returns:
        Makale bilgileri listesi.
    """
    if not pmids:
        return []

    articles = []
    batch_size = 50  # NCBI API batcj limit

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="medline",
                retmode="text",
            )
            records = Medline.parse(handle)

            for record in records:
                article = {
                    "pmid": record.get("PMID", ""),
                    "title": record.get("TI", ""),
                    "abstract": record.get("AB", ""),
                    "authors": record.get("AU", []),
                    "journal": record.get("JT", ""),
                    "pub_date": record.get("DP", ""),
                    "mesh_terms": record.get("MH", []),
                    "keywords": record.get("OT", []),
                    "doi": "",
                    "pmc_id": "",
                    "full_text": "",
                }

                # DOI ve PMC ID cikarma
                aid_list = record.get("AID", [])
                for aid in aid_list:
                    if aid.endswith("[doi]"):
                        article["doi"] = aid.replace(" [doi]", "")
                    elif aid.endswith("[pii]"):
                        pass  # pii kullanmiyoruz

                pmc = record.get("PMC", "")
                if pmc:
                    article["pmc_id"] = pmc

                # Sadece abstract'i olan makaleleri al
                if article["abstract"]:
                    articles.append(article)

            handle.close()

        except Exception as e:
            logger.warning(f"  Batch fetch hatasi (index {i}): {e}")

        # API rate limit: API key ile 10 req/sec
        time.sleep(0.15)

    return articles


def fetch_full_text_pmc(pmc_id: str) -> Optional[str]:
    """
    PMC'den tam metin ceker (acik erisimli makaleler icin).

    Args:
        pmc_id: PMC ID (ornegin "PMC1234567").

    Returns:
        Tam metin veya None.
    """
    try:
        handle = Entrez.efetch(
            db="pmc",
            id=pmc_id,
            rettype="xml",
            retmode="xml",
        )
        xml_content = handle.read()
        handle.close()

        # XML'den metin cikar
        root = ElementTree.fromstring(xml_content)
        paragraphs = []

        # Body icerigini cek
        for body in root.iter("body"):
            for p in body.iter("p"):
                text = "".join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)

        if paragraphs:
            return "\n\n".join(paragraphs)

        return None

    except Exception as e:
        logger.debug(f"  PMC tam metin alinamadi ({pmc_id}): {e}")
        return None


def save_articles(
    articles: List[Dict],
    output_dir: Path,
    query_name: str,
) -> int:
    """
    Makaleleri JSON dosyalari olarak kaydeder.

    Args:
        articles: Makale listesi.
        output_dir: Cikti dizini.
        query_name: Sorgu kategorisi adi.

    Returns:
        Kaydedilen makale sayisi.
    """
    category_dir = output_dir / query_name
    category_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for article in articles:
        pmid = article["pmid"]
        filepath = category_dir / f"{pmid}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
        saved += 1

    return saved


def build_index(output_dir: Path) -> pd.DataFrame:
    """
    Tum indirilen makalelerin indeksini olusturur.

    Args:
        output_dir: Makale dizini.

    Returns:
        Indeks DataFrame.
    """
    records = []

    for category_dir in sorted(output_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        for json_file in category_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                article = json.load(f)

            records.append({
                "pmid": article["pmid"],
                "title": article["title"],
                "journal": article["journal"],
                "pub_date": article["pub_date"],
                "category": category_dir.name,
                "has_abstract": bool(article.get("abstract")),
                "has_full_text": bool(article.get("full_text")),
                "has_pmc": bool(article.get("pmc_id")),
                "doi": article.get("doi", ""),
                "mesh_count": len(article.get("mesh_terms", [])),
                "file_path": str(json_file.relative_to(output_dir)),
            })

    df = pd.DataFrame(records)
    index_path = output_dir / "article_index.csv"
    df.to_csv(index_path, index=False)
    logger.info(f"Makale indeksi kaydedildi: {index_path} ({len(df)} makale)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Dermato-RAG Bilgi Tabani")
    parser.add_argument(
        "--max-per-query", type=int, default=DEFAULT_PER_QUERY,
        help=f"Her sorgu basina max makale (varsayilan: {DEFAULT_PER_QUERY})",
    )
    parser.add_argument(
        "--max-articles", type=int, default=0,
        help="Toplam max makale (0 = sinir yok)",
    )
    parser.add_argument(
        "--fetch-full-text", action="store_true",
        help="PMC'den tam metin cekmeyi dene (yavas)",
    )
    parser.add_argument(
        "--queries", nargs="+", default=None,
        help="Sadece belirli sorgulari calistir",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Dermato-RAG Bilgi Tabani Olusturma")
    logger.info("=" * 60)

    setup_entrez()

    output_dir = PROJECT_ROOT / "data" / "knowledge_base" / "raw_docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hangi sorgular calistirilacak?
    if args.queries:
        queries = {k: v for k, v in SEARCH_QUERIES.items() if k in args.queries}
    else:
        queries = SEARCH_QUERIES

    total_articles = 0
    all_articles = []
    seen_pmids = set()

    # Daha once indirilmis PMID'leri topla (tekrar indirmeyi onle)
    for category_dir in output_dir.iterdir():
        if category_dir.is_dir():
            for json_file in category_dir.glob("*.json"):
                seen_pmids.add(json_file.stem)

    if seen_pmids:
        logger.info(f"Daha once indirilmis: {len(seen_pmids)} makale (atlanacak)")

    for query_name, query_str in queries.items():
        logger.info(f"\n--- {query_name.upper()} ---")
        logger.info(f"  Sorgu: {query_str[:80]}...")

        # Arama
        pmids = search_pubmed(query_str, max_results=args.max_per_query)

        # Zaten indirilmisleri filtrele
        new_pmids = [p for p in pmids if p not in seen_pmids]
        if len(new_pmids) < len(pmids):
            logger.info(f"  Yeni: {len(new_pmids)} (zaten mevcut: {len(pmids) - len(new_pmids)})")

        if not new_pmids:
            logger.info("  Yeni makale yok, atlaniyor.")
            continue

        # Detaylari cek
        articles = fetch_article_details(new_pmids)
        logger.info(f"  Abstract'li makale: {len(articles)}")

        # Tam metin cek (opsiyonel)
        if args.fetch_full_text:
            ft_count = 0
            for article in articles:
                if article.get("pmc_id"):
                    full_text = fetch_full_text_pmc(article["pmc_id"])
                    if full_text:
                        article["full_text"] = full_text
                        ft_count += 1
                    time.sleep(0.15)
            logger.info(f"  Tam metin alinan: {ft_count}")

        # Kaydet
        saved = save_articles(articles, output_dir, query_name)
        logger.info(f"  Kaydedilen: {saved}")

        total_articles += saved
        all_articles.extend(articles)

        # PMID'leri kaydet
        for a in articles:
            seen_pmids.add(a["pmid"])

        # Toplam limit kontrolu
        if args.max_articles > 0 and total_articles >= args.max_articles:
            logger.info(f"\nMaksimum makale sayisina ulasildi: {total_articles}")
            break

        time.sleep(0.5)  # Sorgular arasi bekleme

    # Indeks olustur
    logger.info("\n" + "=" * 60)
    logger.info("Indeks olusturuluyor...")
    index_df = build_index(output_dir)

    # Ozet
    logger.info("\n" + "=" * 60)
    logger.info("SONUC OZETI")
    logger.info("=" * 60)
    logger.info(f"Toplam makale: {len(index_df)}")
    logger.info(f"Kategoriler:")
    for cat, count in index_df["category"].value_counts().items():
        logger.info(f"  {cat}: {count}")
    logger.info(f"Abstract'li: {index_df['has_abstract'].sum()}")
    logger.info(f"Tam metinli: {index_df['has_full_text'].sum()}")
    logger.info(f"PMC'li: {index_df['has_pmc'].sum()}")
    logger.info(f"\nDosyalar: {output_dir}")


if __name__ == "__main__":
    main()
