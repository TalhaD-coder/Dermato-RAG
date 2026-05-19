"""
Dermato-RAG — Knowledge Base Incremental Ekleme Scripti.

Mevcut ChromaDB'ye dokunmadan, belirtilen kategori(ler)deki
yeni makaleleri chunk'layip ekler. `build_from_articles` aksine
mevcut kayıtları silmez.

Kullanım:
    python scripts/add_to_knowledge_base.py --categories seborrheic_keratosis
    python scripts/add_to_knowledge_base.py --categories seborrheic_keratosis dermatofibroma
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import ArticleChunker
from src.rag.knowledge_base import KnowledgeBase
from src.utils.logger import get_logger

logger = get_logger("add_to_kb")


def add_categories(categories: list[str], batch_size: int = 100) -> None:
    raw_docs = PROJECT_ROOT / "data" / "knowledge_base" / "raw_docs"
    if not raw_docs.exists():
        raise FileNotFoundError(f"Bilgi tabani kaynagi yok: {raw_docs}")

    kb = KnowledgeBase(
        collection_name="dermato_kb",
        persist_dir=str(PROJECT_ROOT / "data" / "embeddings" / "chromadb"),
        embedding_model="pritamdeka/S-PubMedBert-MS-MARCO",
    )
    existing_count = kb.collection.count()
    logger.info(f"Mevcut KB: {existing_count} chunk")

    chunker = ArticleChunker(chunk_size=512, chunk_overlap=64)

    for category in categories:
        cat_dir = raw_docs / category
        if not cat_dir.exists():
            logger.warning(f"Kategori dizini yok, atlanıyor: {category}")
            continue

        logger.info(f"\n=== {category} ===")
        # 1) Bu kategoriyi chunk'la
        articles = list(cat_dir.glob("*.json"))
        logger.info(f"  {len(articles)} makale bulundu, chunk'lanıyor...")

        # ArticleChunker'in dizin metodunu kullanalim — kategori bazında yapacak
        all_chunks = chunker.chunk_all_articles(cat_dir.parent)
        cat_chunks = [c for c in all_chunks if c.metadata.get("category") == category]
        logger.info(f"  {len(cat_chunks)} chunk üretildi")

        if not cat_chunks:
            continue

        # 2) Aynı chunk_id'ler varsa önce sil (rebuild kategorisi gibi)
        existing_ids_in_cat = []
        try:
            res = kb.collection.get(where={"category": category}, include=[])
            existing_ids_in_cat = res.get("ids", [])
        except Exception:
            pass
        if existing_ids_in_cat:
            logger.info(f"  Eski {len(existing_ids_in_cat)} chunk siliniyor (re-index)...")
            kb.collection.delete(ids=existing_ids_in_cat)

        # 3) Embedding hesapla
        logger.info("  Embedding'ler hesaplanıyor...")
        texts = [c.text for c in cat_chunks]
        embeddings = kb.embedding_model.encode_texts(texts)

        # 4) ChromaDB'ye ekle (batch)
        added = 0
        for i in range(0, len(cat_chunks), batch_size):
            batch = cat_chunks[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = []
            for c in batch:
                m = c.metadata
                metadatas.append({
                    "pmid": str(m.get("pmid", "")),
                    "title": str(m.get("title", ""))[:500],
                    "category": str(m.get("category", "")),
                    "source_section": str(m.get("source_section", "")),
                    "journal": str(m.get("journal", ""))[:200],
                    "pub_date": str(m.get("pub_date", "")),
                    "doi": str(m.get("doi", "")),
                })
            kb.collection.add(
                ids=ids,
                documents=documents,
                embeddings=batch_embs.tolist(),
                metadatas=metadatas,
            )
            added += len(batch)
        logger.info(f"  {added} chunk eklendi")

    final_count = kb.collection.count()
    logger.info(f"\n✓ Bitti. Toplam KB chunk: {existing_count} → {final_count} (+{final_count - existing_count})")


def main():
    parser = argparse.ArgumentParser(description="Dermato-RAG KB incremental ekleme")
    parser.add_argument("--categories", nargs="+", required=True,
                        help="Eklenecek kategori adları (raw_docs altındaki klasör adları)")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    add_categories(args.categories, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
