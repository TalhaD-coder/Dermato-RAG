"""
Dermato-RAG Knowledge Base Modulu.

ChromaDB tabanli vektor veritabani yonetimi.
Chunk'lari indeksler ve aranabilir hale getirir.

Kullanim:
    from src.rag.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    kb.build_from_articles("data/knowledge_base/raw_docs")
    results = kb.search("melanoma dermoscopy features", top_k=5)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.chunking import ArticleChunker, TextChunk
from src.rag.embedding import EmbeddingModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Proje kok dizini
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class KnowledgeBase:
    """
    Dermatoloji bilgi tabani.

    PubMed makalelerini chunk'layip ChromaDB'ye
    indeksler ve semantik arama saglar.

    Args:
        collection_name: ChromaDB koleksiyon adi.
        persist_dir: ChromaDB kalicilik dizini.
        embedding_model: Kullanilacak embedding model adi.
        chunk_size: Chunk boyutu (karakter).
        chunk_overlap: Overlap boyutu (karakter).
    """

    def __init__(
        self,
        collection_name: str = "dermato_kb",
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.collection_name = collection_name

        if persist_dir is None:
            self.persist_dir = str(PROJECT_ROOT / "data" / "embeddings" / "chromadb")
        else:
            self.persist_dir = persist_dir

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Chunker
        self.chunker = ArticleChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Embedding model (lazy load)
        self._embedding_model_name = embedding_model
        self._embedding_model = None

        # ChromaDB client (lazy load)
        self._client = None
        self._collection = None

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Embedding modelini lazy load eder."""
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel(
                model_name=self._embedding_model_name
            )
        return self._embedding_model

    @property
    def client(self):
        """ChromaDB client'i lazy load eder."""
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info(f"ChromaDB basladi: {self.persist_dir}")
        return self._client

    @property
    def collection(self):
        """ChromaDB koleksiyonunu dondurur veya olusturur."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Koleksiyon: {self.collection_name} "
                f"({self._collection.count()} kayit)"
            )
        return self._collection

    def build_from_articles(
        self,
        articles_dir: str,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Makale dizininden bilgi tabanini olusturur.

        Args:
            articles_dir: Makale JSON dizini.
            batch_size: ChromaDB ekleme batch boyutu.

        Returns:
            Islem istatistikleri.
        """
        articles_path = Path(articles_dir)
        if not articles_path.exists():
            raise FileNotFoundError(f"Dizin bulunamadi: {articles_dir}")

        logger.info("=" * 60)
        logger.info("Bilgi tabani olusturuluyor...")
        logger.info("=" * 60)

        # 1. Chunk'la
        logger.info("\n[1/3] Makaleler chunk'laniyor...")
        chunks = self.chunker.chunk_all_articles(articles_path)
        stats = self.chunker.get_stats(chunks)
        logger.info(f"Toplam chunk: {stats['total_chunks']}")
        logger.info(f"Ortalama uzunluk: {stats['avg_length']:.0f} karakter")

        if not chunks:
            logger.warning("Hic chunk olusturulamadi!")
            return {"status": "empty"}

        # 2. Embedding hesapla
        logger.info("\n[2/3] Embedding'ler hesaplaniyor...")
        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.encode_texts(texts)
        logger.info(f"Embedding boyutu: {embeddings.shape}")

        # 3. ChromaDB'ye ekle
        logger.info("\n[3/3] ChromaDB'ye indeksleniyor...")

        # Mevcut koleksiyonu temizle
        existing = self.collection.count()
        if existing > 0:
            logger.info(f"Mevcut {existing} kayit temizleniyor...")
            self.collection.delete(
                where={"pmid": {"$ne": ""}}
            )

        # Batch halinde ekle
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.text for c in batch_chunks]
            metadatas = []
            for c in batch_chunks:
                meta = {
                    "pmid": str(c.metadata.get("pmid", "")),
                    "title": str(c.metadata.get("title", ""))[:500],
                    "category": str(c.metadata.get("category", "")),
                    "source_section": str(c.metadata.get("source_section", "")),
                    "journal": str(c.metadata.get("journal", ""))[:200],
                    "pub_date": str(c.metadata.get("pub_date", "")),
                    "doi": str(c.metadata.get("doi", "")),
                }
                metadatas.append(meta)

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings.tolist(),
                metadatas=metadatas,
            )
            total_added += len(batch_chunks)

            if (i // batch_size + 1) % 5 == 0:
                logger.info(f"  Eklenen: {total_added}/{len(chunks)}")

        logger.info(f"\nIndeksleme tamamlandi: {total_added} chunk")

        result = {
            "status": "success",
            "total_chunks": len(chunks),
            "embedding_dim": int(embeddings.shape[1]),
            "model": self.embedding_model.model_name,
            "chunk_stats": stats,
        }

        # Sonuclari kaydet
        stats_path = Path(self.persist_dir) / "kb_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Bilgi tabaninda semantik arama yapar.

        Args:
            query: Arama sorgusu.
            top_k: Dondurulecek sonuc sayisi.
            category_filter: Kategori filtresi (ornegin "melanoma").

        Returns:
            Sonuc listesi [{text, metadata, score}, ...].
        """
        # Sorguyu embed et
        query_embedding = self.embedding_model.encode_query(query)

        # ChromaDB sorgusu
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Sonuclari duzelt
        search_results = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                search_results.append({
                    "text": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # cosine distance -> similarity
                })

        return search_results

    def get_stats(self) -> Dict[str, Any]:
        """Bilgi tabani istatistiklerini dondurur."""
        count = self.collection.count()
        return {
            "collection": self.collection_name,
            "total_chunks": count,
            "persist_dir": self.persist_dir,
            "embedding_model": self.embedding_model.model_name,
            "embedding_dim": self.embedding_model.dimension,
        }
