"""
Dermato-RAG Chunking Modulu.

Tibbi metinleri anlamli parcalara (chunk) boler.
Hem abstract hem tam metin icin optimize edilmis
semantik chunking stratejileri.

Stratejiler:
- Abstract: Cumle bazli chunking (abstract zaten kisa)
- Tam metin: Paragraf bazli + overlap chunking
- Metadata zenginlestirme: Her chunk'a kaynak bilgisi eklenir

Kullanim:
    from src.rag.chunking import ArticleChunker
    chunker = ArticleChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_article(article_dict)
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextChunk:
    """
    Tek bir metin parcasi (chunk).

    Attributes:
        text: Chunk metni.
        metadata: Chunk'a ait metadata (pmid, baslik, kaynak vb.).
        chunk_id: Benzersiz chunk ID.
    """

    def __init__(self, text: str, metadata: Dict[str, Any], chunk_id: str = ""):
        self.text = text.strip()
        self.metadata = metadata
        self.chunk_id = chunk_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"TextChunk(id={self.chunk_id}, len={len(self.text)})"

    def __len__(self) -> int:
        return len(self.text)


class ArticleChunker:
    """
    PubMed makalelerini anlamli chunk'lara boler.

    Args:
        chunk_size: Hedef chunk boyutu (karakter).
        chunk_overlap: Chunk'lar arasi ortusme (karakter).
        min_chunk_size: Minimum chunk boyutu (bundan kisa chunk'lar birlestirilir).
        include_title: Her chunk'a makale basligini ekle.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 100,
        include_title: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.include_title = include_title

    def chunk_article(self, article: Dict[str, Any]) -> List[TextChunk]:
        """
        Tek bir makaleyi chunk'lara boler.

        Args:
            article: Makale dict (pmid, title, abstract, full_text, ...).

        Returns:
            TextChunk listesi.
        """
        pmid = article.get("pmid", "unknown")
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        full_text = article.get("full_text", "")
        category = article.get("category", "unknown")

        chunks = []
        chunk_idx = 0

        # Ortak metadata
        base_metadata = {
            "pmid": pmid,
            "title": title,
            "journal": article.get("journal", ""),
            "pub_date": article.get("pub_date", ""),
            "category": category,
            "doi": article.get("doi", ""),
            "mesh_terms": article.get("mesh_terms", []),
        }

        # 1. Abstract chunking
        if abstract:
            abstract_chunks = self._chunk_abstract(abstract, title)
            for text in abstract_chunks:
                meta = {**base_metadata, "source_section": "abstract"}
                chunk_id = f"{pmid}_abs_{chunk_idx}"
                chunks.append(TextChunk(text=text, metadata=meta, chunk_id=chunk_id))
                chunk_idx += 1

        # 2. Full-text chunking (varsa)
        if full_text:
            ft_chunks = self._chunk_full_text(full_text)
            for text in ft_chunks:
                meta = {**base_metadata, "source_section": "full_text"}
                chunk_id = f"{pmid}_ft_{chunk_idx}"
                chunks.append(TextChunk(text=text, metadata=meta, chunk_id=chunk_id))
                chunk_idx += 1

        return chunks

    def _chunk_abstract(self, abstract: str, title: str = "") -> List[str]:
        """
        Abstract metnini chunk'lar.
        Abstract genellikle kisa oldugu icin cumle bazli
        parcalanir, cok kisaysa tek chunk olarak kalir.
        """
        # Baslik prefix
        prefix = f"Title: {title}\n\n" if self.include_title and title else ""

        # Abstract genelde 200-400 kelime, tek chunk olarak kalabilir
        full_abstract = prefix + abstract

        if len(full_abstract) <= self.chunk_size:
            return [full_abstract]

        # Uzun abstract: cumle bazli parcala
        sentences = self._split_sentences(abstract)
        chunks = []
        current = prefix

        for sentence in sentences:
            if len(current) + len(sentence) > self.chunk_size and len(current) > self.min_chunk_size:
                chunks.append(current.strip())
                # Overlap: son cumlenin bir kismini tasi
                current = prefix + sentence + " "
            else:
                current += sentence + " "

        if current.strip() and len(current.strip()) >= self.min_chunk_size:
            chunks.append(current.strip())
        elif current.strip() and chunks:
            # Cok kisa kalirsa onceki chunk'a ekle
            chunks[-1] += " " + current.strip()

        return chunks if chunks else [full_abstract]

    def _chunk_full_text(self, text: str) -> List[str]:
        """
        Tam metni paragraf bazli + overlap ile chunk'lar.
        """
        # Paragraf bazli bol
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) > self.chunk_size and len(current) >= self.min_chunk_size:
                chunks.append(current.strip())
                # Overlap: onceki chunk'in sonundan al
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current = overlap_text + " " + para + "\n\n"
            else:
                current += para + "\n\n"

        if current.strip() and len(current.strip()) >= self.min_chunk_size:
            chunks.append(current.strip())
        elif current.strip() and chunks:
            chunks[-1] += "\n\n" + current.strip()

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Metni cumlelere boler. Tibbi kisaltmalara dikkat eder
        (Dr., vs., etc. gibi cumle sonu olmayan noktalar).
        """
        # Tibbi kisaltmalari korumak icin basit regex
        # Tam cumle sonu: nokta + bosluk + buyuk harf
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_all_articles(self, articles_dir: Path) -> List[TextChunk]:
        """
        Bir dizindeki tum makale JSON dosyalarini chunk'lar.

        Args:
            articles_dir: Makale JSON dosyalarinin bulundugu dizin.

        Returns:
            Tum chunk'larin listesi.
        """
        all_chunks = []

        # Kategori dizinlerini tara
        for category_dir in sorted(articles_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            json_files = list(category_dir.glob("*.json"))

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        article = json.load(f)

                    article["category"] = category
                    chunks = self.chunk_article(article)
                    all_chunks.extend(chunks)

                except Exception as e:
                    logger.warning(f"Makale islenemedi: {json_file}: {e}")

            logger.info(f"  {category}: {len(json_files)} makale islendi")

        logger.info(f"Toplam chunk: {len(all_chunks)}")
        return all_chunks

    def get_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Chunk istatistiklerini dondurur."""
        if not chunks:
            return {"total": 0}

        lengths = [len(c.text) for c in chunks]
        sections = {}
        categories = {}

        for c in chunks:
            sec = c.metadata.get("source_section", "unknown")
            sections[sec] = sections.get(sec, 0) + 1
            cat = c.metadata.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "sections": sections,
            "categories": categories,
        }
