"""
Dermato-RAG — Değerlendirme Metrikleri (Faz 7).

İçerik:
    Sınıflandırma metrikleri:
        - top_k_accuracy
        - precision/recall/f1 (per-class + macro/micro)
        - confusion_matrix
        - balanced_accuracy
        - cohen_kappa
    Retrieval metrikleri:
        - precision_at_k
        - recall_at_k
        - mean_average_precision (MAP)
        - ndcg_at_k
    RAG kalite metrikleri:
        - article_diversity (unique PMID oranı)
        - category_purity (kategori filtresi sadakati)
        - faithfulness_lexical (RAG snippet'leri ile cevap kelime örtüşümü)

Tüm fonksiyonlar saf NumPy/sklearn; LLM API çağrısı yapmaz.

Kullanım:
    from src.evaluation.metrics import top_k_accuracy
    acc1 = top_k_accuracy(y_true, probs, k=1)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =========================================================
# 1. SINIFLANDIRMA METRİKLERİ
# =========================================================

def top_k_accuracy(y_true: Sequence[int], probs: np.ndarray, k: int = 1) -> float:
    """
    Top-k accuracy: gerçek sınıf, modelin top-k tahmininde mi?

    Args:
        y_true: (N,) gerçek sınıf indeksleri.
        probs: (N, C) olasılık matrisi.
        k: top-k.
    Returns:
        Doğru tahmin oranı (0-1).
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError("probs (N, C) şeklinde olmalı")
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]  # son k sütun = top-k
    return float(np.mean([y in row for y, row in zip(y_true, top_k_preds)]))


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Kapsamlı sınıflandırma metrikleri.

    Returns:
        {
          "accuracy", "balanced_accuracy", "cohen_kappa",
          "precision_macro", "recall_macro", "f1_macro",
          "precision_micro", "recall_micro", "f1_micro",
          "precision_weighted", "recall_weighted", "f1_weighted",
          "per_class": {class_name: {precision, recall, f1, support}},
          "confusion_matrix": np.ndarray,
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    for avg in ("macro", "micro", "weighted"):
        metrics[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        )
        metrics[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        )
        metrics[f"f1_{avg}"] = float(
            f1_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        )

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = {
        k: v for k, v in report.items()
        if k not in ("accuracy", "macro avg", "micro avg", "weighted avg")
    }
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics


# =========================================================
# 2. RETRIEVAL METRİKLERİ
# =========================================================

def precision_at_k(relevant: Sequence[int], retrieved: Sequence[int], k: int) -> float:
    """
    Precision@k = retrieved top-k içindeki relevant oranı.
    `relevant` ve `retrieved` doküman id (ya da pmid) listeleridir.
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = list(retrieved)[:k]
    rel_set = set(relevant)
    hit = sum(1 for r in top_k if r in rel_set)
    return hit / k


def recall_at_k(relevant: Sequence[int], retrieved: Sequence[int], k: int) -> float:
    """Recall@k = ilk k içinde yakalanan relevant / toplam relevant."""
    if not relevant:
        return 0.0
    top_k = list(retrieved)[:k]
    rel_set = set(relevant)
    hit = sum(1 for r in top_k if r in rel_set)
    return hit / len(rel_set)


def average_precision(relevant: Sequence[int], retrieved: Sequence[int]) -> float:
    """Average Precision (AP): tek bir query için."""
    rel_set = set(relevant)
    if not rel_set or not retrieved:
        return 0.0
    score = 0.0
    hits = 0
    for i, doc in enumerate(retrieved, 1):
        if doc in rel_set:
            hits += 1
            score += hits / i
    return score / len(rel_set)


def mean_average_precision(queries: List[Dict[str, Sequence[int]]]) -> float:
    """
    Mean Average Precision = ortalama AP.
    queries: [{"relevant":[ids], "retrieved":[ids]}, ...]
    """
    if not queries:
        return 0.0
    aps = [average_precision(q["relevant"], q["retrieved"]) for q in queries]
    return float(np.mean(aps))


def ndcg_at_k(
    relevant_with_scores: Dict[int, float],
    retrieved: Sequence[int],
    k: int = 10,
) -> float:
    """
    Normalized Discounted Cumulative Gain @ k.

    Args:
        relevant_with_scores: {doc_id: relevance_score} sözlüğü
            (örn. tek kategori için 1.0, diğerlerine 0).
        retrieved: sıralı doküman id listesi.
        k: top-k.
    """
    if not retrieved or not relevant_with_scores or k <= 0:
        return 0.0

    # DCG@k
    dcg = 0.0
    for i, doc in enumerate(list(retrieved)[:k], 1):
        rel = relevant_with_scores.get(doc, 0.0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)

    # Ideal DCG@k (en yüksek rel'leri sırala)
    ideal_rels = sorted(relevant_with_scores.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    return float(dcg / idcg) if idcg > 0 else 0.0


# =========================================================
# 3. RAG KALİTE METRİKLERİ
# =========================================================

def article_diversity(articles: List[Dict[str, Any]]) -> float:
    """
    Unique PMID oranı: 1.0 = her makale benzersiz.
    Pipeline article'larında 'pmid' veya 'link' alanını kullanır.
    """
    if not articles:
        return 0.0
    keys = []
    for a in articles:
        pmid = (a.get("pmid") or "").strip()
        if not pmid:
            link = a.get("link", "") or ""
            pmid = link.rstrip("/").split("/")[-1]
        keys.append(pmid or a.get("title", "")[:80])
    if not any(keys):
        return 0.0
    return len(set(keys)) / len(keys)


def category_purity(articles: List[Dict[str, Any]], expected_category: str) -> float:
    """
    Beklenen kategoriyle eşleşen makale oranı.
    Article'da explicit category yoksa title/snippet içinde kategori adı arar.
    """
    if not articles:
        return 0.0
    expected = expected_category.lower().replace("_", " ")
    hits = 0
    for a in articles:
        text = (a.get("title", "") + " " + a.get("snippet", "")).lower()
        if expected in text or all(w in text for w in expected.split()):
            hits += 1
    return hits / len(articles)


_WORD_RE = re.compile(r"[A-Za-zĞÜŞİÖÇğüşıöç]{4,}")


def _tokens(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def faithfulness_lexical(generated_answer: str, articles: List[Dict[str, Any]]) -> float:
    """
    Üretilen cevabın **kelime düzeyinde** kaynaklara dayanma oranı.

    Yöntem: Cevabın "tıbbi içerik kelimeleri" (4+ harf) ile
    RAG snippet'leri'nin kelime kümesinin Jaccard benzerliği
    (cevap-yönlü: cevap içindeki kelimelerin kaç tanesi kaynaklarda var?).

    NOT: LLM-bazlı RAGAS faithfulness daha doğrudur ama API çağrısı gerektirir.
    Bu hızlı bir lexical proxy.
    """
    if not generated_answer or not articles:
        return 0.0
    answer_tokens = _tokens(generated_answer)
    if not answer_tokens:
        return 0.0
    source_text = " ".join(
        (a.get("snippet", "") + " " + a.get("title", ""))
        for a in articles
    )
    source_tokens = _tokens(source_text)
    if not source_tokens:
        return 0.0
    overlap = answer_tokens & source_tokens
    return len(overlap) / len(answer_tokens)


def citation_count(generated_answer: str) -> int:
    """LLM cevabında [Kaynak N] / [Source N] atıflarını sayar."""
    if not generated_answer:
        return 0
    pattern = re.compile(r"\[(?:Kaynak|Source|Kayn)\s*\d+\]", re.IGNORECASE)
    return len(pattern.findall(generated_answer))


# =========================================================
# 4. ÖZET (RAGAS benzeri tek-skor)
# =========================================================

def rag_quality_summary(
    articles: List[Dict[str, Any]],
    generated_answer: str,
    expected_category: str,
) -> Dict[str, float]:
    """
    Bir analiz çıktısı için RAG kalite özeti.
    """
    return {
        "n_articles": len(articles),
        "diversity": article_diversity(articles),
        "category_purity": category_purity(articles, expected_category),
        "faithfulness_lexical": faithfulness_lexical(generated_answer, articles),
        "citation_count": citation_count(generated_answer),
    }
