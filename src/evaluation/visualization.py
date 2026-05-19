"""
Dermato-RAG — Görselleştirme Modülü (Faz 7).

Sınıflandırma ve retrieval sonuçlarını makale kalitesinde
matplotlib grafiklerine çevirir.

Üretilen grafikler:
    - plot_confusion_matrix
    - plot_per_class_metrics (precision/recall/f1 bar)
    - plot_topk_curve (top-k accuracy çizgi)
    - plot_rag_diversity (sınıf başına unique PMID)
    - plot_retrieval_comparison (dense vs hybrid vs reranker)
    - plot_rag_vs_norag (literatürün katkısı)

Tüm grafikler PNG olarak `outputs/figures/` altına kaydedilir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless ortam için
import matplotlib.pyplot as plt


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    cmap: str = "Blues",
) -> Path:
    """sklearn confusion_matrix çıktısını ısı haritasına çevirir."""
    out = _ensure_dir(out_path)
    cm_arr = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_arr, row_sums, where=row_sums > 0)
        display = cm_norm
        fmt = ".2f"
        title = title + " (row-normalized)"
    else:
        display = cm_arr
        fmt = "d"

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(display, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=13, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)

    thresh = display.max() / 2.0
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = display[i, j]
            ax.text(
                j, i, format(val, fmt) if (val > 0 or not normalize) else "",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_class_metrics(
    per_class: Dict[str, Dict[str, float]],
    out_path: str,
    title: str = "Per-Class Precision / Recall / F1",
) -> Path:
    """Per-class precision/recall/f1 bar chart."""
    out = _ensure_dir(out_path)
    classes = list(per_class.keys())
    p = [per_class[c].get("precision", 0) for c in classes]
    r = [per_class[c].get("recall", 0) for c in classes]
    f = [per_class[c].get("f1-score", 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, p, width, label="Precision", color="#6366F1")
    ax.bar(x,        r, width, label="Recall",    color="#06B6D4")
    ax.bar(x + width, f, width, label="F1",       color="#10B981")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_topk_curve(
    topk_results: Dict[int, float],
    out_path: str,
    title: str = "Top-k Accuracy",
) -> Path:
    """Top-k accuracy çizgi grafiği."""
    out = _ensure_dir(out_path)
    ks = sorted(topk_results.keys())
    accs = [topk_results[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, accs, marker="o", color="#6366F1", linewidth=2)
    for k, a in zip(ks, accs):
        ax.text(k, a + 0.012, f"{a*100:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k")
    ax.set_ylabel("Top-k Accuracy")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rag_diversity(
    diversity_by_class: Dict[str, float],
    out_path: str,
    title: str = "RAG Article Diversity per Class",
) -> Path:
    """Sınıf başına unique PMID oranı (RAG çeşitliliği)."""
    out = _ensure_dir(out_path)
    classes = list(diversity_by_class.keys())
    vals = [diversity_by_class[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#10B981" if v >= 0.8 else "#F59E0B" if v >= 0.5 else "#EF4444" for v in vals]
    ax.bar(classes, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Unique PMID ratio")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.axhline(0.8, linestyle="--", color="gray", alpha=0.5, label="80% target")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_retrieval_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str,
    out_path: str,
    title: Optional[str] = None,
) -> Path:
    """
    Birden fazla retrieval modunun (dense, hybrid, reranker) tek bir metrik üzerinde karşılaştırması.

    results = {
        "dense":         {"map": 0.62, "ndcg@5": 0.71, ...},
        "hybrid":        {...},
        "hybrid+rerank": {...},
    }
    """
    out = _ensure_dir(out_path)
    modes = list(results.keys())
    vals = [results[m].get(metric, 0.0) for m in modes]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(modes, vals, color=["#6366F1", "#06B6D4", "#10B981", "#F59E0B"][:len(modes)])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, (max(vals) if vals else 1) * 1.25 + 0.05)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Retrieval Comparison — {metric}", fontsize=12, weight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rag_vs_norag(
    rag_metrics: Dict[str, float],
    norag_metrics: Dict[str, float],
    metric_names: List[str],
    out_path: str,
    title: str = "RAG vs No-RAG",
) -> Path:
    """
    İki konfigürasyonu yan yana bar chart — literatür entegrasyonunun katkısı.
    """
    out = _ensure_dir(out_path)
    rag_vals = [rag_metrics.get(m, 0.0) for m in metric_names]
    norag_vals = [norag_metrics.get(m, 0.0) for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, rag_vals, width, label="RAG (with literature)", color="#6366F1")
    ax.bar(x + width / 2, norag_vals, width, label="No-RAG (LLM only)",   color="#94A3B8")
    for i, v in enumerate(rag_vals):
        ax.text(i - width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(norag_vals):
        ax.text(i + width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylim(0, max(max(rag_vals, default=0), max(norag_vals, default=0)) * 1.2 + 0.05)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
