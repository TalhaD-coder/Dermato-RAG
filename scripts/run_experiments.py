"""
Dermato-RAG — Faz 7 Deney Çalıştırma Script'i.

Tüm akademik karşılaştırma deneylerini çalıştırır ve sonuçları
`outputs/` klasörüne (CSV + JSON + PNG) yazar.

Çalıştırılan deneyler:
    [1] Vision sınıflandırma (TTA on/off karşılaştırması)
    [2] Retrieval modu karşılaştırması (dense / hybrid / hybrid+rerank)
    [3] RAG vs No-RAG (LLM kaynağa sadakat — LLM API çağrısı, pahalı)
    [4] Top-k accuracy curve

Kullanım:
    # Hızlı (test örneklem azaltılır, LLM deneyi atlanır)
    python scripts/run_experiments.py --quick

    # Tam (test seti, LLM dahil — 30-60 dk + API kotası gerekir)
    python scripts/run_experiments.py --full

    # Belirli deneyler
    python scripts/run_experiments.py --experiments vision retrieval

Çıktılar:
    outputs/
      ├── experiment_results.json      # tüm sayısal sonuçlar
      ├── classification_report.csv    # per-class precision/recall/f1
      ├── topk_accuracy.csv
      ├── retrieval_comparison.csv
      ├── rag_vs_norag.csv             # (LLM dahilse)
      └── figures/
          ├── confusion_matrix.png
          ├── per_class_metrics.png
          ├── topk_curve.png
          ├── rag_diversity.png
          ├── retrieval_map.png
          ├── retrieval_ndcg.png
          └── rag_vs_norag.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.utils.logger import get_logger
from src.evaluation.benchmarks import (
    CLASS_LABELS,
    benchmark_retrieval,
    benchmark_rag_vs_norag,
    benchmark_vision_classification,
    build_test_queries,
    pick_test_cases,
)
from src.evaluation.metrics import (
    article_diversity,
    top_k_accuracy,
)
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_rag_diversity,
    plot_rag_vs_norag,
    plot_retrieval_comparison,
    plot_topk_curve,
)

logger = get_logger("experiments")

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"


# =========================================================
# Yardımcı: NumPy → JSON-uyumlu
# =========================================================
def _to_jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


# =========================================================
# Deney 1: Vision Classification
# =========================================================
def run_vision_classification(max_samples: int) -> Dict[str, Any]:
    logger.info("\n" + "=" * 60)
    logger.info("DENEY 1: Vision Classification (TTA on vs off)")
    logger.info("=" * 60)

    import torch
    from src.models.vision_encoder import DermatoVisionEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DermatoVisionEncoder.load_checkpoint(
        str(PROJECT_ROOT / "models" / "best_model.pt"),
        device=device,
    )
    model.eval()

    test_csv = str(PROJECT_ROOT / "data" / "processed" / "test_metadata.csv")

    # No-TTA
    logger.info("\n→ No-TTA baseline çalıştırılıyor...")
    no_tta = benchmark_vision_classification(
        model, test_csv, device, use_tta=False,
        max_samples=max_samples, stratified=True,
    )

    # TTA
    logger.info("\n→ TTA-6 çalıştırılıyor...")
    tta = benchmark_vision_classification(
        model, test_csv, device, use_tta=True,
        max_samples=max_samples, stratified=True,
    )

    return {"no_tta": no_tta, "tta": tta}


# =========================================================
# Deney 2: Retrieval comparison
# =========================================================
def run_retrieval_comparison(modes: List[str], n_queries_per_class: int) -> Dict[str, Any]:
    logger.info("\n" + "=" * 60)
    logger.info("DENEY 2: Retrieval Mode Comparison")
    logger.info("=" * 60)

    from src.rag.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(
        persist_dir=str(PROJECT_ROOT / "data" / "embeddings" / "chromadb"),
        embedding_model="pritamdeka/S-PubMedBert-MS-MARCO",
    )
    queries = build_test_queries(per_class=n_queries_per_class)
    logger.info(f"Toplam test sorgusu: {len(queries)} ({n_queries_per_class}/sınıf × {len(CLASS_LABELS)} sınıf)")

    return benchmark_retrieval(kb, queries, modes=modes, top_k=5)


# =========================================================
# Deney 3: RAG vs No-RAG
# =========================================================
def run_rag_vs_norag(n_per_class: int, language: str = "en") -> Dict[str, Any]:
    logger.info("\n" + "=" * 60)
    logger.info(f"DENEY 3: RAG vs No-RAG (LLM çıktıları, {n_per_class} vaka/sınıf)")
    logger.info("=" * 60)
    logger.info("⚠ Bu deney LLM API çağrıları yapar (her vaka 2 çağrı)")

    from src.pipeline import DermatoRAGPipeline
    pipeline = DermatoRAGPipeline(
        llm_provider="gemini",
        llm_model="gemini-2.5-flash",
        retrieval_mode="dense",
    )
    test_csv = str(PROJECT_ROOT / "data" / "processed" / "test_metadata.csv")
    cases = pick_test_cases(test_csv, n_per_class=n_per_class)
    logger.info(f"Test vakaları: {len(cases)}")

    return benchmark_rag_vs_norag(pipeline, cases, language=language)


# =========================================================
# Çıktıları kaydet
# =========================================================
def save_classification_outputs(label: str, result: Dict[str, Any]) -> None:
    """Vision benchmark için CSV + grafikler."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    clf = result["classification"]
    per_class = clf.get("per_class", {})
    rows = []
    for cls_name, m in per_class.items():
        if isinstance(m, dict):
            rows.append({
                "class": cls_name,
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1-score": m.get("f1-score"),
                "support": m.get("support"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / f"classification_report_{label}.csv", index=False)

    # Confusion matrix
    cm = clf.get("confusion_matrix")
    if cm is not None:
        plot_confusion_matrix(
            np.asarray(cm), CLASS_LABELS,
            out_path=str(FIGURES / f"confusion_matrix_{label}.png"),
            title=f"Confusion Matrix — {label.upper()}",
            normalize=True,
        )
        # Per-class metrics
        plot_per_class_metrics(
            per_class,
            out_path=str(FIGURES / f"per_class_metrics_{label}.png"),
            title=f"Per-Class Metrics — {label.upper()}",
        )

    # Top-k tablosu
    topk_df = pd.DataFrame([
        {"k": k, "accuracy": v} for k, v in result["topk"].items()
    ])
    topk_df.to_csv(OUTPUTS / f"topk_accuracy_{label}.csv", index=False)
    plot_topk_curve(
        result["topk"],
        out_path=str(FIGURES / f"topk_curve_{label}.png"),
        title=f"Top-k Accuracy — {label.upper()}",
    )


def save_retrieval_outputs(retrieval_results: Dict[str, Dict[str, float]]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(retrieval_results).T
    df.to_csv(OUTPUTS / "retrieval_comparison.csv")
    for metric in ("map", "ndcg@k", "diversity", "category_purity"):
        plot_retrieval_comparison(
            retrieval_results, metric,
            out_path=str(FIGURES / f"retrieval_{metric.replace('@','_at_').replace(' ','_')}.png"),
            title=f"Retrieval Comparison — {metric}",
        )


def save_rag_vs_norag_outputs(rvn: Dict[str, Any]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rag = rvn.get("rag", {})
    norag = rvn.get("norag", {})
    df = pd.DataFrame([
        {"mode": "RAG",   **rag},
        {"mode": "NoRAG", **norag},
    ])
    df.to_csv(OUTPUTS / "rag_vs_norag.csv", index=False)
    plot_rag_vs_norag(
        rag, norag,
        metric_names=["faithfulness", "citation_count", "diversity", "category_purity"],
        out_path=str(FIGURES / "rag_vs_norag.png"),
        title="RAG vs No-RAG (literatür entegrasyonunun etkisi)",
    )


def save_diversity_outputs(retrieval_results: Dict[str, Any]) -> None:
    """Sınıf bazlı diversity (dense mod için ekstra grafik)."""
    # Bu opsiyonel — retrieval_results yapısına bağlı. Atla.
    pass


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Hızlı mod (azaltılmış örneklem, LLM atlanır)")
    parser.add_argument("--full", action="store_true", help="Tam mod (tüm test seti + LLM dahil)")
    parser.add_argument(
        "--experiments", nargs="+",
        choices=["vision", "retrieval", "rag_vs_norag"],
        default=None,
        help="Sadece belirli deneyleri çalıştır",
    )
    parser.add_argument("--vision-samples", type=int, default=None,
                        help="Vision benchmark için max örnek (None=tüm test seti)")
    parser.add_argument("--rag-cases-per-class", type=int, default=2,
                        help="RAG vs No-RAG için sınıf başına vaka")
    parser.add_argument("--retrieval-modes", nargs="+",
                        default=["dense", "hybrid", "hybrid+rerank"])
    parser.add_argument("--retrieval-queries-per-class", type=int, default=2)
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Mod ayarları
    if args.quick:
        vision_max = args.vision_samples or 90  # ~10/sınıf
        rag_n = 1
        do_llm = False
    elif args.full:
        vision_max = args.vision_samples or 0  # 0 = tümü
        rag_n = args.rag_cases_per_class
        do_llm = True
    else:
        # Varsayılan: orta yol
        vision_max = args.vision_samples or 270  # ~30/sınıf
        rag_n = args.rag_cases_per_class
        do_llm = True

    # Seçili deneyler
    selected = args.experiments or ["vision", "retrieval", "rag_vs_norag"]

    all_results: Dict[str, Any] = {
        "config": {
            "vision_max": vision_max,
            "rag_n": rag_n,
            "do_llm": do_llm,
            "selected": selected,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    t0 = time.time()

    # --- Deney 1 ---
    if "vision" in selected:
        try:
            vis = run_vision_classification(max_samples=vision_max or 0)
            for label, result in vis.items():
                # JSON'a yazarken probs/cm ağır — clf metrikleri ayır
                save_classification_outputs(label, result)
                # Boyutu küçültmek için JSON'a probs kaydetme
                slim = {**result}
                slim.pop("probs", None)
                slim["classification"] = {**result["classification"]}
                cm = slim["classification"].pop("confusion_matrix", None)
                slim["classification"]["confusion_matrix"] = cm.tolist() if cm is not None else None
                all_results.setdefault("vision", {})[label] = slim
        except Exception as e:
            logger.exception(f"Vision deneyi hata: {e}")
            all_results["vision_error"] = str(e)

    # --- Deney 2 ---
    if "retrieval" in selected:
        try:
            retr = run_retrieval_comparison(
                modes=args.retrieval_modes,
                n_queries_per_class=args.retrieval_queries_per_class,
            )
            save_retrieval_outputs(retr)
            all_results["retrieval"] = retr
        except Exception as e:
            logger.exception(f"Retrieval deneyi hata: {e}")
            all_results["retrieval_error"] = str(e)

    # --- Deney 3 ---
    if "rag_vs_norag" in selected and do_llm:
        try:
            rvn = run_rag_vs_norag(n_per_class=rag_n)
            save_rag_vs_norag_outputs(rvn)
            all_results["rag_vs_norag"] = rvn
        except Exception as e:
            logger.exception(f"RAG vs No-RAG deneyi hata: {e}")
            all_results["rag_vs_norag_error"] = str(e)
    elif "rag_vs_norag" in selected:
        logger.info("→ RAG vs No-RAG quick modda atlandı (LLM API çağrısı pahalı)")

    all_results["elapsed_sec"] = time.time() - t0

    # JSON kaydet
    out_json = OUTPUTS / "experiment_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(all_results), f, indent=2, ensure_ascii=False)
    logger.info(f"\n✓ Tüm sonuçlar yazıldı: {out_json}")
    logger.info(f"  Grafikler: {FIGURES}")
    logger.info(f"  Toplam süre: {all_results['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
