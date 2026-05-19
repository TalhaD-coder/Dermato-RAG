"""
Dermato-RAG — Benchmark / Ablation Study Modülü (Faz 7).

Karşılaştırma deneylerini orkestrer:

    1. Vision-only baseline: BiomedCLIP fine-tuned (TTA on/off)
    2. RAG vs No-RAG (literatür entegrasyonu LLM çıktısını ne kadar etkiliyor?)
    3. Retrieval modu karşılaştırması: dense vs hybrid vs hybrid+reranker
    4. Top-k optimizasyonu: k ∈ {1, 3, 5, 10}

Bu modül `run_experiments.py` tarafından çağrılır.

Tasarım:
    - Sınıflandırma deneyleri test setinin tamamını (ya da örneklem) kullanır.
    - RAG/LLM deneyleri pahalıdır (LLM çağrısı), bu yüzden örneklem (default 30 vaka/sınıf)
      üzerinde çalışır. `n_per_class` parametresi ile ayarlanır.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from src.evaluation.metrics import (
    article_diversity,
    category_purity,
    citation_count,
    classification_metrics,
    faithfulness_lexical,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    top_k_accuracy,
)
from src.utils.logger import get_logger

logger = get_logger("benchmarks")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Sınıf etiketleri (pipeline ile aynı sıra)
CLASS_LABELS = [
    "actinic_keratosis", "basal_cell_carcinoma", "benign_keratosis",
    "dermatofibroma", "melanoma", "nevus", "seborrheic_keratosis",
    "squamous_cell_carcinoma", "vascular_lesion",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LABELS)}


# =========================================================
# 1. Vision sınıflandırma benchmark (TTA on/off)
# =========================================================

def _preprocess_image(model, image_path: str):
    """Vision model için ön işleme."""
    from torchvision import transforms
    image = Image.open(image_path).convert("RGB")
    if hasattr(model, "_preprocess") and model._preprocess:
        prep = model._preprocess
    else:
        prep = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return image, prep


def _predict_one(model, image_path: str, device: str, use_tta: bool) -> np.ndarray:
    """Tek bir görüntü için olasılık vektörü döndürür."""
    image, prep = _preprocess_image(model, image_path)
    if use_tta:
        variants = [
            image, ImageOps.mirror(image), ImageOps.flip(image),
            ImageOps.mirror(ImageOps.flip(image)),
            image.rotate(90), image.rotate(270),
        ]
    else:
        variants = [image]
    probs_acc = []
    with torch.no_grad():
        for img in variants:
            t = prep(img).unsqueeze(0).to(device)
            logits = model(t)
            probs_acc.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
    return np.mean(probs_acc, axis=0)


def benchmark_vision_classification(
    model,
    test_csv: str,
    device: str,
    use_tta: bool = True,
    max_samples: Optional[int] = None,
    stratified: bool = True,
) -> Dict[str, Any]:
    """
    Test seti üzerinde vision modelin sınıflandırma performansı.

    Args:
        model: DermatoVisionEncoder (eval moduna alınmış).
        test_csv: test_metadata.csv path.
        device: "cuda" / "cpu".
        use_tta: 6-augment TTA kullan.
        max_samples: None → tüm test seti; int → max örnek.
        stratified: True ise her sınıftan eşit örnekle.

    Returns:
        {
            "config": {...},
            "n_samples": int,
            "elapsed_sec": float,
            "y_true": List[int], "y_pred": List[int],
            "probs": np.ndarray (N x 9),
            "topk": {1: ..., 3: ..., 5: ...},
            "classification": classification_metrics(...)
        }
    """
    df = pd.read_csv(test_csv)
    df = df[df["label"].isin(CLASS_LABELS)].copy()

    if max_samples is not None and max_samples > 0:
        if stratified:
            per_class = max(1, max_samples // len(CLASS_LABELS))
            sampled = []
            for cls in CLASS_LABELS:
                sub = df[df["label"] == cls]
                if len(sub) == 0:
                    continue
                sampled.append(sub.sample(min(len(sub), per_class), random_state=42))
            df = pd.concat(sampled, ignore_index=True) if sampled else df
        else:
            df = df.sample(min(len(df), max_samples), random_state=42)

    logger.info(f"Vision benchmark — {len(df)} örnek, TTA={use_tta}, device={device}")

    y_true, probs_all = [], []
    t0 = time.time()
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        img_path = str(PROJECT_ROOT / "data" / "processed" / row["relative_path"])
        try:
            probs = _predict_one(model, img_path, device, use_tta)
        except Exception as e:
            logger.warning(f"Atlandı: {img_path} ({e})")
            continue
        y_true.append(CLASS_TO_IDX[row["label"]])
        probs_all.append(probs)
        if idx % 50 == 0:
            logger.info(f"  [{idx}/{len(df)}] elapsed {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    probs_all = np.vstack(probs_all)
    y_true = np.array(y_true)
    y_pred = probs_all.argmax(axis=1)

    topk = {k: top_k_accuracy(y_true, probs_all, k) for k in (1, 2, 3, 5)}
    clf = classification_metrics(
        y_true, y_pred,
        labels=list(range(len(CLASS_LABELS))),
        target_names=CLASS_LABELS,
    )

    return {
        "config": {"use_tta": use_tta, "n_samples": int(len(y_true)), "device": device},
        "n_samples": int(len(y_true)),
        "elapsed_sec": elapsed,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "probs": probs_all,
        "topk": topk,
        "classification": clf,
    }


# =========================================================
# 2. RAG retrieval benchmark (dense vs hybrid vs reranker)
# =========================================================

def benchmark_retrieval(
    kb,
    test_queries: List[Dict[str, Any]],
    modes: List[str] = ("dense", "hybrid", "hybrid+rerank"),
    top_k: int = 5,
    use_category_filter: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Retrieval modlarını MAP, NDCG@k, diversity, purity üzerinde karşılaştır.

    Args:
        kb: KnowledgeBase instance.
        test_queries: [{"query":str, "category":str}, ...]
        modes: Karşılaştırılacak modlar.
        top_k: Retrieval derinliği.
        use_category_filter: True ise ChromaDB filtresi kullanılır
            (purity yapay olarak yükselir; daha akademik bir karşılaştırma
            için False bırakılması önerilir).

    Returns:
        {
          "dense":         {"map": ..., "ndcg@k": ..., "diversity": ..., "purity": ...},
          "hybrid":        {...},
          "hybrid+rerank": {...},
        }
    """
    from src.rag.retriever import HybridRetriever

    retrievers = {}
    if "dense" in modes:
        retrievers["dense"] = ("dense", None)
    if "hybrid" in modes:
        retrievers["hybrid"] = ("hybrid", HybridRetriever(knowledge_base=kb))
    if "hybrid+rerank" in modes:
        from src.rag.reranker import CrossEncoderReranker
        retrievers["hybrid+rerank"] = ("hybrid", HybridRetriever(knowledge_base=kb), CrossEncoderReranker())

    summary = {}
    for mode_name, retr_cfg in retrievers.items():
        logger.info(f"\n--- Retrieval mode: {mode_name} ---")
        per_query_diversity = []
        per_query_purity = []
        map_records = []
        ndcg_scores = []
        # Her kategori için "doğru cevap kümesi" — o kategoriye ait tüm PMID'ler.
        # Bu, kategori filtresi olsa da olmasa da MAP/NDCG için ground-truth.
        relevant_by_category: Dict[str, set] = {}
        for q in test_queries:
            cat = q["category"]
            if cat not in relevant_by_category:
                try:
                    cat_res = kb.collection.get(where={"category": cat}, include=["metadatas"])
                    relevant_by_category[cat] = {
                        str(m.get("pmid", "")) for m in cat_res.get("metadatas", []) if m.get("pmid")
                    }
                except Exception:
                    relevant_by_category[cat] = set()

        for q in test_queries:
            cat = q["category"]
            query = q["query"]
            filt = cat if use_category_filter else None

            # Mode'a göre retrieval
            if mode_name == "dense":
                results = kb.search(query, top_k=top_k * 4, category_filter=filt)
            else:
                hr: HybridRetriever = retr_cfg[1]
                results = hr.retrieve(query, top_k=top_k * 4, category_filter=filt, mode="hybrid")
                if mode_name == "hybrid+rerank":
                    rer = retr_cfg[2]
                    results = rer.rerank(query, results, top_k=top_k * 4)

            # PMID-bazlı dedup (pipeline ile aynı mantık)
            seen = set()
            unique_results = []
            for r in results:
                pmid = str(r.get("metadata", {}).get("pmid", "")).strip()
                key = pmid or r.get("text", "")[:120]
                if not key or key in seen:
                    continue
                seen.add(key)
                unique_results.append(r)
                if len(unique_results) >= top_k:
                    break

            retrieved_pmids = [str(r.get("metadata", {}).get("pmid", "")) for r in unique_results]
            relevant_set = relevant_by_category.get(cat, set())

            # MAP & NDCG: relevant = kategoriye ait gerçek PMID kümesi
            rel_scores = {p: (1.0 if p in relevant_set else 0.0) for p in retrieved_pmids}
            ndcg_scores.append(ndcg_at_k(rel_scores, retrieved_pmids, k=top_k))
            map_records.append({
                "relevant": list(relevant_set),
                "retrieved": retrieved_pmids,
            })

            # Frontend article formatına çevirip diversity/purity
            articles = []
            for r in unique_results:
                m = r.get("metadata", {})
                articles.append({
                    "pmid": str(m.get("pmid", "")),
                    "title": m.get("title", ""),
                    "snippet": r.get("text", ""),
                    "link": f"https://pubmed.ncbi.nlm.nih.gov/{m.get('pmid')}/",
                })
            per_query_diversity.append(article_diversity(articles))
            per_query_purity.append(category_purity(articles, cat))

        summary[mode_name] = {
            "map": mean_average_precision(map_records),
            "ndcg@k": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "diversity": float(np.mean(per_query_diversity)) if per_query_diversity else 0.0,
            "category_purity": float(np.mean(per_query_purity)) if per_query_purity else 0.0,
            "n_queries": len(test_queries),
        }
        logger.info(f"  {mode_name}: {summary[mode_name]}")

    return summary


def build_test_queries(per_class: int = 2) -> List[Dict[str, Any]]:
    """
    Her sınıf için sabit test sorguları üretir.

    Args:
        per_class: Sınıf başına sorgu sayısı.
    """
    templates = [
        "Dermoscopic features and clinical management of {cls}.",
        "Differential diagnosis and treatment guidelines for {cls}.",
        "Histopathology and risk factors of {cls}.",
    ]
    queries = []
    for cls in CLASS_LABELS:
        readable = cls.replace("_", " ")
        for t in templates[:per_class]:
            queries.append({"query": t.format(cls=readable), "category": cls})
    return queries


# =========================================================
# 3. RAG vs No-RAG (LLM çıktısı, küçük örnek üzerinde)
# =========================================================

def benchmark_rag_vs_norag(
    pipeline,
    test_cases: List[Dict[str, Any]],
    language: str = "en",
) -> Dict[str, Dict[str, float]]:
    """
    Aynı görüntü için RAG'lı ve RAG'sız LLM çıktısını üretip karşılaştırır.

    test_cases: [{"image_path":..., "category":..., "clinical_info":...}, ...]

    Returns:
        {
          "rag":   {"faithfulness", "citation_count", "diversity", "purity", "n_articles"},
          "norag": {"faithfulness", "citation_count", ...},
        }
    """
    rag_records = []
    norag_records = []

    for i, case in enumerate(test_cases, 1):
        logger.info(f"[{i}/{len(test_cases)}] RAG vs No-RAG: {case.get('category')}")

        # --- RAG ON ---
        try:
            result = pipeline.analyze(
                image_path=case["image_path"],
                clinical_info=case.get("clinical_info", ""),
                run_faithfulness_check=False,
                language=language,
            )
            articles = result.get("articles", [])
            diagnosis = result.get("diagnosis", "")
            rag_records.append({
                "n_articles": len(articles),
                "diversity": article_diversity(articles),
                "category_purity": category_purity(articles, case["category"]),
                "faithfulness": faithfulness_lexical(diagnosis, articles),
                "citation_count": citation_count(diagnosis),
            })
        except Exception as e:
            logger.warning(f"  RAG analyze hata: {e}")

        # --- RAG OFF (LLM'i direkt çağır, kaynak olmadan) ---
        try:
            # _predict_image kullan
            v = pipeline._predict_image(case["image_path"])
            class_name = v["top_class_display"]
            vision_text = f"{class_name} ({v['confidence']*100:.1f}% confidence)"

            # Dil-tutarlı sabit metinler — sistem promptu zaten language'a göre seçilecek
            is_en = str(language).lower().startswith("en")
            no_info_label = (
                "(no patient information provided)"
                if is_en
                else "(hasta bilgisi verilmedi)"
            )
            features_label = (
                "Vision-only baseline (RAG disabled)"
                if is_en
                else "Yalnız görüntü tahmini (RAG devre dışı)"
            )
            no_rag_diagnosis = pipeline.generator.generate_diagnosis(
                clinical_info=case.get("clinical_info", "") or no_info_label,
                vision_prediction=vision_text,
                vision_features=features_label,
                rag_results=[],  # ← boş
                language=language,
            )
            # Boş RAG için faithfulness mantıksız; sadece kaynak yokken
            # halüsinasyon eğilimini ölçmek için cevabın uzunluğu + citation
            norag_records.append({
                "n_articles": 0,
                "diversity": 0.0,
                "category_purity": 0.0,
                "faithfulness": 0.0,
                "citation_count": citation_count(no_rag_diagnosis),
            })
        except Exception as e:
            logger.warning(f"  No-RAG analyze hata: {e}")

    def _avg(records: List[Dict[str, float]]) -> Dict[str, float]:
        if not records:
            return {}
        keys = records[0].keys()
        return {k: float(np.mean([r[k] for r in records])) for k in keys}

    return {
        "rag": _avg(rag_records),
        "norag": _avg(norag_records),
        "n_cases": len(test_cases),
    }


def pick_test_cases(test_csv: str, n_per_class: int = 3, random_state: int = 42) -> List[Dict[str, Any]]:
    """RAG/LLM benchmark için stratified test örneklemi seçer."""
    df = pd.read_csv(test_csv)
    cases = []
    for cls in CLASS_LABELS:
        sub = df[df["label"] == cls].sample(
            min(n_per_class, len(df[df["label"] == cls])),
            random_state=random_state,
        )
        for _, row in sub.iterrows():
            cases.append({
                "image_path": str(PROJECT_ROOT / "data" / "processed" / row["relative_path"]),
                "category": cls,
                "clinical_info": f"Age: {row.get('age', '?')}, Location: {row.get('localization', 'unknown')}",
            })
    return cases
