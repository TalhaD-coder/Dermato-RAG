"""
Dermato-RAG FastAPI Backend.

Endpoints:
    GET  /          → Arayüzü serve eder
    GET  /health    → Sistem durumu
    POST /analyze   → Görüntü analizi
"""

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Proje kökünü ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# Global pipeline
_pipeline = None
_pipeline_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken pipeline'ı yükle."""
    global _pipeline, _pipeline_error
    try:
        print("Pipeline yukleniyor, lutfen bekleyin...")
        from src.pipeline import DermatoRAGPipeline
        _pipeline = DermatoRAGPipeline(
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
        )
        print("Pipeline hazir!")
    except Exception as e:
        _pipeline_error = str(e)
        print(f"Pipeline yuklenemedi: {e}")
    yield
    print("Uygulama kapatiliyor...")


app = FastAPI(
    title="Dermato-RAG API",
    description="Dermatolojik görüntü analizi ve tıbbi literatür entegrasyonu",
    version="1.0.0",
    lifespan=lifespan,
)

# Static dosyalar
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Ana sayfayı serve eder."""
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html bulunamadı")
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    """Sistem durumunu döndürür."""
    model_path = PROJECT_ROOT / "models" / "best_model.pt"
    chromadb_path = PROJECT_ROOT / "data" / "embeddings" / "chromadb"
    gemini_key = os.getenv("GOOGLE_API_KEY", "")

    return {
        "status": "ok" if _pipeline is not None else "loading",
        "pipeline_ready": _pipeline is not None,
        "pipeline_error": _pipeline_error,
        "model_exists": model_path.exists(),
        "chromadb_exists": chromadb_path.exists(),
        "gemini_configured": bool(gemini_key and "your_" not in gemini_key),
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    clinical_info: str = Form(""),
    language: str = Form("en"),
):
    """
    Dermatolojik görüntüyü analiz eder.

    Args:
        image: Yüklenen görüntü dosyası (JPEG/PNG)
        clinical_info: Hasta klinik bilgileri
        language: Rapor dili ("en" veya "tr")

    Returns:
        JSON: Analiz sonuçları
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline henüz hazır değil. Hata: {_pipeline_error or 'Yükleniyor...'}",
        )

    # Dosya türü kontrolü
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya türü: {image.content_type}. JPEG/PNG kullanın.",
        )

    # Geçici dosyaya yaz
    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        # Pipeline'ı çalıştır — dil prompt template seviyesinde belirlenir,
        # clinical_info'yu olduğu gibi geçir (artık TR/EN prefix gerekmiyor).
        result = _pipeline.analyze(
            image_path=tmp_path,
            clinical_info=clinical_info,
            run_faithfulness_check=False,
            language=language,
        )

        # Sonuçları formatla
        vision = result["vision"]
        articles = result.get("articles", [])
        diagnosis = result.get("diagnosis", "")

        # Tüm sınıf skorları (frontend için)
        all_probs_sorted = sorted(
            vision["all_probs"].items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Makale meta verilerini temizle (field names match frontend renderLit)
        clean_articles = []
        for a in articles[:5]:
            clean_articles.append({
                "title": a.get("title", "Unknown Title")[:120],
                "journal": a.get("journal", ""),
                "year": a.get("year", ""),
                "link": a.get("link", "#"),
                "snippet": a.get("snippet", ""),
            })

        # Malign sınıflar
        MALIGNANT = {"melanoma", "basal_cell_carcinoma", "squamous_cell_carcinoma"}
        CLASS_DISPLAY_NAMES = {
            "melanoma": "Melanom", "basal_cell_carcinoma": "Bazal Hücreli Karsinom",
            "squamous_cell_carcinoma": "Skuamöz Hücreli Karsinom", "actinic_keratosis": "Aktinik Keratoz",
            "benign_keratosis": "Benign Keratoz", "dermatofibroma": "Dermatofibrom",
            "nevus": "Nevus (Ben)", "seborrheic_keratosis": "Seboreik Keratoz",
            "vascular_lesion": "Vasküler Lezyon",
        }
        is_malignant = vision["top_class"] in MALIGNANT
        is_ood_flag = bool(vision.get("is_ood", False))

        # OOD durumunda yanıltıcı tanı adını ve güven değerini gizle
        if is_ood_flag:
            top_display = "Tanı Verilemedi" if language == "tr" else "Diagnosis Unavailable"
            confidence_to_send = 0.0
            is_malignant = False  # OOD ise malign de gösterme
        else:
            top_display = vision["top_class_display"]
            confidence_to_send = round(vision["confidence"] * 100, 1)

        return JSONResponse({
            "success": True,
            "vision": {
                "top_class": vision["top_class"],
                "top_class_display": top_display,
                "confidence": confidence_to_send,
                "is_malignant": is_malignant,
                "is_ood": is_ood_flag,
                "top3": [
                    {
                        "class": cls,
                        "display": CLASS_DISPLAY_NAMES.get(cls, cls),
                        "probability": round(prob * 100, 1),
                    }
                    for cls, prob in vision["top3"]
                ],
                "all_probs": [
                    {
                        "class": cls,
                        "probability": round(prob * 100, 1),
                        "is_malignant": cls in MALIGNANT,
                    }
                    for cls, prob in all_probs_sorted
                ],
            },
            "diagnosis": diagnosis,
            "articles": clean_articles,
            "article_count": len(articles),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")

    finally:
        # Geçici dosyayı sil
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
