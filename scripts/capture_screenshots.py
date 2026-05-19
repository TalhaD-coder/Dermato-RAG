"""
Dermato-RAG — Web Arayüzü Ekran Görüntüsü Yakalama Scripti.

Playwright (chromium headless) kullanarak çalışan FastAPI sunucusundan
README için ekran görüntüleri toplar.

Önkoşul:
    1. `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000` çalışıyor olmalı
    2. `pip install playwright && playwright install chromium`

Kullanım:
    python scripts/capture_screenshots.py
    python scripts/capture_screenshots.py --base http://localhost:8000

Çıktı:
    docs/images/01_landing.png            — Açılış ekranı
    docs/images/02_upload.png             — Görüntü yükleme adımı
    docs/images/03_clinical_form.png      — Hasta bilgisi formu
    docs/images/04_loader.png             — Analiz yükleme ekranı
    docs/images/05_results_bento.png      — Sonuç bento grid
    docs/images/06_report_card.png        — Detaylı tanı raporu kartı
    docs/images/07_history.png            — Hasta geçmişi paneli
    docs/images/08_settings.png           — Ayarlar paneli
    docs/images/09_landing_en.png         — Açılış ekranı (EN)
    docs/images/10_landing_light.png      — Açık tema açılış
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "docs" / "images"
SAMPLE_IMAGE = PROJECT_ROOT / "app" / "static" / "samples" / "bcc.jpg"


def take_screenshots(base_url: str) -> None:
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1.5,
        )
        page = ctx.new_page()

        # Tour ve diğer ilk-açılış davranışlarını bastır (her sayfa yüklenmesinde)
        prep_js = """
            localStorage.setItem('dermato_tour_done','1');
            localStorage.setItem('dermato_lang','tr');
            localStorage.setItem('dermato_theme','dark');
            sessionStorage.removeItem('dermato_launched');
        """
        page.add_init_script(prep_js)

        def wait_settle(ms: int = 1100) -> None:
            page.wait_for_timeout(ms)

        # ---------- 1. LANDING (TR) ----------
        print("[1/10] Landing TR ...")
        page.goto(base_url, wait_until="networkidle")
        page.wait_for_selector("#landing-view.active", state="attached")
        # Tour overlay'i emin olmak için kapat
        page.evaluate("document.getElementById('onboarding-overlay')?.classList.add('hidden')")
        wait_settle()
        page.screenshot(path=str(OUT_DIR / "01_landing.png"), full_page=False)

        # ---------- 9. LANDING (EN) ----------
        print("[9/10] Landing EN ...")
        page.evaluate("setLanguage('en', false)")
        wait_settle(600)
        page.screenshot(path=str(OUT_DIR / "09_landing_en.png"), full_page=False)

        # ---------- 10. LANDING (light theme) ----------
        print("[10/10] Landing Light ...")
        page.evaluate("setLanguage('tr', false); applyTheme('light')")
        wait_settle(700)
        page.screenshot(path=str(OUT_DIR / "10_landing_light.png"), full_page=False)

        # Tekrar dark + TR'ye dön
        page.evaluate("applyTheme('dark'); setLanguage('tr', false)")
        wait_settle(400)

        # ---------- App'e gir ----------
        # launchApp() — global olarak tanımlı
        page.evaluate("launchApp()")
        page.wait_for_selector("#app-view.active", state="attached")
        page.wait_for_timeout(900)

        # ---------- 2. UPLOAD ----------
        print("[2/10] Upload (step-1) ...")
        page.evaluate("goToStep(1)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT_DIR / "02_upload.png"), full_page=False)

        # Sample image ile dosyayı yükle
        if SAMPLE_IMAGE.exists():
            page.set_input_files("#file-input", str(SAMPLE_IMAGE))
            page.wait_for_timeout(600)

        # ---------- 3. CLINICAL FORM (step-2) ----------
        print("[3/10] Clinical form (step-2) ...")
        page.evaluate("goToStep(2)")
        page.wait_for_timeout(700)
        page.fill("#inp-age", "70")
        page.select_option("#inp-gender", value="Kadın")
        page.fill("#inp-loc", "Burun ucu")
        page.fill("#inp-dur", "6 aydır yavaşça büyüyen")
        page.fill("#inp-symp", "İnci tanesi gibi parlak, ortası çukur")
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT_DIR / "03_clinical_form.png"), full_page=False)

        # ---------- 4. LOADER ----------
        print("[4/10] Loader ...")
        # Loader'ı simüle et — gerçek analyze API çağrısı LLM kotasına yüklenecek; sadece UI göster.
        page.evaluate("""
            (async () => {
                await showView('analysis');
                await goToStep(3);
                document.getElementById('loader-view').classList.remove('hidden');
                document.getElementById('results-view').classList.add('hidden');
                startLoadingTimer();
                const tv = document.getElementById('t-vision');
                const tr = document.getElementById('t-rag');
                const tl = document.getElementById('t-llm');
                if (tv) { tv.className='task active'; tv.querySelector('.t-icon').innerText='⏳'; }
                if (tr) { tr.className='task active'; tr.querySelector('.t-icon').innerText='⏳'; }
                if (tl) { tl.className='task active'; tl.querySelector('.t-icon').innerText='⏳'; }
            })();
        """)
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT_DIR / "04_loader.png"), full_page=False)

        # ---------- 5/6. RESULTS — mock data ile ----------
        print("[5/10] Results bento + [6/10] Report card ...")
        mock_diagnosis_md = (
            "1. LEZYON ANALİZİ:\n\n"
            "Görüntü modeli, lezyonu **Bazal Hücreli Karsinom (BHK)** olarak %88.9 güvenle sınıflandırdı. "
            "70 yaşındaki hastada, güneşe maruz kalmış burun yan kısmında, 6 aydır yavaşça büyüyen "
            "inci tanesi parlaklığında, ortası hafif çukur bir nodül tarif edilmiştir. "
            "Bu klinik görünüm BHK için klasik bulgudur ve modelin tahmiyle uyumludur. [Kaynak 1]\n\n"
            "2. AYIRİCİ TANI LİSTESİ:\n\n"
            "- **Tanı 1: Bazal Hücreli Karsinom (BHK) — güneşe bağlı, en sık görülen cilt kanseri** (Güven Oranı: %88.9)\n"
            "  * Destekleyen bulgular: 70 yaşında, burun bölgesi, yavaş büyüme, inci parlaklığı.\n"
            "  * Risk seviyesi: Malign (düşük metastaz, yüksek lokal hasar)\n"
            "  * Literatür: [Kaynak 1], [Kaynak 2]\n"
            "- **Tanı 2: Skuamöz Hücreli Karsinom (SHK) — kabuklu, sert nodül** (Güven Oranı: %5.0)\n"
            "  * Destekleyen bulgular: yaş, güneş hasarı.\n"
            "  * Zayıflatan faktörler: Tipik kabuklanma yok.\n"
            "- **Tanı 3: Aktinik Keratoz — güneş hasarına bağlı pre-kanseröz lezyon** (Güven Oranı: %3.1)\n\n"
            "3. HALK DİLİNDE AÇIKLAMA:\n\n"
            "Yıllar içinde güneşin etkisiyle oluşan, çoğunlukla yavaş büyüyen bir cilt kanseridir. "
            "Korkutucu olsa da neredeyse her zaman tedavi edilebilir; başka organlara yayılma ihtimali çok düşüktür. "
            "Erken müdahale ile küçük bir cerrahi işlemle tamamen çıkarılabilir.\n\n"
            "4. İLAÇ, TETİKLEYİCİ FAKTÖRLER VE KORUNMA:\n\n"
            "- UV maruziyeti birincil risk faktörüdür.\n"
            "- Bağışıklık baskılayıcı ilaçlar riski artırır.\n"
            "- SPF 50+ koruyucu, geniş kenarlı şapka önerilir.\n\n"
            "5. KLİNİK YÖNETİM VE SONRAKI ADIMLAR:\n\n"
            "- Birincil: Dermoskopi + biyopsi (shave veya punch).\n"
            "- Tedavi: Eksizyonel cerrahi veya Mohs cerrahisi (yüz bölgesi için).\n"
            "- Takip: 3-6 ay aralıkla cilt taraması.\n\n"
            "6. DOKTORA YÖNLENDİRME VE ACİL DURUMLAR:\n\n"
            "Bir dermatoloji uzmanına gecikmeden başvurun. Lezyonda hızlı büyüme, kanama veya ülserasyon "
            "olursa daha acil değerlendirme gerekir. 🩺\n\n"
            "⚠️ YZ SİSTEM UYARISI:\nBu rapor BiomedCLIP + Gemini 2.5 Flash tabanlı yapay zeka destek "
            "aracı tarafından üretilmiştir. Kesin tanı için dermatoloji uzmanına başvurun."
        )
        mock_data = {
            "vision": {
                "top_class": "basal_cell_carcinoma",
                "top_class_display": "Bazal Hücreli Karsinom",
                "confidence": 88.9,
                "is_malignant": True,
                "is_ood": False,
                "top3": [
                    {"class": "basal_cell_carcinoma", "display": "BHK", "probability": 88.9},
                    {"class": "squamous_cell_carcinoma", "display": "SHK", "probability": 5.0},
                    {"class": "actinic_keratosis", "display": "AK", "probability": 3.1},
                ],
                "all_probs": [
                    {"class": "basal_cell_carcinoma", "probability": 88.9, "is_malignant": True},
                    {"class": "squamous_cell_carcinoma", "probability": 5.0, "is_malignant": True},
                    {"class": "actinic_keratosis", "probability": 3.1, "is_malignant": False},
                    {"class": "melanoma", "probability": 1.2, "is_malignant": True},
                    {"class": "nevus", "probability": 0.9, "is_malignant": False},
                    {"class": "benign_keratosis", "probability": 0.5, "is_malignant": False},
                    {"class": "dermatofibroma", "probability": 0.2, "is_malignant": False},
                    {"class": "seborrheic_keratosis", "probability": 0.1, "is_malignant": False},
                    {"class": "vascular_lesion", "probability": 0.1, "is_malignant": False},
                ],
            },
            "diagnosis": mock_diagnosis_md,
            "articles": [
                {"title": "Dermoscopy of basal cell carcinoma.",
                 "journal": "Dermatologic Clinics", "year": "2018",
                 "link": "https://pubmed.ncbi.nlm.nih.gov/29341291/",
                 "snippet": "Basal cell carcinoma (BCC) is the most common cancer in humans, with characteristic dermoscopic features including arborizing vessels, ulceration, and blue-gray ovoid nests..."},
                {"title": "Dermoscopic features in different morphologic types of BCC.",
                 "journal": "Journal of the American Academy of Dermatology", "year": "2014",
                 "link": "https://pubmed.ncbi.nlm.nih.gov/25111343/",
                 "snippet": "We analyzed the dermoscopic features of 609 histologically proven basal cell carcinomas to define subtype-specific patterns..."},
                {"title": "Dermoscopic features of BCC and its subtypes: a systematic review.",
                 "journal": "JEADV", "year": "2019",
                 "link": "https://pubmed.ncbi.nlm.nih.gov/31706938/",
                 "snippet": "Systematic review of 73 studies confirms that arborizing vessels, shiny white structures, and blue-gray globules are the strongest dermoscopic predictors..."},
                {"title": "Periungual basal cell carcinoma.",
                 "journal": "JAAD Case Reports", "year": "2018",
                 "link": "https://pubmed.ncbi.nlm.nih.gov/29641710/",
                 "snippet": "Rare presentation of BCC in the periungual region; reflectance confocal microscopy aids diagnosis..."},
                {"title": "Dermoscopic findings in presurgical evaluation of BCC.",
                 "journal": "Lasers in Medical Science", "year": "2020",
                 "link": "https://pubmed.ncbi.nlm.nih.gov/32804889/",
                 "snippet": "Pre-Mohs dermoscopic assessment improves margin delineation and reduces re-excision rates..."},
            ],
            "article_count": 5,
        }
        # Mocked render
        import json as _json
        page.evaluate(f"""
            (() => {{
                const mock = {_json.dumps(mock_data)};
                // image preview to results
                const url = currentFileDataUrl || '/static/samples/bcc.jpg';
                document.getElementById('loader-view').classList.add('hidden');
                stopLoadingTimer && stopLoadingTimer();
                renderResults(mock, url);
            }})();
        """)
        page.wait_for_timeout(1400)
        # 5. Bento full
        page.screenshot(path=str(OUT_DIR / "05_results_bento.png"), full_page=False)

        # 6. Sadece report card — scroll'a getir
        try:
            page.eval_on_selector(".bento-card.full-report", "el => el.scrollIntoView({behavior:'instant', block:'start'})")
            page.wait_for_timeout(400)
            box = page.locator(".bento-card.full-report").bounding_box()
            if box:
                page.screenshot(
                    path=str(OUT_DIR / "06_report_card.png"),
                    clip={
                        "x": max(0, box["x"] - 10),
                        "y": max(0, box["y"] - 10),
                        "width": min(1440 - box["x"] + 10, box["width"] + 20),
                        "height": min(900, box["height"] + 20),
                    },
                )
            else:
                page.screenshot(path=str(OUT_DIR / "06_report_card.png"))
        except Exception as e:
            print("  report card crop fallback:", e)
            page.screenshot(path=str(OUT_DIR / "06_report_card.png"))

        # ---------- 7. HISTORY ----------
        print("[7/10] History ...")
        # Sahte history ekle ki dolu görünsün
        page.evaluate("""
            const now = Date.now();
            const fake = [
                { id: now,     timestamp: new Date().toISOString(), imageName: 'bcc.jpg',
                  imageDataUrl: null, lang: 'tr',
                  vision: { top_class:'basal_cell_carcinoma', top_class_display:'Bazal Hücreli Karsinom',
                            confidence:88.9, is_malignant:true, is_ood:false,
                            top3:[], all_probs:[] },
                  diagnosis: 'Mock', articles: [] },
                { id: now-1, timestamp: new Date(Date.now()-86400000).toISOString(), imageName: 'nevus.jpg',
                  imageDataUrl: null, lang: 'tr',
                  vision: { top_class:'nevus', top_class_display:'Nevus (Ben)',
                            confidence:83.4, is_malignant:false, is_ood:false,
                            top3:[], all_probs:[] },
                  diagnosis: 'Mock', articles: [] },
                { id: now-2, timestamp: new Date(Date.now()-3*86400000).toISOString(), imageName: 'melanoma.jpg',
                  imageDataUrl: null, lang: 'tr',
                  vision: { top_class:'melanoma', top_class_display:'Melanom',
                            confidence:77.8, is_malignant:true, is_ood:false,
                            top3:[], all_probs:[] },
                  diagnosis: 'Mock', articles: [] },
            ];
            analysisHistory = fake;
            localStorage.setItem('dermato_history', JSON.stringify(fake));
            updateHistoryBadge();
            showView('history');
            renderHistory();
        """)
        page.wait_for_timeout(900)
        page.screenshot(path=str(OUT_DIR / "07_history.png"), full_page=False)

        # ---------- 8. SETTINGS ----------
        print("[8/10] Settings ...")
        page.evaluate("showView('settings'); renderSettings();")
        page.wait_for_timeout(700)
        page.screenshot(path=str(OUT_DIR / "08_settings.png"), full_page=False)

        # ---------- 11. ONBOARDING TOUR ----------
        print("[11/11] Onboarding tour ...")
        # Temiz başlatma: landing'e geri dön
        page.goto(base_url, wait_until="networkidle")
        page.wait_for_selector("#landing-view.active", state="attached")
        page.wait_for_timeout(500)
        page.evaluate("startOnboarding(true)")
        page.wait_for_selector("#onboarding-overlay:not(.hidden)", state="attached")
        page.wait_for_timeout(700)
        page.screenshot(path=str(OUT_DIR / "11_tour.png"), full_page=False)

        browser.close()
    print(f"\n✓ Tüm ekran görüntüleri kaydedildi → {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    args = parser.parse_args()
    t0 = time.time()
    take_screenshots(args.base)
    print(f"Toplam süre: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
