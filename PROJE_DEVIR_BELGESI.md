# 🏥 Dermato-RAG — Proje Devir Belgesi

> **Hazırlayan:** Dijle Doğan  
> **Tarih:** 17 Mayıs 2026  
> **GitHub:** https://github.com/TalhaD-coder/Dermato-RAG

---

## 📌 Proje Nedir?

Dermato-RAG, cilt lezyonu fotoğraflarını analiz eden, tıbbi literatürden kanıt toplayan ve yapay zeka (Gemini) ile klinisyen seviyesinde tanı raporu üreten bir sistemdir.

**Teknoloji yığını:**
- **Vision Model:** BiomedCLIP (fine-tuned, %73 doğruluk, 9 sınıf)
- **RAG:** ChromaDB + 706 PubMed makalesi
- **LLM:** Google Gemini 1.5 Pro
- **Dil:** Python 3.14

---

## ✅ Şimdiye Kadar Tamamlanan Fazlar

### Faz 1 — Proje Altyapısı ✅
- Klasör yapısı, `requirements.txt`, loglama, config sistemi kuruldu

### Faz 2 — Veri İşleme ✅
- ISIC + PAD-UFES veri setleri işlendi (27.629 görüntü, 9 sınıf)
- Sınıf dengesizliği tespiti yapıldı, `class_weights.csv` oluşturuldu
- Augmentation pipeline kuruldu (`src/data/augmentation.py`)

### Faz 3 — RAG Pipeline ✅
- PubMed'den 706 makale indirildi (`data/knowledge_base/raw_docs/`)
- ChromaDB vektör veritabanı oluşturuldu (`data/embeddings/chromadb/`)
- RAG retriever ve chunk sistemi yazıldı (`src/rag/`)

### Faz 4 — Vision Model Eğitimi ✅
- BiomedCLIP tabanlı `DermatoVisionEncoder` yazıldı
- Google Colab (T4 GPU) üzerinde 3 aşamalı eğitim yapıldı:
  - Head eğitimi (10 epoch)
  - 12 katman unfreeze (15 epoch)
  - Tam unfreeze (15 epoch)
- **Sonuç: Test Acc = %73.00** — model `best_model.pt` olarak kaydedildi
- Dropout=0.3, AMP, Early Stopping, Class-Weighted Sampler kullanıldı
- Eğitim notebook'u: `notebooks/03_train_vision_colab.ipynb`

### Faz 5 — LLM Entegrasyonu ✅
- `src/llm/prompt_templates.py` — Uzman dermatolog sistem promptu
- `src/llm/generator.py` — Gemini/OpenAI bağlantısı (LangChain)
- `src/llm/confidence.py` — Hallüsinasyon (uydurma) kontrol sistemi
- `src/pipeline.py` — **Ana pipeline** (Görüntü → Vision → RAG → LLM → Rapor)
- `tests/test_llm.py` — 17/17 birim testi geçiyor

---

## 🔜 Sıradaki Adımlar (Devam Edecek Arkadaş İçin)

### Faz 6 — Dashboard / Arayüz
Kullanıcının görüntü yükleyip sonucu görebileceği bir arayüz:
- **Seçenek A:** Streamlit (`pip install streamlit`) — en kolay, 1-2 günde biter
- **Seçenek B:** FastAPI + HTML dashboard — daha profesyonel

### Faz 7 — Gerçek Görüntü Testi
```python
from src.pipeline import DermatoRAGPipeline

pipeline = DermatoRAGPipeline()
result = pipeline.analyze(
    image_path="lesion.jpg",
    clinical_info="45 yaş, sırt bölgesi, 3 aydır büyüyen lezyon"
)
print(result["diagnosis"])
```

### Faz 8 — Deployment (opsiyonel)
Hugging Face Spaces veya Google Cloud Run üzerinde yayınlanabilir.

---

## 💻 Kurulum Rehberi (Arkadaş İçin Adım Adım)

### Adım 1: Kodu GitHub'dan İndir
```bash
git clone https://github.com/TalhaD-coder/Dermato-RAG.git
cd Dermato-RAG
```

### Adım 2: Sanal Ortam Oluştur ve Paketleri Kur
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Adım 3: Dijle'den Alınan Dosyaları Yerleştir
- `best_model.pt` → `models/` klasörüne koy
  ```
  Dermato-RAG/
  └── models/
      └── best_model.pt   ← buraya
  ```

> **Not:** `data/embeddings/chromadb/` zaten sende var, tekrar gönderilmesine gerek yok.

### Adım 4: Kendi API Anahtarını Oluştur

1. **[aistudio.google.com](https://aistudio.google.com)** adresine git
2. Google hesabınla giriş yap
3. **"Get API Key"** → **"Create API Key"**
4. Anahtarı kopyala

### Adım 5: `.env` Dosyasını Oluştur
Proje klasöründe `.env` adında bir dosya oluştur ve şunu yaz:
```
GOOGLE_API_KEY=buraya_kendi_anahtarını_yaz
OPENAI_API_KEY=your_openai_api_key_here
NCBI_EMAIL=talha@dermato-rag.dev
NCBI_API_KEY=5e039c6ab71534f662d91df1ff093702d008
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=dermato-rag
HF_TOKEN=your_huggingface_token_here
```

> ⚠️ `.env` dosyasını asla GitHub'a yükleme!

### Adım 6: Testleri Çalıştır ve Her Şeyin Çalıştığını Doğrula
```bash
python -m pytest tests/test_llm.py -v
```
17 testin hepsi `PASSED` görünmeli.

### Adım 7: Pipeline'ı İlk Kez Test Et
```python
# test_pipeline_quick.py
from src.pipeline import DermatoRAGPipeline

pipeline = DermatoRAGPipeline(
    llm_provider="gemini",
    llm_model="gemini-1.5-pro"
)
print("Pipeline hazır!")
```
```bash
python test_pipeline_quick.py
```

---

## 📁 Klasör Yapısı (Özet)

```
Dermato-RAG/
├── src/
│   ├── models/
│   │   ├── vision_encoder.py   ← BiomedCLIP encoder
│   │   └── trainer.py          ← Eğitim döngüsü
│   ├── llm/
│   │   ├── prompt_templates.py ← Gemini promptları
│   │   ├── generator.py        ← LLM bağlantısı
│   │   └── confidence.py       ← Hallüsinasyon kontrolü
│   ├── rag/                    ← RAG pipeline
│   ├── data/                   ← Veri işleme
│   ├── pipeline.py             ← ⭐ ANA PIPELINE
│   └── utils/
├── models/
│   └── best_model.pt           ← ⭐ Eğitilmiş model (329 MB, GitHub'da değil)
├── data/
│   ├── knowledge_base/         ← 706 PubMed makalesi
│   └── embeddings/chromadb/    ← Vektör veritabanı (GitHub'da değil)
├── notebooks/
│   └── 03_train_vision_colab.ipynb  ← Colab eğitim notebook'u
├── tests/
│   ├── test_llm.py             ← 17/17 geçiyor
│   └── test_models.py
├── .env                        ← KENDİ ANAHTARINI KOY (GitHub'da değil)
├── requirements.txt
└── README.md
```

---

## 🔑 Kritik Bilgiler

| Bilgi | Değer |
|-------|-------|
| Vision Model Doğruluğu | %73.00 (test seti) |
| Veri Seti | 27.629 görüntü, 9 sınıf |
| LLM | Gemini 1.5 Pro (ücretsiz tier) |
| Eğitim Ortamı | Google Colab T4 GPU |
| Dropout | 0.3 + 0.15 (iki katmanlı) |
| Augmentation | Color Jitter, Affine, Blur, RandomFlip |
| Makale Sayısı | 706 (ChromaDB'de vektörize edilmiş) |

---

## ❓ Sık Sorulan Sorular

**S: `best_model.pt` nerede?**  
C: Dijle'den Drive linki üzerinden alacaksın. `models/` klasörüne koy.

**S: ChromaDB'yi de Dijle'den alacak mıyım?**  
C: Hayır. Faz 3'te sen oluşturmuştun, sende zaten var.

**S: Hangi Python versiyonu?**  
C: Python 3.14 (en az 3.10 olmalı)

**S: GPU gerekli mi?**  
C: Eğitim için gerekli (Colab'da yapıldı). Inference (tahmin) için CPU yeterli ama yavaş olur.

**S: Test başarısız olursa?**  
C: `pip install -r requirements.txt` komutunu tekrar çalıştır.
