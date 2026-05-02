# 🏥 Dermato-RAG

> **Dermatoloji Multimodal Görüntü Analizi ve Tıbbi Literatür Entegrasyonu ile Tanı Destek Sistemi**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📋 Proje Hakkında

Dermato-RAG, dermatolojik görüntüleri analiz eden ve tıbbi literatürü (PubMed, dermatoloji kitapları, kılavuzlar) entegre ederek **kanıta dayalı tanı önerileri** sunan bir yapay zeka sistemidir.

Sistem, **Retrieval-Augmented Generation (RAG)** mimarisini **multimodal (görüntü + metin)** yeteneklerle birleştirerek çalışır:

- 🖼️ **Görüntü Analizi**: BiomedCLIP/ViT ile dermatolojik lezyon analizi
- 📚 **Literatür Entegrasyonu**: PubMed makaleleri ve kılavuzlardan kanıt retrieval
- 🧠 **Akıllı Tanı**: LLM tabanlı ayırıcı tanı ve açıklama üretimi
- 📊 **Güven Skoru**: Her tanı için güvenilirlik değerlendirmesi

> ⚠️ **Uyarı**: Bu bir tanı destek sistemidir, kesin tanı koymaz. Yalnızca araştırma amaçlıdır.

---

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.10 veya üstü
- GPU (önerilen: NVIDIA, 16GB+ VRAM)
- Git

### Kurulum

```bash
# 1. Repository'yi klonlayın
git clone https://github.com/talha/dermato-rag.git
cd dermato-rag

# 2. Sanal ortam oluşturun
python -m venv venv

# 3. Sanal ortamı aktifleştirin
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 5. Geliştirme araçlarını yükleyin (opsiyonel)
pip install -e ".[dev]"

# 6. Ortam değişkenlerini ayarlayın
copy .env.example .env
# .env dosyasını düzenleyip API anahtarlarınızı girin
```

### Testleri Çalıştırma

```bash
pytest tests/ -v
```

---

## 📁 Proje Yapısı

```
Dermato-RAG/
├── configs/                    # Konfigürasyon dosyaları
│   ├── config.yaml            # Ana konfigürasyon
│   ├── model_config.yaml      # Model ayarları
│   └── data_config.yaml       # Veri ayarları
├── data/                       # Veri dizini (git'e eklenmez)
│   ├── raw/                   # Ham veriler
│   ├── processed/             # İşlenmiş veriler
│   ├── knowledge_base/        # RAG bilgi tabanı
│   └── embeddings/            # Embedding'ler
├── src/                        # Ana kaynak kodu
│   ├── data/                  # Veri yükleme ve işleme
│   ├── models/                # Model tanımları
│   ├── rag/                   # RAG pipeline
│   ├── llm/                   # LLM entegrasyonu
│   ├── evaluation/            # Değerlendirme
│   └── utils/                 # Yardımcı fonksiyonlar
├── app/                        # Web arayüzü
├── tests/                      # Birim testleri
├── scripts/                    # Yardımcı scriptler
├── notebooks/                  # Jupyter notebook'lar
└── docs/                       # Dokümantasyon
```

---

## 🛠️ Teknoloji Yığını

| Kategori | Teknoloji |
|----------|-----------|
| **Derin Öğrenme** | PyTorch, HuggingFace Transformers |
| **Vision** | BiomedCLIP, ViT, timm |
| **NLP / Embedding** | PubMedBERT, sentence-transformers |
| **RAG** | LangChain, ChromaDB, FAISS |
| **LLM** | GPT-4o, Gemini Pro |
| **Değerlendirme** | scikit-learn, RAGAS |
| **Arayüz** | Gradio |

---

## 📝 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

## 📧 İletişim

Sorularınız için issue açabilir veya doğrudan iletişime geçebilirsiniz.
