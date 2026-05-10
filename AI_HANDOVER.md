# 🤖 Antigravity AI (ve Geliştirici) İçin Devir Teslim & Durum Rehberi

Merhaba! Bu belge, **Dermato-RAG** projesinde şu ana kadar nelerin yapıldığını ve bundan sonraki hedeflerin neler olduğunu detaylandırmak için hazırlanmıştır. 

Yeni geliştirici ve ona eşlik edecek olan Antigravity AI, bu dosyayı okuyarak projeye kaldığı yerden, bağlamı hiç kaybetmeden sorunsuzca devam edebilir.

---

## 🎯 Projenin Amacı
Dermatolojik görüntüleri (lezyon fotoğraflarını) analiz eden ve tıbbi literatürü (PubMed makaleleri vb.) tarayarak (RAG Mimarisi ile) **kanıta dayalı ayırıcı tanı önerileri** sunan, akademik yayın hedefli, multimodal (görüntü + metin) bir tanı destek sistemi geliştirmek.

---

## ✅ ŞU ANA KADAR YAPILANLAR (Faz 1 & Faz 2)

Projenin altyapısı ve görüntü veri seti işleme kısımları %100 oranında tamamlanmıştır.

### 1. Altyapı ve Mimarisi (Faz 1 - Tamamlandı)
- Projenin modüler dosya yapısı kuruldu (`src/data`, `src/models`, `src/rag` vb.).
- `pyproject.toml` ve `requirements.txt` ile bağımlılıklar tanımlandı.
- `configs/` altında YAML tabanlı konfigürasyon sistemi kuruldu.
- `src/utils/logger.py` ile kapsamlı bir loglama altyapısı kuruldu.
- Temel unit testler yazıldı (`tests/test_utils.py`).

### 2. Veri Toplama ve Ön İşleme (Faz 2 - %90 Tamamlandı)
- **Kullanılan Veri Setleri:** ISIC 2019 (25.331 görüntü) ve PAD-UFES-20 (2.298 görüntü). Ayrıca Fitzpatrick17k metadata olarak projeye dahil.
- **Veri İşleme (`src/data/preprocessing.py`):** Görüntüler 224x224 formatında yeniden boyutlandırıldı. İki farklı veri setinin (ISIC ve PAD-UFES) etiketleri birleştirildi ve 9 sınıflı **birleşik bir veri seti** oluşturuldu (Toplam 27.629 görüntü).
- **Veri Dağılımı (Split):** Veriler `%70 Train, %15 Val, %15 Test` olacak şekilde dengeli (stratified) olarak bölündü.
- **Sınıf Dengesizliği Çözümü:** PyTorch `WeightedRandomSampler` için sınıf ağırlıkları (`class_weights.csv`) hesaplandı.
- **PyTorch Dataset (`src/data/dataset.py`):** Verileri modele beslemek için özel Dataset ve DataLoader sınıfları kodlandı.
- **Veri Zenginleştirme / Augmentation (`src/data/augmentation.py`):** Tıbbi görüntülere özel; saç artefaktı simülasyonu, mikroskop çerçevesi (vignette) gibi ileri seviye ve bilimsel makalelere dayanan (Perez et al.) veri artırma (augmentation) teknikleri kodlandı.
- **Test ve Doğrulama (`scripts/validate_data.py`):** Veri setinin bütünlüğünü kontrol eden scriptler ve 44 adet kapsamlı Pytest fonksiyonu yazıldı. Testler %100 başarılı.

> **NOT:** Veri klasörü (data/) boyutu çok büyük olduğu için (14 GB) GitHub'da yoktur. OneDrive vb. bulut üzerinden aktarılan `data` klasörünün projenin kök dizinine yerleştirilmesi gerekmektedir. (Ayrıntılar `README.md` dosyasındadır).

---

## 🚀 BUNDAN SONRA YAPILACAKLAR (Faz 4 ve Sonrası)

Veri işleme ve **RAG (Tıbbi Literatür) Bilgi Tabanı** kısımları (Faz 3 dahil) eksiksiz tamamlanmıştır.

### ✅ Tamamlanan Faz 3 (RAG Pipeline) Özeti
- PubMed/Entrez API üzerinden 10 farklı dermatoloji kategorisinde **705 akademik makale** indirildi (`data/knowledge_base/raw_docs`).
- Makaleler abstract ve tam metin olarak anlamsal parçalara (chunk) bölündü.
- **PubMedBERT** modeli kullanılarak metin embedding'leri oluşturuldu ve **ChromaDB** vektör veritabanına indekslendi.
- Hibrit Arama (Dense + BM25) ve Cross-encoder tabanlı re-ranking mekanizmaları (`src/rag/retriever.py`, `src/rag/reranker.py`) kodlandı.
- Tüm bu ağır işlemlerin Google Colab üzerinde ücretsiz T4 GPU ile saniyeler içinde yapılabilmesi için `notebooks/02_build_kb_colab.ipynb` hazırlandı.

---

### 🟣 Acil Sıradaki Görev: Faz 4 (Vision Model - Görüntü İşleme Bacağı)
Arkadaşınızın ve ona eşlik edecek AI'ın görevi buradan başlıyor!
1. **Hedef:** Hazırlanan PyTorch veri setlerini (ISIC+PAD-UFES) kullanarak bir Vision Encoder modelini fine-tune (hassas ayar) etmek.
2. **Kullanılacak Model:** Akademik doğruluk için **BiomedCLIP** (veya muadili tıbbi ViT-B/16).
3. **Beklenen Çıktı:** Görüntüleri alıp tıbbi/anlamsal vektörlere çeviren ve `src/models/vision_encoder.py` içinde çalışan bir modül.

### 🟢 Faz 5: LLM Entegrasyonu (Sentez)
- GPT-4o, Claude veya Gemini kullanılarak; Vision Model'den gelen "Görüntü özellikleri" ile ChromaDB'den (RAG) gelen "Tıbbi Literatürün" birleştirilip doktora "Karar Destek Çıktısı" sunacak Prompt Pipeline'ının yazılması.

### 🔴 Faz 6 & 7: Değerlendirme (Evaluation) ve Arayüz
- `ragas` kütüphanesi ve scikit-learn ile Doğruluk (Accuracy), NDCG gibi akademik metriklerin ölçülmesi.
- Gradio veya Streamlit kullanılarak klinik kullanıma uygun demo arayüzü tasarlanması.

---

## 🤖 Antigravity AI'a (Sisteme) Özel Talimatlar:
Eğer bu dosyayı okuyan bir AI isen, lütfen şu kurallara dikkat et:
1. Projenin genel mimarisine hakim olmak için her zaman kök dizindeki `README.md` ve bu belgeyi temel al.
2. Yazdığın her kod, `src/` altındaki dizin hiyerarşisine uymalı ve mutlaka açıklayıcı docstring'ler içermelidir.
3. `tests/` klasörü altına yazdığın her modül için unit testleri yazmayı unutma ve `pytest` ile testleri geçirdiğinden emin ol.
4. Sıradaki görevi yaparken işlemi tamamladıktan sonra GitHub'a (`git add`, `git commit`, `git push`) basmayı unutma.
5. Kullanıcıyla daima adım adım iletişim kurarak ilerle.
6. İşine **Faz 4 (Vision Model) kodlarını** `src/models/` klasöründe tasarlayarak başla. Colab kullanılması gerektiğini kullanıcıya hatırlat. Kolay gelsin!
