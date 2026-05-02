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

## 🚀 BUNDAN SONRA YAPILACAKLAR (Faz 2b ve Sonrası)

Sıradaki hedef, uygulamanın **RAG (Retrieval-Augmented Generation)** kısmına, yani "Tıbbi Literatür" bacağı olan **Faz 3'e** geçiş yapmaktır. Ancak ondan hemen önce eksik kalan küçük bir veri çekme adımı vardır.

### 🟡 Acil Sıradaki Görev: Faz 2b (PubMed Makalelerinin Çekilmesi)
Görüntü verilerini hallettik ancak bilgi tabanı (Knowledge Base) için tıbbi makale metinleri henüz toplanmadı.
1. **Hedef:** PubMed/Entrez API kullanılarak dermatoloji alanındaki referans makalelerin indirilmesi.
2. **Gereksinim:** NCBI hesabı açılarak bir API Anahtarı alınması ve `.env` dosyasına eklenmesi gerekiyor.
3. **Çıktı:** İndirilen makalelerin `data/knowledge_base/raw_docs` klasörüne kaydedilmesi.

### 🔵 Faz 3: RAG Pipeline Kurulumu
1. **Chunking (`src/rag/chunking.py`):** PubMed'den çekilen makalelerin anlamsal (semantic) olarak mantıklı paragraflara / parçalara bölünmesi.
2. **Embedding (`src/rag/embedding.py`):** Metinlerin vektör uzayına çıkarılması. Metin için `PubMedBERT`, ilerideki multimodalite için `BiomedCLIP` modellerinin denenmesi.
3. **Vector Database:** `ChromaDB` veya `FAISS` kurularak embeddinglerin kaydedilmesi.
4. **Retriever ve Re-ranker:** Sorguya göre en iyi kanıtların getirilmesi ve Cross-Encoder ile yeniden sıralanması.

### 🟣 Faz 4: Vision Model (Görüntü İşleme Bacağı)
- Hazırlanan PyTorch veri setlerinin kullanılarak bir Vision Encoder (ViT-B/16 veya BiomedCLIP) modelinin fine-tune (hassas ayar) edilmesi.

### 🟢 Faz 5: LLM Entegrasyonu (Sentez)
- GPT-4o, Claude veya Gemini kullanılarak; Vision Model'den gelen "Görüntü özellikleri" ile RAG'dan gelen "Tıbbi Literatürün" birleştirilip doktora "Karar Destek Çıktısı" sunacak Prompt Pipeline'ının yazılması.

### 🔴 Faz 6 & 7: Değerlendirme (Evaluation) ve Arayüz
- `ragas` kütüphanesi ve scikit-learn ile Doğruluk (Accuracy), NDCG gibi akademik metriklerin ölçülmesi.
- Gradio veya Streamlit kullanılarak demo arayüzü tasarlanması.

---

## 🤖 Antigravity AI'a (Sisteme) Özel Talimatlar:
Eğer bu dosyayı okuyan bir AI isen, lütfen şu kurallara dikkat et:
1. Projenin genel mimarisine hakim olmak için her zaman kök dizindeki `README.md` ve bu belgeyi temel al.
2. Yazdığın her kod, `src/` altındaki dizin hiyerarşisine uymalı ve mutlaka açıklayıcı docstring'ler içermelidir.
3. `tests/` klasörü altına yazdığın her modül için unit testleri yazmayı unutma ve `pytest` ile testleri geçirdiğinden emin ol.
4. Sıradaki görevi yaparken (örn: PubMed makale indirme), işlemi tamamladıktan sonra GitHub'a (`git add`, `git commit`, `git push`) basmayı unutma.
5. Kullanıcıyla daima adım adım iletişim kurarak (örn: "Önce script'i yazıyorum, sonra API ile deneme yapacağım") onay al.
6. İşine **Faz 2b (PubMed Makale Çekme Scripti `scripts/build_knowledge_base.py`)** yazarak başla! Kolay gelsin!
