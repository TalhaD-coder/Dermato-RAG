# 🖥️ Dermato-RAG Dashboard — Tasarım Spesifikasyonu

> **Uygulayacak:** Talha  
> **Teknoloji:** Streamlit (Python)  
> **Hedef:** Modern, sade, anlaşılır medikal AI arayüzü

---

## 🎨 Görsel Referans

![Dashboard Mockup](C:\Users\DİJLE DOĞAN\.gemini\antigravity\brain\6e3f9030-2f73-4693-a116-e887bfc609e0\dermato_rag_dashboard_1779022828177.png)

---

## 🎨 Renk Paleti

| Kullanım | Renk | Hex |
|----------|------|-----|
| Ana arka plan | Koyu lacivert | `#0F1623` |
| Kart arka planı | Koyu mavi | `#1E2A3E` |
| İkincil arka plan | Orta lacivert | `#131C2E` |
| Ana metin | Beyaz | `#FFFFFF` |
| İkincil metin | Açık gri | `#8B9AB1` |
| Vurgu (buton) | Mavi→Mor gradient | `#3B82F6 → #8B5CF6` |
| Başarı/Bağlı | Yeşil | `#10B981` |
| Tehlike (melanom) | Kırmızı | `#EF4444` |
| Uyarı | Turuncu | `#F59E0B` |
| Güvenli | Yeşil | `#10B981` |

---

## 📐 Layout Yapısı

```
┌─────────────────────────────────────────────────────────┐
│  SIDEBAR (Sol, 280px)  │     ANA İÇERİK ALANI          │
│                        │                                 │
│  🧬 DermatoRAG         │  📋 Lezyon Analizi             │
│  ─────────────────     │  ─────────────────────────────  │
│  📊 Analiz             │  ┌──────────┐  ┌────────────┐  │
│  📁 Geçmiş (opsiyonel) │  │  YÜKLEME │  │  SONUÇLAR  │  │
│  ℹ️  Hakkında          │  │  KARTI   │  │  KARTI     │  │
│                        │  │          │  │            │  │
│  ─────────────────     │  │ [📷 sürükle│ Güven Barları│  │
│  ● Gemini Bağlı        │  │  & bırak]│  │            │  │
│                        │  │          │  │  AI Raporu │  │
│                        │  │ [Hasta   │  │            │  │
│                        │  │  Bilgisi]│  │  Kaynaklar │  │
│                        │  │          │  │            │  │
│                        │  │ [ANALİZ  │  │            │  │
│                        │  │    ET]   │  │            │  │
│                        │  └──────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🧩 Bileşenler ve Açıklamalar

### 1. Sidebar
- **Logo:** `🧬 DermatoRAG` — büyük, beyaz, bold
- **Navigasyon:** Analiz, Hakkında (basit linkler)
- **Durum göstergesi:** `● Gemini 1.5 Pro Bağlı` — yeşil nokta, küçük yazı
- **Alt bilgi:** `v1.0 | ISIC + PAD-UFES`

### 2. Yükleme Kartı (Sol Panel)
- **Görüntü yükleme:** Drag & drop alanı, kesik çerçeve, kamera ikonu
- **Önizleme:** Yüklenen görüntü yüklenince aynı alanda gösterilsin
- **Hasta Bilgisi:** Çok satırlı metin kutusu
  - Placeholder: `"Yaş, lezyon bölgesi, ne zamandır var, semptomlar..."`
- **Analiz Et butonu:** Tam genişlik, mavi-mor gradient, büyük font

### 3. Sonuçlar Kartı (Sağ Panel)
Analiz tamamlanınca 3 bölüm sırayla görünür:

**Bölüm A — Vision Model Tahmini**
- Başlık: `🔬 Vision Model Tahmini`
- Her sınıf için yatay progress bar:
  - Renk: %70+ = kırmızı, %30-70 = turuncu, %30- = yeşil
  - Format: `Melanoma ████████░░ 82%`

**Bölüm B — AI Tanı Raporu**
- Başlık: `🤖 AI Tanı Raporu`
- Genişletilebilir (expander) metin alanı
- Gemini'dan gelen rapor markdown formatında gösterilsin

**Bölüm C — Literatür Kaynakları**
- Başlık: `📚 Literatür Kaynakları`
- RAG'dan gelen her makale için küçük kart:
  - Makale adı, dergi, yıl
  - Tıklanabilir (DOI linki varsa)

---

## 💻 Kurulum ve Çalıştırma

### Streamlit Kurulumu
```bash
pip install streamlit plotly
```

### Dosya Yapısı
```
Dermato-RAG/
└── app/
    └── streamlit_app.py   ← Ana dashboard dosyası
```

### Çalıştırma
```bash
streamlit run app/streamlit_app.py
```
Tarayıcıda `http://localhost:8501` adresinde açılır.

---

## 📋 Streamlit Tema Ayarı

`.streamlit/config.toml` dosyası oluştur:
```toml
[theme]
primaryColor = "#3B82F6"
backgroundColor = "#0F1623"
secondaryBackgroundColor = "#1E2A3E"
textColor = "#FFFFFF"
font = "sans serif"
```

---

## ⚠️ Önemli Notlar

1. **`.env` dosyasını** kendi Google Gemini API anahtarınla doldur
2. **`best_model.pt`** dosyası `models/` klasöründe olmalı
3. Pipeline'ı import etmeden önce `PYTHONPATH` ayarı:
   ```bash
   # Windows:
   set PYTHONPATH=.
   streamlit run app/streamlit_app.py
   
   # Mac/Linux:
   PYTHONPATH=. streamlit run app/streamlit_app.py
   ```
4. İlk yüklemede model ~15-30 saniye bekleyebilir (normal)

---

## 🚫 Yapılmaması Gerekenler
- `.env` dosyasını asla GitHub'a yükleme
- `best_model.pt` dosyasını asla GitHub'a yükleme
- API anahtarını kod içine yazma
