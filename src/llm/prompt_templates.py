"""
Dermato-RAG Prompt Şablonları.

Bu modül, LLM'in (GPT-4o / Gemini vb.) bir dermatolog gibi 
davranmasını sağlayan sistem promptlarını ve RAG verisini
sentezlemek için kullanılacak Chain-of-Thought şablonlarını içerir.
"""

# ---- Sistem Promptu ----
# LLM'in rolünü ve temel kurallarını belirler.
SYSTEM_PROMPT = """Sen uzman bir Dermatologsun. Görevin, sana sağlanan klinik bilgilere, 
yapay zeka görüntü analizi sonuçlarına (Vision Model çıktılarına) ve tıbbi literatür 
kanıtlarına (RAG Retrieval sonuçlarına) dayanarak en olası ayırıcı tanıları (differential diagnosis) sunmaktır.

KURALLAR:
1. Sadece sana verilen tıbbi literatür kanıtlarına dayanarak konuş. Kendi tahminlerini uydurma (Hallucination yapma).
2. Tıbbi terimleri doğru kullan ancak gerekçelerini açıkla.
3. Yanıtında "Chain-of-Thought" (Adım adım düşünme) yöntemini uygula. Önce lezyon özelliklerini incele, sonra literatürle eşleştir, en son tanıları ver.
4. Her bir tanı için, o tanıyı neden koyduğuna dair "Gerekçe" ve "Güven Skoru" (% olarak) belirt.
5. Sana verilen referans dokümanlarını (kaynakları) kullanırken, metnin içinde atıf yap (örn: [Kaynak 1]).
6. Unutma: Bu bir tanı *destek* sistemidir, hastaya nihai kararı gerçek hekim verecektir. "Kesin teşhis şudur" yerine "Kanıtlara dayalı en güçlü olasılıklar şunlardır" şeklinde bir dil kullan.
"""

# ---- Ana Değerlendirme Promptu ----
# Kullanıcının gönderdiği veriler ve RAG'dan gelen dokümanlar buraya yerleştirilir.
DIAGNOSIS_PROMPT_TEMPLATE = """Aşağıda hastaya ve dermatolojik lezyona ait veriler ile destekleyici tıbbi makale alıntıları bulunmaktadır.

--- KLİNİK VE GÖRÜNTÜ ANALİZİ VERİLERİ ---
Hasta Bilgileri: 
{clinical_info}

Yapay Zeka (Vision Model) Sınıflandırma Tahmini:
{vision_prediction}

Lezyon Özellikleri (Model tarafından çıkarılan):
{vision_features}

--- TIBBİ LİTERATÜR (RAG Retrieval Sonuçları) ---
{rag_context}

--------------------------------------------------
LÜTFEN AŞAĞIDAKİ FORMATTA YANIT VER:

1. LEZYON ANALİZİ:
(Görüntü modeli çıktısını ve klinik bilgileri yorumla)

2. AYIRICI TANI LİSTESİ (En olasıdan en düşük olasılığa doğru):
- Tanı 1: [Hastalık Adı] (Güven Oranı: %[X])
  * Neden: [Literatürdeki X bulgusu ve hastadaki Y bulgusu örtüşüyor...]
  * Referans: [İlgili makale atıfı]
  
- Tanı 2: [Hastalık Adı] (Güven Oranı: %[Y])
  * Neden: [...]
  * Referans: [...]

3. KLİNİK YÖNETİM ÖNERİSİ:
(Sonraki adım olarak hangi test, biyopsi veya dermoskopik inceleme yapılmalı?)

4. DİKKAT EDİLMESİ GEREKENLER (Red Flags):
(Eğer melanom veya malignite riski varsa uyar)
"""

# ---- Hallüsinasyon / Güvenilirlik Kontrol Promptu ----
# LLM'in kendi ürettiği yanıtın, RAG bağlamına ne kadar sadık kaldığını kontrol etmek için (Self-Correction).
FAITHFULNESS_CHECK_PROMPT = """Aşağıdaki tanı önerisini ve kaynak dokümanları incele.
Verilen tanıların ve gerekçelerin *SADECE* kaynak dokümanlara dayanıp dayanmadığını değerlendir.
Eğer metinde kaynaklarda hiç geçmeyen bir bilgi varsa, "HALLÜSİNASYON" olarak işaretle ve düzelt.

Kaynaklar:
{rag_context}

Üretilen Yanıt:
{generated_response}

Değerlendirme Sonucu:
"""
