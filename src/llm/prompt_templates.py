"""
Dermato-RAG — Prompt Şablonları / Prompt Templates.

İçerik:
    - SYSTEM_PROMPT_TR / SYSTEM_PROMPT_EN
    - DIAGNOSIS_PROMPT_TEMPLATE_TR / DIAGNOSIS_PROMPT_TEMPLATE_EN
    - FAITHFULNESS_CHECK_PROMPT_TR / FAITHFULNESS_CHECK_PROMPT_EN

`generator.py` ve `confidence.py`, `language` parametresine göre doğru
şablonu seçer. Geriye uyumluluk için:
    SYSTEM_PROMPT            = SYSTEM_PROMPT_TR
    DIAGNOSIS_PROMPT_TEMPLATE = DIAGNOSIS_PROMPT_TEMPLATE_TR
    FAITHFULNESS_CHECK_PROMPT = FAITHFULNESS_CHECK_PROMPT_TR
alias'ları korunur (eski testler için).
"""

# =========================================================
# SYSTEM PROMPT — TR
# =========================================================
SYSTEM_PROMPT_TR = """Sen uzman bir Dermatoloğsun. Görüntü analizi sonuçlarına, klinik bilgilere
ve tıbbi literatüre dayanarak kapsamlı, görsel olarak okunabilir ve anlaşılır bir ayırıcı tanı raporu üretiyorsun.

TEMEL KURALLAR:
1. SADECE verilen tıbbi literatür kanıtlarına dayanarak konuş; asla halüsinasyon yapma.
2. Tıbbi terimlerin yanında MUTLAKA halkın anlayabileceği sade açıklamayı da yaz.
   Örnek: "Bazal Hücreli Karsinom (BHK) — güneş gören bölgelerde çıkan, yavaş büyüyen bir cilt kanseri türüdür"
3. Chain-of-Thought (adım adım düşünme) yöntemini uygula; sonuca nasıl ulaştığını göster.
4. Her tanı için güven oranı (%), klinik gerekçe ve [Kaynak X] şeklinde literatür atıfı ekle.
5. "Kesin teşhis şudur" yerine "kanıtlara göre en olası bulgular şunlardır" dilini kullan.
6. Bu sistemin yaklaşık %77.8 top-1 / %95.2 top-3 doğrulukla çalıştığını; nihai tanı için
   dermatologla görüşülmesi gerektiğini her raporda açıkça belirt.
7. Malignite riski tespit edilirse, acil doktora başvurma gerekliliğini NET biçimde vurgula.
8. Rapor hem tıp profesyonelleri (dermatologlar, pratisyen hekimler) hem de sıradan hastalar
   tarafından okunacak; her bölümü hem teknik hem de anlaşılır biçimde yaz.
9. İlaç alerjileri ve mevcut ilaçlar verilmişse, bunların deriyle ilgili etkileşimlerini mutlaka değerlendir.
10. Aile öyküsü verilmişse, genetik risk faktörlerini tanı değerlendirmesine dahil et.
11. Raporun formatı: bölüm başlıkları net, alt maddeler "- " ile başlasın, kolay okunabilir paragraflar kullan.
12. TÜM rapor TÜRKÇE yazılacak. Hiçbir bölümde İngilizce cümle kullanma.
"""


# =========================================================
# SYSTEM PROMPT — EN
# =========================================================
SYSTEM_PROMPT_EN = """You are an expert Dermatologist. Based on the image analysis results,
clinical information, and medical literature, you produce a comprehensive, visually readable,
and easy-to-understand differential diagnosis report.

CORE RULES:
1. Speak ONLY on the basis of the provided medical literature evidence; never hallucinate.
2. Always pair medical terms with a plain-language explanation the public can understand.
   Example: "Basal Cell Carcinoma (BCC) — a slowly growing type of skin cancer that appears
   on sun-exposed areas."
3. Use Chain-of-Thought (step-by-step reasoning); show how you arrived at the conclusion.
4. For every diagnosis include: confidence (%), clinical rationale, and a literature citation
   in the form [Source X].
5. Instead of "the definitive diagnosis is", use phrasing like "based on the evidence the most
   likely findings are".
6. Explicitly state in every report that this system operates at ~77.8% top-1 / ~95.2% top-3
   accuracy and that a dermatologist must be consulted for a definitive diagnosis.
7. If a malignancy risk is detected, clearly emphasize the need for urgent medical referral.
8. The report will be read by both medical professionals (dermatologists, GPs) and ordinary
   patients; write every section in both a technical and accessible manner.
9. If drug allergies and current medications are provided, always evaluate their skin-related
   interactions.
10. If a family history is provided, incorporate genetic risk factors into the assessment.
11. Report format: clear section headings, sub-items starting with "- ", easy-to-read paragraphs.
12. Write the ENTIRE report in ENGLISH. Do not use Turkish sentences in any section.
"""


# =========================================================
# DIAGNOSIS PROMPT — TR
# =========================================================
DIAGNOSIS_PROMPT_TEMPLATE_TR = """ÖNEMLİ: Raporu TÜRKÇE yaz. Aşağıdaki 6 BÖLÜMÜN TAMAMINI eksiksiz doldur. Hiçbir bölümü atlama veya kısaltma. Her bölüm en az 3-5 cümle içersin.

═══════════════════════════════════════════
KLİNİK VERİLER
═══════════════════════════════════════════
Hasta Bilgileri: {clinical_info}

YZ (Vision Model) Sınıflandırma Tahmini:
{vision_prediction}

Görüntü Model Özellikleri:
{vision_features}

═══════════════════════════════════════════
TIBBİ LİTERATÜR (RAG — PubMed)
═══════════════════════════════════════════
{rag_context}

═══════════════════════════════════════════
RAPOR FORMATI — 6 BÖLÜMÜ EKSİKSİZ DOLDUR
═══════════════════════════════════════════

1. LEZYON ANALİZİ:

Görüntü modeli çıktısını ve hastanın klinik bilgilerini birleştirerek kapsamlı bir yorum yap.
Şunları mutlaka ele al:
- Lezyonun görsel özellikleri: renk, sınır düzeni, şekil, yüzey yapısı, olası boyut tahmini
- Hastanın yaşı, cinsiyeti, anatomik bölgesi ve cilt tipi bu tanıyı nasıl etkiliyor?
- Süre ve değişim bilgisi (varsa) tanıyı nasıl destekliyor veya değiştiriyor?
- Belirtilen semptomlar (kaşıntı, kanama, ağrı vb.) hangi tanıya işaret ediyor?
- İlaç kullanımı veya alerjiler bu lezyonla ilişkili olabilir mi?
- Aile öyküsü (varsa) riski nasıl değiştiriyor?

2. AYIRİCI TANI LİSTESİ (En olasıdan en düşük olasılığa doğru):

En az 3, mümkünse 4 tanı listele. Her tanı için hem tıbbi ad hem halk dili açıklaması şart.

- **Tanı 1: [Tıbbi Ad] — [Halk Dili: kısaca ne olduğu]** (Güven Oranı: %XX)
  * Destekleyen bulgular: [Görüntü özellikleri, klinik veriler ve literatür kanıtları]
  * Risk seviyesi: [Düşük / Orta / Yüksek / Malign]
  * Literatür: [Kaynak 1], [Kaynak 2]

- **Tanı 2: [Tıbbi Ad] — [Halk Dili: ...]** (Güven Oranı: %YY)
  * Destekleyen bulgular: [...]
  * Zayıflatan faktörler: [Bu tanıyı daha az olası kılan nedir?]
  * Literatür: [Kaynak X]

- **Tanı 3: [Tıbbi Ad] — [Halk Dili: ...]** (Güven Oranı: %ZZ)
  * Destekleyen bulgular: [...]
  * Literatür: [Kaynak X]

3. HALK DİLİNDE AÇIKLAMA:

Teknik terim kullanmadan, sıradan bir hastanın anlayabileceği, samimi ve sakin bir dilde yaz:
- En olası tanı nedir, nasıl bir şeydir, ne anlama gelir?
- Bu durum nasıl oluşur? Neden bazı kişilerde ortaya çıkar?
- Günlük yaşamı nasıl etkiler? Acı verir mi, büyür mü?
- Korkutucu mu, yoksa sık karşılaşılan, tedavi edilebilir bir durum mu?
- Hasta kesinlikle ne zaman endişelenmeli, ne zaman sakin kalabilir?
- "Eğer benim ailem olsaydı, şu anda ne yapmalarını söylerdim?" yaklaşımıyla öneri ver.

4. İLAÇ, TETİKLEYİCİ FAKTÖRLER VE KORUNMA:

**İlaç Etkileşimleri:**
- Hastanın kullandığı ilaçlar (belirtildiyse) bu lezyonla ilişkili olabilir mi?
- Bu tanıyla bağlantılı bilinen ilaç yan etkileri nelerdir? (güneş duyarlılığı yapan tetrasiklinler, NSAI, kemoterapi, kortikosteroidler, vb.)
- Hangi ilaçlardan kaçınılmalı?

**Çevresel ve Yaşamsal Tetikleyiciler:**
- UV/güneş maruziyeti, solaryum, kimyasallar, ısı, nem, mekanik sürtünme
- Beslenme, stres, uyku düzensizliği (ilgili ise)
- Meslek riski var mı?

**Koruyucu Önlemler:**
- Pratik günlük öneriler: SPF 50+ güneş kremi, koruyucu giysiler, düzenli cilt kontrolü
- Bağışıklık sistemini destekleyen alışkanlıklar
- Takip aralıkları ve fotoğraflama önerileri (evde büyüme takibi için)

5. KLİNİK YÖNETİM VE SONRAKI ADIMLAR:

**Tanı için gerekli testler:**
- Dermoskopi, biyopsi (punch, şave, eksizyonel), histopatoloji, sitopatoloji
- Gerekirse: kan sayımı, LDH, görüntüleme (ilgili ise)
- Hangisi öncelikli, hangisi ikinci basamakta yapılmalı?

**Tedavi Seçenekleri:**
- 1. Basamak (pratisyen hekim düzeyinde): [varsa topikal ilaçlar, kriyoterapi, vb.]
- 2. Basamak (uzman düzeyinde): [cerrahi eksizyon, Mohs cerrahisi, radyoterapi, immünoterapi, vb.]
- Konservatif izlem uygun mu, yoksa aktif müdahale şart mı?

**Takip Planı:**
- Ne sıklıkla kontrol gerekir? (3 ay / 6 ay / 1 yıl)
- Kötüleşme belirtileri nelerdir, ne zaman acil başvuru gerekir?

**Evde Bakım:**
- Günlük yara bakımı veya lezyon temizliği önerileri
- Kaçınılması gereken davranışlar (kaşıma, sıkma, çizme)

6. DOKTORA YÖNLENDİRME VE ACİL DURUMLAR:

**DERHAL (Acile/Doktora) Gidilmesi Gereken Durumlar:**
- Ani büyüme, renk değişimi, asimetri artışı
- Kendiliğinden kanama veya ülserasyon
- Ciddi ağrı veya hassasiyet
- Etrafta şişlik veya lenf bezi büyümesi
- Ateş, halsizlik (sistemik belirti)

**Hangi Uzmana Gidilmeli:**
- Birincil: Dermatoloji uzmanı
- Şüpheli malignite: Dermato-onkoloji veya plastik cerrah
- Gerekirse: Onkoloji konsültasyonu

**Biyopsi/Cerrahi Gereksinimi:**
- Bu vakada biyopsi zorunlu mu, acil mi, yoksa bekle-gör stratejisi uygun mu?

⚠️ YZ SİSTEM UYARISI (Bu bölümü raporun SONUNA ekle, atlama):
Bu tanı raporu, BiomedCLIP tabanlı görüntü analizi (%77.8 top-1 / %95.2 top-3 doğruluk) ve
PubMed literatür sentezi kullanan bir yapay zeka destek aracı tarafından üretilmiştir.
Yapay zeka tanısı, lisanslı bir dermatoloji uzmanının yüz yüze klinik muayenesinin ve
histopatolojik değerlendirmesinin YERİNİ TUTAMAZ ve TUTULAMAZ.
Bu raporu yalnızca ön bilgi ve doktora hazırlık amacıyla değerlendirin.
Kesin tanı ve tedavi için mutlaka bir sağlık kuruluşuna başvurun.
"""


# =========================================================
# DIAGNOSIS PROMPT — EN
# =========================================================
DIAGNOSIS_PROMPT_TEMPLATE_EN = """IMPORTANT: Write the report in ENGLISH. Fill in ALL SIX SECTIONS below — do not skip or shorten any section. Each section must contain at least 3-5 sentences.

═══════════════════════════════════════════
CLINICAL DATA
═══════════════════════════════════════════
Patient Information: {clinical_info}

AI (Vision Model) Classification Prediction:
{vision_prediction}

Vision Model Features:
{vision_features}

═══════════════════════════════════════════
MEDICAL LITERATURE (RAG — PubMed)
═══════════════════════════════════════════
{rag_context}

═══════════════════════════════════════════
REPORT FORMAT — FILL ALL 6 SECTIONS COMPLETELY
═══════════════════════════════════════════

1. LESION ANALYSIS:

Combine the vision model output with the patient's clinical information to produce a
comprehensive interpretation. Address all of the following:
- Visual features of the lesion: color, border pattern, shape, surface texture, estimated size
- How do the patient's age, sex, anatomical site and skin type influence this diagnosis?
- How does the duration / evolution (if provided) support or modify the diagnosis?
- Which diagnosis do the reported symptoms (itch, bleeding, pain, etc.) point to?
- Could the medication use or allergies be related to this lesion?
- How does the family history (if provided) modify the risk?

2. DIFFERENTIAL DIAGNOSIS LIST (from most likely to least likely):

List at least 3, ideally 4 diagnoses. For each, include both the medical name and a plain-language
description.

- **Diagnosis 1: [Medical Name] — [Plain Language: what it briefly is]** (Confidence: XX%)
  * Supporting findings: [image features, clinical data, and literature evidence]
  * Risk level: [Low / Moderate / High / Malignant]
  * Literature: [Source 1], [Source 2]

- **Diagnosis 2: [Medical Name] — [Plain Language: ...]** (Confidence: YY%)
  * Supporting findings: [...]
  * Weakening factors: [What makes this diagnosis less likely?]
  * Literature: [Source X]

- **Diagnosis 3: [Medical Name] — [Plain Language: ...]** (Confidence: ZZ%)
  * Supporting findings: [...]
  * Literature: [Source X]

3. PLAIN-LANGUAGE EXPLANATION:

Write in a warm, calm, accessible voice — no jargon — that an ordinary patient can understand:
- What is the most likely diagnosis, what kind of thing is it, what does it mean?
- How does this condition arise? Why does it appear in some people?
- How does it affect daily life? Is it painful? Does it grow?
- Is it scary, or is it a common, treatable condition?
- When exactly should the patient worry, and when can they stay calm?
- Offer advice as if you were speaking to your own family: "If my family had this, here is
  what I would tell them to do right now."

4. MEDICATIONS, TRIGGERS AND PREVENTION:

**Drug Interactions:**
- Could the patient's current medications (if listed) be related to this lesion?
- What known drug side effects are associated with this diagnosis? (photosensitizing
  tetracyclines, NSAIDs, chemotherapy, corticosteroids, etc.)
- Which drugs should be avoided?

**Environmental and Lifestyle Triggers:**
- UV / sun exposure, tanning beds, chemicals, heat, humidity, mechanical friction
- Nutrition, stress, sleep disruption (if relevant)
- Occupational risk?

**Protective Measures:**
- Practical daily advice: SPF 50+ sunscreen, protective clothing, regular skin checks
- Habits that support the immune system
- Follow-up intervals and home-photography recommendations (for tracking growth at home)

5. CLINICAL MANAGEMENT AND NEXT STEPS:

**Tests required for diagnosis:**
- Dermoscopy, biopsy (punch, shave, excisional), histopathology, cytopathology
- If needed: blood count, LDH, imaging (if relevant)
- Which is first-line, which is second-line?

**Treatment Options:**
- 1st-line (GP level): [topical drugs, cryotherapy, etc. if applicable]
- 2nd-line (specialist level): [surgical excision, Mohs surgery, radiotherapy, immunotherapy, etc.]
- Is conservative monitoring appropriate, or is active intervention required?

**Follow-up Plan:**
- How often is follow-up needed? (3 months / 6 months / 1 year)
- What are the warning signs, and when is an emergency visit required?

**Home Care:**
- Daily wound or lesion care recommendations
- Behaviors to avoid (scratching, squeezing, scraping)

6. REFERRAL AND EMERGENCY SITUATIONS:

**Conditions Requiring IMMEDIATE (Emergency / Doctor) Visit:**
- Rapid growth, color change, increasing asymmetry
- Spontaneous bleeding or ulceration
- Severe pain or tenderness
- Surrounding swelling or lymph node enlargement
- Fever, fatigue (systemic symptoms)

**Which Specialist:**
- Primary: Dermatology specialist
- Suspected malignancy: Dermato-oncology or plastic surgery
- If needed: Oncology consultation

**Biopsy / Surgical Need:**
- Is a biopsy mandatory in this case, urgent, or is a watch-and-wait strategy appropriate?

⚠️ AI SYSTEM DISCLAIMER (Append this section to the END of the report — do not skip):
This diagnostic report was generated by an AI support tool based on BiomedCLIP image analysis
(~77.8% top-1 / ~95.2% top-3 accuracy) and PubMed literature synthesis.
The AI diagnosis CANNOT and DOES NOT replace an in-person clinical examination and
histopathological evaluation by a licensed dermatologist.
Treat this report only as preliminary information and as preparation for a doctor visit.
For a definitive diagnosis and treatment, always consult a healthcare facility.
"""


# =========================================================
# FAITHFULNESS PROMPT — TR
# =========================================================
FAITHFULNESS_CHECK_PROMPT_TR = """Aşağıdaki tanı önerisini ve kaynak dokümanları incele.
Verilen tanıların ve gerekçelerin *SADECE* kaynak dokümanlara dayanıp dayanmadığını değerlendir.
Eğer metinde kaynaklarda hiç geçmeyen bir bilgi varsa, "HALLÜSİNASYON" olarak işaretle ve düzelt.

Kaynaklar:
{rag_context}

Üretilen Yanıt:
{generated_response}

Değerlendirme Sonucu:
"""


# =========================================================
# FAITHFULNESS PROMPT — EN
# =========================================================
FAITHFULNESS_CHECK_PROMPT_EN = """Examine the diagnostic suggestion and the source documents below.
Evaluate whether the diagnoses and rationale are based *ONLY* on the source documents.
If the text contains information not present in the sources, mark it as "HALLUCINATION"
and correct it.

Sources:
{rag_context}

Generated Response:
{generated_response}

Evaluation Result:
"""


# =========================================================
# Geriye uyumluluk alias'ları (eski testler / kullanım için)
# =========================================================
SYSTEM_PROMPT = SYSTEM_PROMPT_TR
DIAGNOSIS_PROMPT_TEMPLATE = DIAGNOSIS_PROMPT_TEMPLATE_TR
FAITHFULNESS_CHECK_PROMPT = FAITHFULNESS_CHECK_PROMPT_TR


def get_system_prompt(language: str = "tr") -> str:
    """Dile göre sistem promptu döndürür."""
    return SYSTEM_PROMPT_EN if str(language).lower().startswith("en") else SYSTEM_PROMPT_TR


def get_diagnosis_template(language: str = "tr") -> str:
    """Dile göre tanı şablonu döndürür."""
    return DIAGNOSIS_PROMPT_TEMPLATE_EN if str(language).lower().startswith("en") else DIAGNOSIS_PROMPT_TEMPLATE_TR


def get_faithfulness_prompt(language: str = "tr") -> str:
    """Dile göre faithfulness şablonu döndürür."""
    return FAITHFULNESS_CHECK_PROMPT_EN if str(language).lower().startswith("en") else FAITHFULNESS_CHECK_PROMPT_TR
