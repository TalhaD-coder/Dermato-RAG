/* ============================================================
   DermatoRAG — Enterprise SPA Logic v3.0
   ============================================================ */

/* ── DICTIONARY ── */
const DICT = {
  en: {
    hero_badge:"Clinical Grade AI", hero_title:"Next-Generation Dermatological Diagnostics",
    hero_desc:"Empowering dermatologists and patients with state-of-the-art vision models and real-time PubMed literature synthesis.",
    hero_btn:"Launch System →", hero_tour:"Tutorial →",
    feat1_title:"Vision AI", feat1_desc:"9-Class Multi-model",
    feat2_title:"RAG Engine", feat2_desc:"PubMed Integration",
    feat3_title:"Differential Dx", feat3_desc:"9 Classes · Confidence",

    menu_analysis:"New Analysis", menu_history:"Patient History", menu_settings:"Settings",
    status_online:"System Online", status_loading:"Connecting...", status_err:"Offline",
    step_1:"Upload Image", step_2:"Clinical Context", step_3:"AI Report",

    upload_head:"Lesion Image Input",
    upload_subhead:"Provide a high-quality dermoscopic or macroscopic clinical image.",
    upload_drag:"Drag & Drop Image", upload_browse:"or click to browse local files",
    sample_text:"Or test with clinical samples:",
    btn_continue:"Continue →",

    ctx_head:"Clinical Context Formulation",
    ctx_subhead:"Patient demographics and lesion history significantly improve AI diagnostic accuracy.",
    step2_notice_head:"Why enter patient information?",
    step2_notice_body:"Age, gender, and anatomical location help the AI search more relevant medical literature and generate a more accurate differential diagnosis. Your data stays in your browser — nothing is stored on the server.",
    lbl_age:"Patient Age", ph_age:"e.g., 45",
    lbl_gender:"Gender", opt_select:"Select...", opt_f:"Female", opt_m:"Male", opt_o:"Other",
    lbl_location:"Anatomical Location", ph_loc:"e.g., Anterior torso, left forearm",
    lbl_duration:"Evolution & Duration", ph_dur:"e.g., Rapidly growing over 3 months",
    lbl_symp:"Clinical Notes & Symptoms", ph_symp:"e.g., Asymmetric borders, occasional bleeding",
    lbl_skintype:"Skin Type (Fitzpatrick)",
    opt_skin1:"Type I — Very fair, always burns", opt_skin2:"Type II — Fair, usually burns",
    opt_skin3:"Type III — Medium, sometimes burns", opt_skin4:"Type IV — Olive, rarely burns",
    opt_skin5:"Type V — Brown, very rarely burns", opt_skin6:"Type VI — Dark brown, never burns",
    lbl_allergy:"Known Allergies", ph_allergy:"e.g., Penicillin, latex, nickel",
    lbl_meds:"Current Medications", ph_meds:"e.g., Immunosuppressants, steroids, NSAIDs",
    lbl_famhist:"Family History", opt_famhist_0:"None / Unknown",
    opt_famhist_1:"Melanoma in family",
    opt_famhist_2:"Basal cell carcinoma (BCC) in family",
    opt_famhist_3:"Squamous cell carcinoma (SCC) in family",
    opt_famhist_4:"Multiple skin cancer cases in family",
    opt_famhist_5:"Psoriasis in family",
    opt_famhist_6:"Atopic dermatitis (eczema) in family",
    opt_famhist_7:"Genetic skin condition in family",
    opt_famhist_8:"Autoimmune / immune condition in family",
    opt_famhist_9:"Other skin cancer history in family",
    remove_image:"Remove Image",
    btn_back:"← Back", btn_analyze:"Run Differential Diagnosis",
    quick_fill_btn:"Load Last Patient",
    clinical_required:"Patient Info Required",
    clinical_required_msg:"Please fill in at least one field (age, location, duration, or symptoms) to enable a more accurate AI diagnosis. Without clinical context, the AI cannot access relevant literature.",
    unsupported_head:"Unsupported Conditions:",
    unsupported_body:"Acne, eczema, fungal infections, psoriasis and other skin conditions are NOT supported by this system. Only the 9 dermatological lesion classes listed above can be analyzed.",

    load_title:"Synthesizing Data...", load_elapsed:"Elapsed:", load_est:"· est. ~20s",
    t_vision:"Vision Model Encoding", t_rag:"PubMed Literature Retrieval", t_llm:"Gemini Reasoning Engine",

    ood_strong:"Image Not Recognized:",
    ood_msg:" The uploaded image did not match any of the 9 supported skin lesion classes. AI prediction withheld.",
    mal_strong:"Malignancy Warning:",
    mal_msg:" Model detects a high probability of a malignant lesion. Biopsy and histopathological review recommended.",
    low_conf_strong:"Low Confidence Warning:",
    low_conf_msg:" Model is uncertain about this image (45–65%). Results are for reference only; clinical evaluation is strongly recommended.",

    card_primary:"Primary AI Prediction", conf_text:"Confidence Score",
    card_image:"Analyzed Image", card_report:"Diagnostic Report (Gemini 2.5)",
    card_probs:"Differential Probabilities", card_lit:"Supporting Literature (RAG)",
    btn_restart:"← New Analysis", open_article:"Open Article →",
    err_select:"Please select an image first.", no_lit:"No supporting literature found.",

    abcde_btn_title:"ABCDE Dermoscopy Guide",
    abcde_guide_title:"ABCDE Dermoscopy Guide",
    abcde_a_title:"Asymmetry", abcde_a_desc:"Two halves are not mirror images",
    abcde_b_title:"Border",    abcde_b_desc:"Irregular, ragged, notched edges",
    abcde_c_title:"Color",     abcde_c_desc:"Multiple shades or uneven distribution",
    abcde_d_title:"Diameter",  abcde_d_desc:"Larger than 6mm (pencil eraser)",
    abcde_e_title:"Evolution", abcde_e_desc:"Change in size, shape or color",

    sim_title:"Similar Past Cases", sim_none:"No similar saved cases.",
    hist_title:"Patient History", hist_sub:"Past analyses are stored locally in your browser.",
    hist_empty:"No analyses yet. Run your first analysis to see it here.",
    hist_view_btn:"View", hist_del_btn:"Delete", hist_clear_btn:"Clear All History", hist_no_image:"No image",

    set_title:"Settings", set_sub:"Manage your application preferences.",
    set_lang_label:"Interface Language", set_theme_label:"Theme",
    theme_dark:"Dark", theme_light:"Light",
    set_hist_label:"Analysis History", set_hist_analyses:"saved analyses",
    set_clear_btn:"Clear All", set_tour_label:"Tutorial", set_tour_btn:"Restart Tutorial",
    set_about_label:"About DermatoRAG",
    set_about_text:"DermatoRAG v1.0 — AI-Powered Dermatological Diagnostic Support System",
    set_model_text:"Vision: BiomedCLIP fine-tuned + TTA | Top-1: 73% · Top-3: ~87% | LLM: Gemini 2.5 Flash | RAG: 706 PubMed articles",
    set_about_note:"⚠️ This system is intended as a diagnostic support tool only. AI output does not replace clinical evaluation by a licensed dermatologist.",

    tour_skip:"Skip", tour_next:"Next →", tour_finish:"Get Started!",
    shortcuts_title:"Keyboard Shortcuts",
    sh_continue:"Continue when image is selected",
    sh_analyze:"Run analysis",
    sh_close:"Close modal / panel",
    sh_shortcuts:"Toggle this panel",

    pdf_preparing:"Preparing PDF...", pdf_done:"PDF downloaded!", pdf_error:"Could not generate PDF.",
    copied:"Copied to clipboard!",
    modal_ok:"OK", modal_cancel:"Cancel",
    sc_label:"9 Supported Classes:",
  },
  tr: {
    hero_badge:"Klinik Seviye Yapay Zeka", hero_title:"Yeni Nesil Dermatolojik Teşhis Asistanı",
    hero_desc:"Dermatologları ve hastaları en son teknoloji görüntü işleme modelleri ve gerçek zamanlı PubMed literatür sentezi ile güçlendiriyoruz.",
    hero_btn:"Sistemi Başlat →", hero_tour:"Öğretici →",
    feat1_title:"Görüntü İşleme AI", feat1_desc:"9 Sınıflı Multi-model",
    feat2_title:"RAG Motoru", feat2_desc:"PubMed Entegrasyonu",
    feat3_title:"Ayırıcı Tanı", feat3_desc:"9 Sınıf · Güven Skoru",

    menu_analysis:"Yeni Analiz", menu_history:"Hasta Geçmişi", menu_settings:"Ayarlar",
    status_online:"Sistem Aktif", status_loading:"Bağlanıyor...", status_err:"Çevrimdışı",
    step_1:"Görüntü Yükle", step_2:"Hasta Bilgisi", step_3:"YZ Raporu",

    upload_head:"Lezyon Görüntüsü Yükle",
    upload_subhead:"Net, iyi aydınlatılmış bir dermoskopik veya klinik lezyon görüntüsü sağlayın.",
    upload_drag:"Görüntüyü Sürükleyip Bırakın", upload_browse:"veya bilgisayardan seçin",
    sample_text:"Veya hazır klinik örneklerle test edin:",
    btn_continue:"Devam Et →",

    ctx_head:"Hasta Bilgisi ve Klinik Bağlam",
    ctx_subhead:"Daha doğru tanı için hasta bilgilerini ve lezyon özelliklerini girin. Bu bilgiler yapay zekanın çok daha iyi çalışmasını sağlar.",
    step2_notice_head:"Neden bilgi girmeliyim?",
    step2_notice_body:"Hastanın yaşı, cinsiyeti ve lezyonun anatomik konumu; YZ sisteminin ilgili tıbbi literatürü daha doğru aramasını ve daha isabetli tanı üretmesini sağlar. Girdiğiniz bilgiler yalnızca tarayıcınızda saklanır, sunucuya kaydedilmez.",
    lbl_age:"Hasta Yaşı", ph_age:"örn. 45",
    lbl_gender:"Cinsiyet", opt_select:"Seçiniz...", opt_f:"Kadın", opt_m:"Erkek", opt_o:"Diğer",
    lbl_location:"Anatomik Bölge", ph_loc:"örn. Sırt üst kısmı, sol ön kol",
    lbl_duration:"Lezyonun Süresi ve Değişimi", ph_dur:"örn. 3 aydır yavaşça büyüyor",
    lbl_symp:"Klinik Notlar ve Belirtiler", ph_symp:"örn. Kaşıntı var, kenarları düzensiz, ara sıra kanama",
    lbl_skintype:"Cilt Tipi (Fitzpatrick)",
    opt_skin1:"Tip I — Çok açık, her zaman yanar", opt_skin2:"Tip II — Açık, genellikle yanar",
    opt_skin3:"Tip III — Orta, bazen yanar", opt_skin4:"Tip IV — Esmer, nadiren yanar",
    opt_skin5:"Tip V — Koyu esmer, çok nadir yanar", opt_skin6:"Tip VI — Çok koyu, neredeyse hiç yanmaz",
    lbl_allergy:"Bilinen Alerjiler", ph_allergy:"örn. Penisilin, lateks, nikel",
    lbl_meds:"Kullanılan İlaçlar", ph_meds:"örn. Bağışıklık baskılayıcı, kortizon, NSAI",
    lbl_famhist:"Aile Öyküsü", opt_famhist_0:"Yok / Bilinmiyor",
    opt_famhist_1:"Ailede melanom geçmişi var",
    opt_famhist_2:"Ailede bazal hücreli karsinom (BHK) geçmişi var",
    opt_famhist_3:"Ailede skuamöz hücreli karsinom geçmişi var",
    opt_famhist_4:"Ailede birden fazla cilt kanseri vakası var",
    opt_famhist_5:"Ailede psoriasis (sedef hastalığı) var",
    opt_famhist_6:"Ailede atopik dermatit (egzama) var",
    opt_famhist_7:"Ailede genetik cilt hastalığı var",
    opt_famhist_8:"Ailede bağışıklık sistemi hastalığı var",
    opt_famhist_9:"Ailede diğer cilt kanseri geçmişi var",
    remove_image:"Fotoğrafı Kaldır",
    btn_back:"← Geri", btn_analyze:"Yapay Zeka Analizini Başlat",
    quick_fill_btn:"Son Hastayı Yükle",
    clinical_required:"Hasta Bilgisi Gerekli",
    clinical_required_msg:"Daha doğru bir tanı için lütfen en az bir alan doldurun (yaş, anatomik bölge, süre veya belirtiler). Klinik bağlam olmadan YZ sistemi tıbbi literatürde doğru arama yapamaz.",
    unsupported_head:"Desteklenmeyen Durumlar:",
    unsupported_body:"Sivilce, egzama, mantar enfeksiyonu, sedef hastalığı gibi deri sorunları bu sistem tarafından desteklenmemektedir. Yalnızca yukarıdaki 9 dermatolojik lezyon sınıfı analiz edilebilir.",

    load_title:"Analiz Yapılıyor...", load_elapsed:"Geçen süre:", load_est:"· tahmini ~20s",
    t_vision:"Görüntü Modeli Analizi", t_rag:"PubMed Literatür Taraması", t_llm:"Gemini Tanı Motoru",

    ood_strong:"Görüntü Tanınamadı:",
    ood_msg:" Yüklenen görüntü desteklenen 9 deri lezyonu sınıfından hiçbirine yeterli güvenle eşleşmedi. Yapay zeka tahmini yapılamadı.",
    mal_strong:"Malignite (Kanser) Uyarısı:",
    mal_msg:" Model, kötü huylu bir lezyon olma olasılığını yüksek buluyor. Lütfen bir dermatoloji uzmanına başvurun ve biyopsi değerlendirmesi yaptırın.",
    low_conf_strong:"Düşük Güven Uyarısı:",
    low_conf_msg:" Model bu görüntüden emin değil (%45–65 arası güven skoru). Sonuçlar yalnızca referans niteliğindedir; klinik değerlendirme yapılması önerilir.",

    card_primary:"Birincil YZ Tahmini", conf_text:"Güven Skoru",
    card_image:"Analiz Edilen Görüntü", card_report:"Tanı Raporu (Gemini 2.5)",
    card_probs:"Ayırıcı Olasılıklar", card_lit:"Destekleyici Literatür (RAG)",
    btn_restart:"← Yeni Analiz", open_article:"Makaleyi Aç →",
    err_select:"Lütfen önce bir görüntü seçin.", no_lit:"Destekleyici literatür bulunamadı.",

    abcde_btn_title:"ABCDE Dermoskopi Kılavuzu",
    abcde_guide_title:"ABCDE Dermoskopi Kılavuzu",
    abcde_a_title:"Asimetri",  abcde_a_desc:"Lezyonun iki yarısı birbirine benzemiyor",
    abcde_b_title:"Sınır",     abcde_b_desc:"Kenarlar düzensiz, pürüzlü veya çentikli",
    abcde_c_title:"Renk",      abcde_c_desc:"Birden fazla renk tonu veya düzensiz dağılım",
    abcde_d_title:"Çap",       abcde_d_desc:"6mm'den büyük (kurşun kalem silgisi büyüklüğü)",
    abcde_e_title:"Evrim",     abcde_e_desc:"Boyut, şekil veya renkte zaman içinde değişim",

    sim_title:"Benzer Geçmiş Vakalar", sim_none:"Benzer kayıtlı vaka yok.",
    hist_title:"Hasta Geçmişi", hist_sub:"Geçmiş analizler tarayıcınızda yerel olarak saklanır.",
    hist_empty:"Henüz analiz yapılmadı. İlk analizinizi çalıştırın.",
    hist_view_btn:"Görüntüle", hist_del_btn:"Sil", hist_clear_btn:"Geçmişi Temizle", hist_no_image:"Görüntü yok",

    set_title:"Ayarlar", set_sub:"Uygulama tercihlerinizi buradan yönetin.",
    set_lang_label:"Arayüz Dili", set_theme_label:"Tema",
    theme_dark:"Koyu", theme_light:"Açık",
    set_hist_label:"Analiz Geçmişi", set_hist_analyses:"kayıtlı analiz",
    set_clear_btn:"Tümünü Temizle", set_tour_label:"Kullanım Öğreticisi", set_tour_btn:"Öğreticiyi Yeniden Başlat",
    set_about_label:"DermatoRAG Hakkında",
    set_about_text:"DermatoRAG v1.0 — YZ Destekli Dermatolojik Tanı Destek Sistemi",
    set_model_text:"Vision: BiomedCLIP fine-tuned + TTA | Top-1: %73 · İlk 3'te: ~%87 | LLM: Gemini 2.5 Flash | RAG: 706 PubMed makalesi",
    set_about_note:"⚠️ Bu sistem yalnızca tanı desteği amacıyla kullanılmalıdır. Yapay zeka çıktısı, yetkili bir dermatoloji uzmanının klinik muayenesinin yerini tutamaz.",

    tour_skip:"Geç", tour_next:"İleri →", tour_finish:"Başla!",
    shortcuts_title:"Klavye Kısayolları",
    sh_continue:"Görüntü seçiliyken devam et",
    sh_analyze:"Analizi başlat",
    sh_close:"Modal / paneli kapat",
    sh_shortcuts:"Bu paneli aç/kapat",

    pdf_preparing:"PDF hazırlanıyor...", pdf_done:"PDF indirildi!", pdf_error:"PDF oluşturulamadı.",
    copied:"Panoya kopyalandı!",
    modal_ok:"Tamam", modal_cancel:"İptal",
    sc_label:"Desteklenen 9 Sınıf:",
  }
};

/* ── CLASS DISPLAY — Full names with lay terms ── */
const CLASS_DISPLAY = {
  en: {
    melanoma:               "Melanoma",
    basal_cell_carcinoma:   "Basal Cell Carcinoma (BCC)",
    squamous_cell_carcinoma:"Squamous Cell Carcinoma (SCC)",
    actinic_keratosis:      "Actinic Keratosis (Sun Damage)",
    benign_keratosis:       "Benign Keratosis",
    dermatofibroma:         "Dermatofibroma (Skin Nodule)",
    nevus:                  "Nevus (Mole)",
    seborrheic_keratosis:   "Seborrheic Keratosis (Waxy Wart)",
    vascular_lesion:        "Vascular Lesion (Blood Spot)"
  },
  tr: {
    melanoma:               "Melanom (Deri Kanseri)",
    basal_cell_carcinoma:   "Bazal Hücreli Karsinom (BHK)",
    squamous_cell_carcinoma:"Skuamöz Hücreli Karsinom (SHK)",
    actinic_keratosis:      "Aktinik Keratoz (Güneş Hasarı)",
    benign_keratosis:       "Benign Keratoz (İyi Huylu Siğil)",
    dermatofibroma:         "Dermatofibrom (Sert Cilt Nodülü)",
    nevus:                  "Nevus (Ben / Köstebek)",
    seborrheic_keratosis:   "Seboreik Keratoz (Yağlı Siğil)",
    vascular_lesion:        "Vasküler Lezyon (Damar Lekesi)"
  }
};

/* ── CLASS SHORT — Abbreviated names for probability bars ── */
const CLASS_SHORT = {
  en: {
    melanoma:               "Melanoma",
    basal_cell_carcinoma:   "Basal Cell Ca.",
    squamous_cell_carcinoma:"Squamous Cell Ca.",
    actinic_keratosis:      "Actinic Keratosis",
    benign_keratosis:       "Benign Keratosis",
    dermatofibroma:         "Dermatofibroma",
    nevus:                  "Nevus (Mole)",
    seborrheic_keratosis:   "Seborrheic Ker.",
    vascular_lesion:        "Vascular Lesion"
  },
  tr: {
    melanoma:               "Melanom",
    basal_cell_carcinoma:   "Bazal H. Karsinom",
    squamous_cell_carcinoma:"Skuamöz H. Karsinom",
    actinic_keratosis:      "Aktinik Keratoz",
    benign_keratosis:       "Benign Keratoz",
    dermatofibroma:         "Dermatofibrom",
    nevus:                  "Nevus (Ben)",
    seborrheic_keratosis:   "Seboreik Keratoz",
    vascular_lesion:        "Vasküler Lezyon"
  }
};

/* ── TOUR CONTENT ── */
const TOUR_STEPS = {
  tr: [
    { icon:"🔬", title:"DermatoRAG'a Hoş Geldiniz", body:"Bu sistem, cilt lezyonlarını yapay zeka ile analiz eder ve 9 farklı sınıf için olasılıklı ayırıcı tanılar sunar. Hem doktorlar hem de hastalar için tasarlanmıştır. Nasıl çalıştığını gösterelim." },
    { icon:"🖼️", title:"Adım 1: Görüntü Yükle", body:"Dermoskopik veya klinik lezyon fotoğrafını sürükleyip bırakın ya da dosya seçin. Sistem yalnızca 9 belirli deri lezyonu sınıfını tanır; sivilce veya egzama gibi durumlar desteklenmez." },
    { icon:"📋", title:"Adım 2: Hasta Bilgisi Girin", body:"Hastanın yaşı, anatomik bölgesi ve belirtileri yapay zekanın PubMed'den doğru literatür çekebilmesi için çok önemlidir. Bu bilgiler tarayıcınızda kalır, sunucuya gönderilmez." },
    { icon:"📊", title:"Adım 3: Raporu İnceleyin", body:"Tanı raporu, ayırıcı olasılıklar ve PubMed literatürü birlikte gösterilir. Raporda hem tıbbi terimler hem de herkesin anlayabileceği açıklamalar yer alır. PDF olarak indirebilirsiniz." },
  ],
  en: [
    { icon:"🔬", title:"Welcome to DermatoRAG", body:"This system analyzes skin lesions using AI and provides probabilistic differential diagnoses for 9 classes. Designed for both clinicians and patients. Let's learn how it works." },
    { icon:"🖼️", title:"Step 1: Upload Image", body:"Drag & drop or select a dermoscopic or clinical lesion image. The system recognizes only 9 specific skin lesion classes — acne, eczema, or other conditions are not supported." },
    { icon:"📋", title:"Step 2: Enter Patient Info", body:"Patient age, anatomical location, and symptoms are crucial for the AI to retrieve relevant PubMed literature. This data stays in your browser, never sent to a server." },
    { icon:"📊", title:"Step 3: Review the Report", body:"The diagnostic report, differential probabilities, and PubMed literature are shown together. Reports include both medical terms and plain-language explanations. Download as PDF." },
  ]
};

/* ── LLM ERROR MESSAGES ── */
const LLM_ERRORS = {
  tr: {
    'LLM_ERROR:QUOTA':   '⚠️ Gemini API günlük ücretsiz kotası doldu (20 istek/gün). Birkaç saat sonra tekrar deneyin veya Google AI Studio\'dan API planınızı yükseltin.',
    'LLM_ERROR:AUTH':    '⚠️ API anahtarı geçersiz veya bulunamadı. .env dosyasındaki GOOGLE_API_KEY değerini kontrol edin.',
    'LLM_ERROR:TIMEOUT': '⚠️ Bağlantı zaman aşımına uğradı. İnternet bağlantınızı kontrol edip tekrar deneyin.',
    'LLM_ERROR:UNKNOWN': '⚠️ Tanı raporu üretilemedi. Görüntü analizi tamamlandı ancak LLM yanıt vermedi. Sunucu loglarını kontrol edin.',
  },
  en: {
    'LLM_ERROR:QUOTA':   '⚠️ Gemini API free quota exceeded (20 req/day). Try again in a few hours or upgrade your plan in Google AI Studio.',
    'LLM_ERROR:AUTH':    '⚠️ Invalid or missing API key. Check the GOOGLE_API_KEY value in your .env file.',
    'LLM_ERROR:TIMEOUT': '⚠️ Connection timed out. Check your internet connection and retry.',
    'LLM_ERROR:UNKNOWN': '⚠️ Could not generate diagnostic report. Image analysis completed but LLM did not respond. Check server logs.',
  }
};

/* ── STATE ── */
let currentLang        = localStorage.getItem('dermato_lang') || 'tr';
let currentTheme       = localStorage.getItem('dermato_theme') || 'dark';
let currentFile        = null;
let currentFileDataUrl = null;
let donutChart         = null;
let currentView        = 'analysis';
let viewingHistory     = false;
let abcdeVisible       = false;
let tourStep           = 0;
let loadingTimer       = null;
let loadingSeconds     = 0;
let analysisHistory    = [];
let currentTopClass    = null; // for similar cases refresh

try { analysisHistory = JSON.parse(localStorage.getItem('dermato_history') || '[]'); } catch { analysisHistory = []; }

/* ── INIT ── */
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(currentTheme);

  // sessionStorage: survives F5 refresh (same tab) but clears on browser close/new tab
  // → F5 = stays on app; fresh open = always shows landing
  if (sessionStorage.getItem('dermato_launched')) {
    document.getElementById('landing-view').classList.remove('active');
    document.getElementById('landing-view').classList.add('hidden');
    document.getElementById('app-view').classList.remove('hidden');
    document.getElementById('app-view').classList.add('active');
  } else {
    // Always show landing on fresh open; clear any stale localStorage flag
    localStorage.removeItem('dermato_launched');
  }

  setLanguage(currentLang, false);
  setupDropZone();
  setupKeyboardShortcuts();
  document.getElementById('file-input').addEventListener('change', handleFileSelect);
  updateHistoryBadge();
  checkHealth();
  setInterval(checkHealth, 30000);

  const resImg = document.getElementById('res-img');
  resImg.addEventListener('mousemove', function(e) {
    const { left, top, width, height } = this.getBoundingClientRect();
    this.style.transformOrigin = `${((e.clientX-left)/width)*100}% ${((e.clientY-top)/height)*100}%`;
  });
  resImg.addEventListener('mouseleave', function() { this.style.transformOrigin = 'center'; });

  renderSupportedClasses();

  // Auto-save form data on any field change/input (so "Load Last Patient" always works)
  const FORM_IDS = ['inp-age','inp-gender','inp-loc','inp-dur','inp-symp','inp-skintype','inp-allergy','inp-meds','inp-famhist'];
  FORM_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', autoSaveForm);
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      el.addEventListener('input', autoSaveForm);
    }
  });

  // Auto-open tour on landing page — first visit only
  if (!localStorage.getItem('dermato_tour_done') && !localStorage.getItem('dermato_launched')) {
    setTimeout(() => startOnboarding(false), 1200);
  }
});

function autoSaveForm() {
  saveFormData({
    age:      document.getElementById('inp-age')?.value || '',
    gender:   document.getElementById('inp-gender')?.value || '',
    loc:      document.getElementById('inp-loc')?.value || '',
    dur:      document.getElementById('inp-dur')?.value || '',
    symp:     document.getElementById('inp-symp')?.value || '',
    skintype: document.getElementById('inp-skintype')?.value || '',
    allergy:  document.getElementById('inp-allergy')?.value || '',
    meds:     document.getElementById('inp-meds')?.value || '',
    famhist:  document.getElementById('inp-famhist')?.value || '',
  });
}

/* ── TRANSLATION ── */
function t(key) { return DICT[currentLang]?.[key] || key; }

function setLanguage(lang, runHealthCheck = true) {
  currentLang = lang;
  localStorage.setItem('dermato_lang', lang);

  document.querySelectorAll('.lang-btn').forEach(btn => {
    const bl = btn.dataset.lang || btn.textContent.trim().toLowerCase();
    btn.classList.toggle('active', bl === lang);
  });

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const val = DICT[lang]?.[el.getAttribute('data-i18n')];
    if (!val) return;
    if ([...el.childNodes].some(n => n.nodeType === 1)) {
      for (const node of el.childNodes) {
        if (node.nodeType === 3 && node.textContent.trim()) { node.textContent = val + ' '; return; }
      }
    } else {
      el.textContent = val;
    }
  });

  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const val = DICT[lang]?.[el.getAttribute('data-i18n-placeholder')];
    if (val) el.placeholder = val;
  });

  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    const val = DICT[lang]?.[el.getAttribute('data-i18n-title')];
    if (val) el.title = val;
  });

  document.querySelectorAll('option[data-i18n]').forEach(el => {
    const val = DICT[lang]?.[el.getAttribute('data-i18n')];
    if (val) el.textContent = val;
  });

  if (currentView === 'history') renderHistory();
  if (currentView === 'settings') renderSettings();
  if (runHealthCheck) checkHealth();
  renderSupportedClasses();
}

/* ── THEME ── */
function applyTheme(theme) {
  currentTheme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('dermato_theme', theme);
  const icon = theme === 'dark' ? '☀️' : '🌙';
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.textContent = icon;
  const mbtn = document.getElementById('mobile-theme-toggle');
  if (mbtn) mbtn.textContent = icon;
  document.querySelectorAll('.theme-opt').forEach(b => b.classList.remove('active'));
  const activeBtn = document.getElementById(theme === 'dark' ? 'theme-dark-btn' : 'theme-light-btn');
  if (activeBtn) activeBtn.classList.add('active');
}

function toggleTheme() { applyTheme(currentTheme === 'dark' ? 'light' : 'dark'); }
function setTheme(theme) { applyTheme(theme); }

/* ── MODAL ── */
function showModal(title, message, type = 'info', onConfirm = null) {
  const icons = { info:'ℹ️', error:'❌', warning:'⚠️', success:'✅' };
  document.getElementById('modal-icon').textContent  = icons[type] || 'ℹ️';
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-body').innerHTML    = message;

  const actions = document.getElementById('modal-actions');
  if (onConfirm) {
    actions.innerHTML = `
      <button class="btn-secondary" onclick="closeModal()">${t('modal_cancel')}</button>
      <button class="btn-primary btn-danger-confirm" onclick="(${onConfirm.toString()})(); closeModal();">${t('modal_ok')}</button>`;
  } else {
    actions.innerHTML = `<button class="btn-primary" onclick="closeModal()">${t('modal_ok')}</button>`;
  }

  const modal = document.getElementById('app-modal');
  modal.classList.remove('hidden');
  requestAnimationFrame(() => modal.classList.add('show'));
}

function closeModal() {
  const modal = document.getElementById('app-modal');
  modal.classList.remove('show');
  setTimeout(() => modal.classList.add('hidden'), 280);
}

function handleModalBackdrop(e) {
  if (e.target === document.getElementById('app-modal')) closeModal();
}

/* ── TOAST ── */
function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toast-container');
  const id  = 'toast-' + Date.now();
  const col = { info:'var(--accent-primary)', success:'var(--color-success)', error:'var(--color-err)', warning:'var(--color-warn)' }[type] || 'var(--accent-primary)';
  const ico = { info:'ℹ️', success:'✅', error:'❌', warning:'⚠️' }[type] || 'ℹ️';
  const el  = document.createElement('div');
  el.className = 'toast';
  el.id = id;
  el.style.borderLeftColor = col;
  el.innerHTML = `<span class="toast-icon">${ico}</span><span class="toast-msg">${message}</span><button class="toast-close" onclick="removeToast('${id}')">×</button>`;
  container.appendChild(el);
  requestAnimationFrame(() => { requestAnimationFrame(() => el.classList.add('show')); });
  setTimeout(() => removeToast(id), duration);
}

function removeToast(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('show');
  setTimeout(() => el.remove(), 320);
}

/* ── SUPPORTED CLASSES PANEL ── */
const SC_MALIGNANT = new Set(['melanoma','basal_cell_carcinoma','squamous_cell_carcinoma']);
const SC_PRECANCER = new Set(['actinic_keratosis']);

function renderSupportedClasses() {
  const container = document.getElementById('supported-chips');
  if (!container) return;
  const names = CLASS_DISPLAY[currentLang] || CLASS_DISPLAY.tr;
  container.innerHTML = Object.entries(names).map(([key, label]) => {
    const cls = SC_MALIGNANT.has(key) ? 'malignant' : SC_PRECANCER.has(key) ? 'precancer' : 'benign';
    return `<span class="sc-chip sc-${cls}" title="${key}">${label}</span>`;
  }).join('');
}

/* ── MOBILE SIDEBAR ── */
function toggleSidebar() {
  const sidebar  = document.querySelector('.sidebar');
  const backdrop = document.getElementById('sidebar-backdrop');
  if (!sidebar) return;
  const isOpen = sidebar.classList.contains('open');
  if (isOpen) { closeSidebar(); } else {
    sidebar.classList.add('open');
    backdrop?.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }
}

function closeSidebar() {
  document.querySelector('.sidebar')?.classList.remove('open');
  document.getElementById('sidebar-backdrop')?.classList.add('hidden');
  document.body.style.overflow = '';
}

/* ── PAGE TRANSITION UTILITY ── */
function fadeOut(el, duration = 280) {
  return new Promise(resolve => {
    el.style.transition = `opacity ${duration}ms ease, transform ${duration}ms ease`;
    el.style.opacity = '0';
    el.style.transform = 'translateY(-10px)';
    setTimeout(resolve, duration);
  });
}

function fadeIn(el, duration = 320) {
  el.style.opacity = '0';
  el.style.transform = 'translateY(10px)';
  el.style.transition = `opacity ${duration}ms ease, transform ${duration}ms ease`;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
      setTimeout(() => {
        el.style.transition = '';
        el.style.opacity = '';
        el.style.transform = '';
      }, duration + 50);
    });
  });
}

/* ── NAVIGATION ── */
function goToLanding() {
  sessionStorage.removeItem('dermato_launched');
  const app = document.getElementById('app-view');
  const landing = document.getElementById('landing-view');
  fadeOut(app).then(() => {
    app.classList.remove('active');
    app.classList.add('hidden');
    app.style.cssText = '';
    landing.classList.remove('hidden');
    landing.classList.add('active');
    fadeIn(landing);
    resetSystem();
  });
}

function launchApp() {
  sessionStorage.setItem('dermato_launched', '1');
  const landing = document.getElementById('landing-view');
  const app = document.getElementById('app-view');
  fadeOut(landing).then(() => {
    landing.classList.remove('active');
    landing.classList.add('hidden');
    landing.style.cssText = '';
    app.classList.remove('hidden');
    app.classList.add('active');
    fadeIn(app);
    showView('analysis');
  });
}

function launchAndTour() {
  launchApp();
  setTimeout(() => startOnboarding(true), 800);
}

/* Yeni Analiz — her zaman step-1'e dönüp formu sıfırla */
function newAnalysis() {
  resetSystem();
  showView('analysis');
}

let _viewTransitioning = false;
async function showView(view) {
  if (_viewTransitioning && currentView !== view) {
    // Force-unlock if stuck more than 1s
    _viewTransitioning = false;
  }
  if (currentView === view) { closeSidebar(); return; }
  _viewTransitioning = true;

  try {
    const curEl = document.getElementById('view-' + currentView);
    if (curEl && !curEl.classList.contains('hidden')) {
      await fadeOut(curEl, 200);
      curEl.classList.add('hidden');
      curEl.style.cssText = '';
    }

    currentView = view;
    ['analysis','history','settings'].forEach(v => {
      document.getElementById('view-' + v)?.classList.add('hidden');
    });
    document.querySelectorAll('.menu-item').forEach(el => el.classList.remove('active'));
    document.getElementById('nav-' + view)?.classList.add('active');
    if (view === 'history') renderHistory();
    if (view === 'settings') renderSettings();

    const newEl = document.getElementById('view-' + view);
    newEl.classList.remove('hidden');
    newEl.style.opacity = '0';
    await fadeIn(newEl, 260);
    closeSidebar();
  } finally {
    _viewTransitioning = false;
  }
}

async function goToStep(step) {
  if (step === 2 && !currentFile) { showModal(t('err_select'), t('err_select'), 'warning'); return; }

  // Fade out current visible step
  const current = document.querySelector('.wizard-step:not(.hidden)');
  if (current) {
    await fadeOut(current, 180);
    document.querySelectorAll('.wizard-step').forEach(s => { s.classList.add('hidden'); s.style.cssText = ''; });
  } else {
    document.querySelectorAll('.wizard-step').forEach(s => s.classList.add('hidden'));
  }

  const target = document.getElementById('step-' + step);
  target.classList.remove('hidden');
  fadeIn(target, 240);

  document.querySelectorAll('.track-step').forEach((el, i) => {
    el.classList.remove('active','done');
    if (i + 1 < step) el.classList.add('done');
    if (i + 1 === step) el.classList.add('active');
  });

  // Scroll to top and auto-focus first required field on step 2
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (step === 2) {
    setTimeout(() => {
      const ageEl = document.getElementById('inp-age');
      if (ageEl && !ageEl.value) ageEl.focus();
    }, 260);
  }
}

/* ── HEALTH CHECK ── */
async function checkHealth() {
  try {
    const data = await fetch('/health').then(r => r.json());
    const ind  = document.querySelector('.health-indicator');
    if (!ind) return;
    const [dot, span] = [ind.querySelector('.dot'), ind.querySelector('span')];
    if (data.pipeline_ready) {
      dot.style.cssText  = 'background:var(--color-success);box-shadow:0 0 10px var(--color-success)';
      span.innerText = t('status_online');
    } else {
      dot.style.cssText  = 'background:var(--color-warn);box-shadow:0 0 10px var(--color-warn)';
      span.innerText = t('status_loading');
    }
  } catch {
    const ind = document.querySelector('.health-indicator');
    if (ind) { ind.querySelector('.dot').style.cssText = 'background:var(--color-err)'; ind.querySelector('span').innerText = t('status_err'); }
  }
}

/* ── UPLOAD ── */
function setupDropZone() {
  const zone = document.getElementById('drop-zone');
  zone.addEventListener('click', () => document.getElementById('file-input').click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('dragover'); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
}

function handleFileSelect(e) { if (e.target.files[0]) setFile(e.target.files[0]); }

function setFile(file) {
  currentFile = file; viewingHistory = false;
  const reader = new FileReader();
  reader.onload = ev => {
    currentFileDataUrl = ev.target.result;
    document.getElementById('preview-img').src = ev.target.result;
    document.getElementById('preview-img').classList.remove('hidden');
    document.getElementById('drop-content').classList.add('hidden');
    // Show remove button
    const rb = document.getElementById('remove-img-btn');
    if (rb) rb.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

function removeFile() {
  currentFile = null; currentFileDataUrl = null;
  const p = document.getElementById('preview-img');
  if (p) { p.src = ''; p.classList.add('hidden'); }
  document.getElementById('drop-content')?.classList.remove('hidden');
  document.getElementById('remove-img-btn')?.classList.add('hidden');
  document.getElementById('file-input').value = '';
  showToast(currentLang==='tr' ? 'Görüntü kaldırıldı.' : 'Image removed.', 'info', 1800);
}

function loadSample(type) {
  fetch('/static/samples/' + type + '.jpg')
    .then(r => { if (!r.ok) throw new Error('Sample not found'); return r.blob(); })
    .then(blob => setFile(new File([blob], type + '.jpg', { type:'image/jpeg' })))
    .catch(() => showToast(currentLang==='tr' ? 'Örnek görüntü yüklenemedi.' : 'Could not load sample image.', 'error'));
}

/* ── LOADING TIMER ── */
function startLoadingTimer() {
  loadingSeconds = 0;
  const el = document.getElementById('loading-timer');
  if (el) el.textContent = '0s';
  loadingTimer = setInterval(() => {
    loadingSeconds++;
    if (el) el.textContent = loadingSeconds + 's';
  }, 1000);
}

function stopLoadingTimer() {
  if (loadingTimer) { clearInterval(loadingTimer); loadingTimer = null; }
}

/* ── ANALYSIS ── */
let _analyzing = false;
async function startAnalysis() {
  if (_analyzing) return; // Çift tıklama / hızlı klavye kısayolu koruması
  const age      = document.getElementById('inp-age').value.trim();
  const gender   = document.getElementById('inp-gender').value;
  const loc      = document.getElementById('inp-loc').value.trim();
  const dur      = document.getElementById('inp-dur').value.trim();
  const symp     = document.getElementById('inp-symp').value.trim();
  const skintype = document.getElementById('inp-skintype')?.value || '';
  const allergy  = document.getElementById('inp-allergy')?.value.trim() || '';
  const meds     = document.getElementById('inp-meds')?.value.trim() || '';
  const famhist  = document.getElementById('inp-famhist')?.value || '';

  // Require the 3 most diagnostically important fields
  const missing = [];
  if (!age)    missing.push(currentLang==='tr' ? '"Hasta Yaşı"' : '"Patient Age"');
  if (!gender) missing.push(currentLang==='tr' ? '"Cinsiyet"'   : '"Gender"');
  if (!loc)    missing.push(currentLang==='tr' ? '"Anatomik Bölge"' : '"Anatomical Location"');

  if (missing.length > 0) {
    const msg = currentLang==='tr'
      ? `Lütfen şu zorunlu alanları doldurun: <strong>${missing.join(', ')}</strong>.<br><br>Bu bilgiler yapay zekanın tıbbi literatürde daha doğru arama yapması ve isabetli tanı üretmesi için gereklidir.`
      : `Please fill in the required fields: <strong>${missing.join(', ')}</strong>.<br><br>These are essential for the AI to search relevant medical literature accurately.`;
    showModal(t('clinical_required'), msg, 'warning');
    return;
  }

  _analyzing = true;
  const btn = document.querySelector('.analyze-btn');
  if (btn) { btn.disabled = true; btn.style.opacity = '0.7'; btn.style.cursor = 'wait'; }

  await showView('analysis');
  await goToStep(3);
  document.getElementById('loader-view').classList.remove('hidden');
  document.getElementById('results-view').classList.add('hidden');
  startLoadingTimer();

  const tv = document.getElementById('t-vision');
  const tr = document.getElementById('t-rag');
  const tl = document.getElementById('t-llm');
  tv.className='task active'; tv.querySelector('.t-icon').innerText='⏳';
  tr.className='task';        tr.querySelector('.t-icon').innerText='⏳';
  tl.className='task';        tl.querySelector('.t-icon').innerText='⏳';

  const arr = [];
  if (age)      arr.push((currentLang==='tr'?'Yaş: ':'Age: ') + age);
  if (gender)   arr.push((currentLang==='tr'?'Cinsiyet: ':'Gender: ') + gender);
  if (loc)      arr.push((currentLang==='tr'?'Bölge: ':'Location: ') + loc);
  if (skintype) arr.push((currentLang==='tr'?'Cilt Tipi: ':'Skin Type: ') + skintype);
  if (dur)      arr.push((currentLang==='tr'?'Süre/Değişim: ':'Duration/Change: ') + dur);
  if (symp)     arr.push((currentLang==='tr'?'Klinik Notlar: ':'Clinical Notes: ') + symp);
  if (allergy)  arr.push((currentLang==='tr'?'Alerjiler: ':'Allergies: ') + allergy);
  if (meds)     arr.push((currentLang==='tr'?'İlaçlar: ':'Medications: ') + meds);
  if (famhist && famhist !== 'none') arr.push((currentLang==='tr'?'Aile Öyküsü: ':'Family History: ') + famhist);

  saveFormData({ age, gender, loc, dur, symp, skintype, allergy, meds, famhist });

  try {
    const fd = new FormData();
    fd.append('image', currentFile);
    fd.append('clinical_info', arr.join(' | '));
    fd.append('language', currentLang);

    setTimeout(()=>{ tv.className='task done'; tv.querySelector('.t-icon').innerText='✓'; tr.className='task active'; }, 1500);
    setTimeout(()=>{ tr.className='task done'; tr.querySelector('.t-icon').innerText='✓'; tl.className='task active'; }, 3500);

    const r    = await fetch('/analyze', { method:'POST', body:fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Analysis failed');

    tl.className='task done'; tl.querySelector('.t-icon').innerText='✓';
    stopLoadingTimer();

    setTimeout(() => {
      document.getElementById('loader-view').classList.add('hidden');
      saveToHistory(data);
      renderResults(data, currentFileDataUrl);
      showToast(
        (currentLang==='tr' ? 'Analiz tamamlandı — ' : 'Analysis done — ') + (loadingSeconds + 1) + 's',
        'success', 2500
      );
    }, 700);

  } catch(err) {
    stopLoadingTimer();
    document.getElementById('loader-view').classList.add('hidden');
    goToStep(2);
    showModal(currentLang==='tr' ? 'Analiz Hatası' : 'Analysis Error', err.message, 'error');
  } finally {
    // Lock'u serbest bırak (renderResults setTimeout 700ms sonra koşar, ona dokunma)
    setTimeout(() => {
      _analyzing = false;
      const btn = document.querySelector('.analyze-btn');
      if (btn) { btn.disabled = false; btn.style.opacity = ''; btn.style.cursor = ''; }
    }, 800);
  }
}

/* ── RENDER RESULTS ── */
function renderResults(data, imageUrl) {
  const vis = data.vision;
  currentTopClass = vis.top_class; // track for similar-cases refresh
  document.getElementById('results-view').classList.remove('hidden');

  if (imageUrl) { document.getElementById('res-img').src = imageUrl; }
  else { document.getElementById('res-img').src=''; document.getElementById('res-img').alt=t('hist_no_image'); }

  document.getElementById('ood-alert').classList.add('hidden');
  document.getElementById('mal-alert').classList.add('hidden');
  document.getElementById('low-conf-alert')?.classList.add('hidden');

  const conf = vis.confidence; // 0-100 scale from backend

  if (vis.is_ood) {
    document.getElementById('ood-alert').classList.remove('hidden');
    // Render a special OOD report
    const closestGuess = CLASS_SHORT[currentLang]?.[vis.top_class] || vis.top_class;
    const oodHtml = currentLang === 'tr'
      ? `<div class="ood-report">
          <p>⚠️ Yüklenen görüntü, sistemin tanıyabildiği <strong>9 dermatolojik sınıfın</strong> hiçbirine yeterli güvenle eşleşmedi (en yüksek güven skoru: <strong>%${conf}</strong>).</p>
          <p>Modelin en yakın tahmini: <strong>${closestGuess}</strong> — ancak bu tahmin düşük güven nedeniyle geçerli sayılamaz.</p>
          <br>
          <p><strong>Bu durum şu sebeplerden kaynaklanabilir:</strong></p>
          <ul>
            <li>Görüntü, sistem tarafından desteklenmeyen bir deri rahatsızlığı içeriyor olabilir (sivilce, egzama, sedef hastalığı, mantar, vb.)</li>
            <li>Fotoğraf kalitesi, aydınlatma veya açısı analiz için uygun değil</li>
            <li>Görüntü bir deri lezyonu içermiyor (el, yüz, dudak, tırnak vb.)</li>
            <li>Lezyonun sınıfı, modelin eğitildiği 9 sınıfın dışında kalıyor</li>
          </ul>
          <br>
          <p>🩺 <strong>Lütfen bir dermatoloji uzmanına başvurun.</strong> Bu sistem yalnızca 9 belirli dermatolojik sınıf için tasarlanmıştır ve diğer durumları değerlendiremez.</p>
        </div>`
      : `<div class="ood-report">
          <p>⚠️ The uploaded image did not match any of the system's <strong>9 supported skin lesion classes</strong> with sufficient confidence (highest score: <strong>${conf}%</strong>).</p>
          <p>Closest model guess: <strong>${closestGuess}</strong> — however this prediction cannot be considered valid due to low confidence.</p>
          <br>
          <p><strong>Possible reasons:</strong></p>
          <ul>
            <li>The image may contain an unsupported skin condition (acne, eczema, psoriasis, fungal infection, etc.)</li>
            <li>Photo quality, lighting, or angle is not suitable for analysis</li>
            <li>The image does not contain a skin lesion (hand, face, lips, nail, etc.)</li>
            <li>The lesion class falls outside the 9 trained categories</li>
          </ul>
          <br>
          <p>🩺 <strong>Please consult a dermatologist.</strong> This system is designed only for 9 specific dermatological classes and cannot evaluate other conditions.</p>
        </div>`;
    document.getElementById('res-report').innerHTML = oodHtml;
  } else if (vis.is_malignant) {
    document.getElementById('mal-alert').classList.remove('hidden');
  } else if (conf >= 45 && conf < 65) {
    document.getElementById('low-conf-alert')?.classList.remove('hidden');
  }

  const cName = CLASS_DISPLAY[currentLang]?.[vis.top_class] || vis.top_class_display || vis.top_class;
  document.getElementById('res-top-class').innerText = cName;
  document.getElementById('res-conf').innerText = vis.confidence;

  // Risk badge
  const rb = document.getElementById('risk-badge');
  if (rb) {
    rb.classList.remove('hidden','risk-low','risk-warn','risk-high','risk-ood');
    if (vis.is_ood) {
      rb.className = 'risk-badge risk-ood';
      rb.textContent = currentLang==='tr' ? '⚠️ Tanınamadı' : '⚠️ Unrecognized';
    } else if (vis.is_malignant) {
      rb.className = 'risk-badge risk-high';
      rb.textContent = currentLang==='tr' ? '🔴 Yüksek Risk' : '🔴 High Risk';
    } else if (conf >= 45 && conf < 65) {
      rb.className = 'risk-badge risk-warn';
      rb.textContent = currentLang==='tr' ? '🟡 Düşük Güven' : '🟡 Low Confidence';
    } else {
      rb.className = 'risk-badge risk-low';
      rb.textContent = currentLang==='tr' ? '🟢 Düşük Risk' : '🟢 Low Risk';
    }
  }

  // Timestamp on report card
  const ts = document.getElementById('report-timestamp');
  if (ts) ts.textContent = new Date().toLocaleString(currentLang==='tr'?'tr-TR':'en-US', { day:'2-digit', month:'short', year:'numeric', hour:'2-digit', minute:'2-digit' });

  renderDonut(vis.confidence, vis.is_malignant, vis.is_ood);
  renderProbs(vis.all_probs);
  if (!vis.is_ood) renderReport(data.diagnosis || '');
  renderLit(data.articles || []);
  renderSimilarCases(vis.top_class);
  setTimeout(makeReportCollapsible, 150);
}

/* ── REPORT FONT SIZE ── */
let _reportFontSize = 93; // percent
function adjustReportFont(delta) {
  _reportFontSize = Math.min(120, Math.max(70, _reportFontSize + delta));
  const r = document.getElementById('res-report');
  if (r) r.style.fontSize = _reportFontSize + '%';
}

function renderDonut(conf, isMal, isOod) {
  const ctx = document.getElementById('confidence-chart').getContext('2d');
  if (donutChart) donutChart.destroy();
  const col = isOod ? '#F59E0B' : (isMal ? '#EF4444' : '#10B981');
  donutChart = new Chart(ctx, {
    type:'doughnut',
    data:{ datasets:[{ data:[conf,100-conf], backgroundColor:[col,'rgba(255,255,255,0.06)'], borderWidth:0 }] },
    options:{ cutout:'78%', responsive:false, animation:{ duration:1500 }, plugins:{ tooltip:{ enabled:false } } }
  });
}

function renderProbs(probs) {
  const c = document.getElementById('res-probs');
  c.innerHTML = '';
  const MALIGNANT = ['melanoma','basal_cell_carcinoma','squamous_cell_carcinoma'];
  probs.forEach(p => {
    const shortName = CLASS_SHORT[currentLang]?.[p.class] || p.class;
    const fullName  = CLASS_DISPLAY[currentLang]?.[p.class] || p.class;
    const col = (MALIGNANT.includes(p.class) && p.probability > 15) ? 'var(--color-err)' : 'var(--accent-secondary)';
    c.innerHTML += `<div class="prob-item">
      <div class="prob-name" title="${fullName}">${shortName}</div>
      <div class="prob-bar"><div class="prob-fill" style="width:0%;background:${col}"></div></div>
      <div class="prob-val">${p.probability}%</div>
    </div>`;
  });
  setTimeout(() => {
    c.querySelectorAll('.prob-item').forEach((item,i) => { item.querySelector('.prob-fill').style.width = probs[i].probability+'%'; });
  }, 100);
}

/* ── MARKDOWN RENDER ── */
function renderReport(text) {
  if (!text) return;
  if (text.startsWith('LLM_ERROR:')) {
    const msg = LLM_ERRORS[currentLang]?.[text] || LLM_ERRORS.tr[text] || text;
    document.getElementById('res-report').innerHTML =
      `<div class="report-llm-error">${msg}</div>`;
    showToast(currentLang==='tr' ? 'Rapor üretilemedi' : 'Report generation failed', 'error', 6000);
    return;
  }
  document.getElementById('res-report').innerHTML = markdownToHtml(text);
}

// Section icon map — matches section keywords in both languages
const SECTION_ICONS = {
  'LEZYON ANALİZİ':'🔬', 'LESION ANALYSIS':'🔬',
  'AYIRİCİ TANI':'🩺', 'DIFFERENTIAL':'🩺',
  'HALK DİLİNDE':'🗣️', 'PLAIN LANGUAGE':'🗣️', 'PATIENT':'🗣️',
  'İLAÇ':'💊', 'MEDICATION':'💊', 'TRIGGER':'💊', 'TETİKLEYİCİ':'💊',
  'KLİNİK YÖNETİM':'📋', 'CLINICAL MANAGEMENT':'📋', 'NEXT STEPS':'📋',
  'DOKTORA':'🏥', 'REFERRAL':'🏥', 'ACİL':'🏥', 'EMERGENCY':'🏥',
  'SİSTEM UYARISI':'⚠️', 'SYSTEM WARNING':'⚠️', 'DISCLAIMER':'⚠️',
};

function getSectionIcon(title) {
  const up = title.toUpperCase();
  for (const [key, icon] of Object.entries(SECTION_ICONS)) {
    if (up.includes(key)) return icon;
  }
  return '📌';
}

function markdownToHtml(text) {
  if (!text) return `<p class="md-empty">${currentLang==='tr'?'Rapor üretilemedi.':'Report could not be generated.'}</p>`;
  const lines = text.split('\n');
  let html='', inList=false, inSubList=false;

  for (let i=0; i<lines.length; i++) {
    const raw  = lines[i];
    const line = raw.trim();
    if (!line) {
      if(inSubList){html+='</ul>';inSubList=false;}
      if(inList){html+='</ul>';inList=false;}
      html+='<div class="md-gap"></div>'; continue;
    }

    const cleanLine = line.replace(/^\s*#{1,3}\s*/,'').replace(/\*+/g,'').trim();

    // Section headers (numbered: 1. TITLE or ⚠️ SYSTEM...)
    const secMatch = cleanLine.match(/^(\d+)\.\s+([A-ZÇŞĞÜÖİa-zçşğüöı &\(\)\-\/🔬🩺💊📋🏥⚠️🗣️]{4,120}):?\s*$/);
    const warnMatch = cleanLine.match(/^[⚠️🏥]+\s+(.{4,80}):?\s*$/);
    if ((secMatch && cleanLine.length < 130) || warnMatch) {
      if(inSubList){html+='</ul>';inSubList=false;}
      if(inList){html+='</ul>';inList=false;}
      const num   = secMatch ? secMatch[1] : '!';
      const title = secMatch ? secMatch[2].trim() : warnMatch[1].trim();
      const icon  = getSectionIcon(title);
      const isWarn = warnMatch || title.toUpperCase().includes('UYARI') || title.toUpperCase().includes('WARNING');
      html+=`<div class="md-section${isWarn?' md-section-warn':''}">
        <span class="md-sec-icon">${icon}</span>
        <div class="md-sec-text">
          <span class="md-num">${num}</span>
          <span class="md-stitle">${title.toUpperCase()}</span>
        </div>
      </div>`;
      continue;
    }

    // Bold sub-headers (e.g., "**Tanı 1:**" or "- **Tanı 1:**")
    const boldHeader = cleanLine.match(/^[\-\*•]?\s*\*\*([^*]{3,60})\*\*\s*[:—–]?\s*$/);
    if (boldHeader) {
      if(inSubList){html+='</ul>';inSubList=false;}
      if(inList){html+='</ul>';inList=false;}
      html+=`<p class="md-subhead">${applyInline(cleanLine.replace(/^[\-\*•]\s*/,''))}</p>`;
      continue;
    }

    // Sub-list items (indented)
    if (/^\s{2,}[\-\*•]\s+/.test(raw) || /^\t[\-\*•]\s+/.test(raw)) {
      if(!inSubList){html+='<ul class="md-sublist">';inSubList=true;}
      html+=`<li>${applyInline(line.replace(/^[\-\*•]\s+/,''))}</li>`;
      continue;
    }
    if(inSubList){html+='</ul>';inSubList=false;}

    // List items
    if (/^[\-\*•]\s+/.test(line)) {
      if(!inList){html+='<ul class="md-list">';inList=true;}
      html+=`<li>${applyInline(line.replace(/^[\-\*•]\s+/,''))}</li>`;
      continue;
    }
    if(inList){html+='</ul>';inList=false;}

    // Numbered list items within content (e.g., "1. text")
    const numListMatch = line.match(/^(\d+)\.\s+(.+)$/);
    if (numListMatch && line.length < 200 && !line.match(/^(\d+)\.\s+[A-ZÇŞĞÜÖİ]{3,}/)) {
      html+=`<p class="md-numitem"><span class="md-ni-num">${numListMatch[1]}.</span> ${applyInline(numListMatch[2])}</p>`;
      continue;
    }

    html+=(/^\s{2,}/.test(raw)||raw.startsWith('\t'))
      ? `<p class="md-indent">${applyInline(line)}</p>`
      : `<p class="md-line">${applyInline(line)}</p>`;
  }
  if(inSubList)html+='</ul>';
  if(inList)html+='</ul>';
  return html;
}

function applyInline(text) {
  return text
    .replace(/\*\*\*(.*?)\*\*\*/g,'<strong><em>$1</em></strong>')
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,'<em>$1</em>')
    .replace(/\[Kaynak\s*(\d+)\]/g,'<span class="md-ref">[K$1]</span>')
    .replace(/\[Source\s*(\d+)\]/g,'<span class="md-ref">[S$1]</span>')
    // Confidence percentages: %85 or 85%
    .replace(/(%\d+|\d+%)/g,'<span class="md-pct">$1</span>')
    // ⚠️ inline warnings
    .replace(/(⚠️[^<\n]+)/g,'<span class="md-warn-inline">$1</span>')
    // 🩺 inline doctor note
    .replace(/(🩺[^<\n]+)/g,'<span class="md-doctor-note">$1</span>');
}

/* ── COLLAPSIBLE REPORT ── */
function makeReportCollapsible() {
  const report   = document.getElementById('res-report');
  const sections = [...report.querySelectorAll('.md-section')];
  sections.forEach(sec => {
    if (sec.querySelector('.section-arrow')) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'section-content';
    let next = sec.nextElementSibling;
    while (next && !next.classList.contains('md-section')) {
      const toMove = next; next = next.nextElementSibling;
      wrapper.appendChild(toMove);
    }
    sec.insertAdjacentElement('afterend', wrapper);
    const arrow = document.createElement('span');
    arrow.className = 'section-arrow'; arrow.innerHTML = ' ▾';
    sec.appendChild(arrow);
    sec.style.cursor = 'pointer';
    sec.addEventListener('click', () => {
      wrapper.classList.toggle('collapsed');
      arrow.innerHTML = wrapper.classList.contains('collapsed') ? ' ▸' : ' ▾';
    });
  });
}

function renderLit(arts) {
  const c = document.getElementById('res-lit');
  c.innerHTML = '';
  if (!arts.length) { c.innerHTML=`<p class="lit-empty">${t('no_lit')}</p>`; return; }
  arts.forEach((a,i) => {
    const link = a.link || '';
    const prev = a.snippet ? `<p class="lit-preview">${a.snippet.substring(0,200)}…</p>` : '';
    c.innerHTML+=`<div class="lit-item">
      <div class="lit-title">${i+1}. ${a.title}</div>
      ${prev}
      <div class="lit-meta">
        ${a.journal?`<span class="lit-journal">${a.journal}</span>`:''}
        ${a.year?`<span class="lit-year">${a.year}</span>`:''}
        ${link && link!=='#'?`<a href="${link}" target="_blank" rel="noopener">${t('open_article')}</a>`:''}
      </div>
    </div>`;
  });
}

function copyReport() {
  navigator.clipboard.writeText(document.getElementById('res-report').innerText)
    .then(() => showToast(t('copied'), 'success', 2000))
    .catch(() => showToast('Kopyalama başarısız', 'error'));
}

function resetSystem() {
  currentFile=null; currentFileDataUrl=null; viewingHistory=false; abcdeVisible=false;
  stopLoadingTimer();
  const p = document.getElementById('preview-img'); if(p){p.src='';p.classList.add('hidden');}
  const d = document.getElementById('drop-content'); if(d) d.classList.remove('hidden');
  document.getElementById('remove-img-btn')?.classList.add('hidden');
  document.getElementById('abcde-panel')?.classList.add('hidden');
  document.getElementById('abcde-btn')?.classList.remove('active');
  document.getElementById('risk-badge')?.classList.add('hidden');
  ['inp-age','inp-gender','inp-loc','inp-dur','inp-symp','inp-skintype','inp-allergy','inp-meds','inp-famhist'].forEach(id=>{const e=document.getElementById(id);if(e)e.value='';});
  // Reset alerts
  ['ood-alert','mal-alert','low-conf-alert'].forEach(id=>document.getElementById(id)?.classList.add('hidden'));
  // Clear report content
  const rr = document.getElementById('res-report'); if(rr) rr.innerHTML='';
  _viewTransitioning = false; // release any stuck transition lock
  _analyzing = false;
  const btn = document.querySelector('.analyze-btn');
  if (btn) { btn.disabled = false; btn.style.opacity = ''; btn.style.cursor = ''; }
  goToStep(1);
}

/* ── PDF DOWNLOAD — diagnosis report only ── */
async function downloadPDF() {
  if (!window.html2canvas || !window.jspdf) {
    showToast(t('pdf_error'), 'error'); return;
  }

  const reportEl = document.getElementById('res-report');
  if (!reportEl || !reportEl.innerText.trim()) {
    showToast(currentLang==='tr' ? 'Rapor bulunamadı.' : 'No report found.', 'warning'); return;
  }

  showToast(t('pdf_preparing'), 'info', 8000);

  // Build a clean printable container off-screen
  const container = document.createElement('div');
  container.style.cssText = `
    position: fixed; left: -9999px; top: 0; width: 760px;
    background: #ffffff; color: #1E293B; font-family: 'Inter', sans-serif;
    font-size: 13px; line-height: 1.7; padding: 32px 36px;
  `;

  const diagClass = document.getElementById('res-top-class')?.textContent || '—';
  const conf      = document.getElementById('res-conf')?.textContent || '—';
  const dateStr   = new Date().toLocaleString(currentLang==='tr'?'tr-TR':'en-US');

  container.innerHTML = `
    <div style="border-bottom:2px solid #6366F1;padding-bottom:12px;margin-bottom:20px;">
      <h1 style="margin:0;font-size:20px;color:#6366F1;">DermatoRAG — ${currentLang==='tr'?'Tanı Raporu':'Diagnostic Report'}</h1>
      <p style="margin:4px 0 0;font-size:12px;color:#64748B;">${dateStr}</p>
    </div>
    <div style="background:#F1F5F9;border-radius:8px;padding:12px 16px;margin-bottom:20px;border-left:4px solid #6366F1;">
      <strong style="color:#0F172A;font-size:15px;">${diagClass}</strong>
      <span style="margin-left:12px;color:#6366F1;font-weight:600;">${conf}%</span>
    </div>
    <div id="pdf-report-body" style="color:#1E293B;">${reportEl.innerHTML}</div>
    <div style="margin-top:24px;padding-top:12px;border-top:1px solid #E2E8F0;font-size:11px;color:#64748B;">
      ⚠️ ${currentLang==='tr'
        ? 'Bu rapor YZ tarafından üretilmiştir. Kesin tanı için dermatoloji uzmanına başvurun.'
        : 'This report was generated by AI. Consult a dermatologist for definitive diagnosis.'}
    </div>`;

  // Inject print-friendly styles
  const style = document.createElement('style');
  style.textContent = `
    .md-section { background:#EEF2FF!important; border-left:3px solid #6366F1!important; padding:6px 10px!important; margin:12px 0 6px!important; border-radius:4px; }
    .md-section-warn { background:#FEF2F2!important; border-left-color:#EF4444!important; }
    .md-sec-icon,.md-num,.md-stitle { display:inline!important; }
    .md-stitle { font-weight:700!important; font-size:12px!important; color:#0F172A!important; }
    .md-num { display:inline-flex!important; width:18px!important; height:18px!important; border-radius:50%!important; background:#6366F1!important; color:white!important; font-size:10px!important; align-items:center!important; justify-content:center!important; margin-right:6px!important; }
    .md-line,.md-indent { color:#1E293B!important; }
    .md-list { color:#1E293B!important; }
    .md-ref { background:#E0E7FF!important; color:#4338CA!important; }
    .md-pct { color:#0284C7!important; font-weight:700!important; }
    .md-warn-inline { color:#B45309!important; }
    .section-content { max-height:none!important; opacity:1!important; }
    .section-arrow { display:none!important; }
  `;
  container.appendChild(style);
  document.body.appendChild(container);

  try {
    const canvas = await html2canvas(container, { scale: 2, useCORS: true, backgroundColor: '#ffffff', logging: false });
    document.body.removeChild(container);

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();
    const imgH  = (canvas.height * pageW) / canvas.width;
    let posY = 0;
    while (posY < imgH) {
      if (posY > 0) pdf.addPage();
      pdf.addImage(canvas.toDataURL('image/jpeg', 0.92), 'JPEG', 0, -posY, pageW, imgH);
      posY += pageH;
    }
    const name = diagClass.replace(/[^a-zA-ZçşğüöıÇŞĞÜÖİ0-9]/g, '_').substring(0, 30);
    pdf.save(`DermatoRAG_${name}_${new Date().toISOString().slice(0,10)}.pdf`);
    showToast(t('pdf_done'), 'success');
  } catch(e) {
    if (document.body.contains(container)) document.body.removeChild(container);
    console.error('PDF error:', e);
    showToast(t('pdf_error'), 'error');
  }
}

/* ── IMAGE ZOOM ── */
function openImageZoom() {
  const src = document.getElementById('res-img').src;
  if (!src) return;
  document.getElementById('zoom-img').src = src;
  document.getElementById('zoom-modal').classList.remove('hidden');
  requestAnimationFrame(() => document.getElementById('zoom-modal').classList.add('show'));
}

function closeImageZoom() {
  const m = document.getElementById('zoom-modal');
  m.classList.remove('show');
  setTimeout(() => m.classList.add('hidden'), 280);
}

/* ── ABCDE ── */
function toggleABCDE() {
  abcdeVisible = !abcdeVisible;
  document.getElementById('abcde-panel').classList.toggle('hidden', !abcdeVisible);
  document.getElementById('abcde-btn').classList.toggle('active', abcdeVisible);
}

/* ── SIMILAR CASES ── */
function renderSimilarCases(topClass) {
  const container = document.getElementById('similar-cases');
  const list      = document.getElementById('similar-list');
  if (!container || !list) return;

  // Filter from CURRENT history (not stale cache)
  const similar = analysisHistory.filter(e => e.vision?.top_class === topClass && e.id).slice(0,4);
  if (!similar.length) { container.classList.add('hidden'); return; }
  container.classList.remove('hidden');

  list.innerHTML = similar.map(e => {
    const dateStr = new Date(e.timestamp).toLocaleDateString(currentLang==='tr'?'tr-TR':'en-US',{day:'2-digit',month:'short',year:'numeric'});
    const thumb   = e.imageDataUrl
      ? `<img src="${e.imageDataUrl}" class="sim-thumb" alt="lezyon">`
      : `<div class="sim-thumb sim-ph">🔬</div>`;
    return `<div class="sim-item" onclick="viewHistoryById(${e.id})">
      ${thumb}
      <div class="sim-info">
        <div class="sim-class">${CLASS_SHORT[currentLang]?.[e.vision?.top_class]||e.vision?.top_class}</div>
        <div class="sim-meta">%${e.vision?.confidence} · ${dateStr}</div>
      </div>
    </div>`;
  }).join('');
}

/* ID tabanlı geçmiş görüntüleme — index değişmesinden etkilenmez */
async function viewHistoryById(id) {
  const entry = analysisHistory.find(e => e.id === id);
  if (!entry) {
    showToast(
      currentLang==='tr' ? 'Bu analiz geçmişte artık bulunamıyor.' : 'This analysis is no longer in history.',
      'warning', 3000
    );
    renderSimilarCases(currentTopClass);
    return;
  }
  viewingHistory = true;
  // Scroll to top immediately so transition is visible
  window.scrollTo({ top: 0, behavior: 'smooth' });
  await showView('analysis');
  await goToStep(3);
  document.getElementById('loader-view').classList.add('hidden');
  renderResults({ vision:entry.vision, diagnosis:entry.diagnosis, articles:entry.articles }, entry.imageDataUrl);
}

/* ── HISTORY ── */
function saveToHistory(data) {
  const entry = {
    id: Date.now(), timestamp: new Date().toISOString(),
    imageName: currentFile?.name||'Unknown', imageDataUrl: currentFileDataUrl||null,
    lang: currentLang, vision: data.vision, diagnosis: data.diagnosis, articles: data.articles,
  };
  analysisHistory.unshift(entry);
  if (analysisHistory.length>50) analysisHistory=analysisHistory.slice(0,50);
  analysisHistory.forEach((e,i)=>{ if(i>=10) e.imageDataUrl=null; });
  try {
    localStorage.setItem('dermato_history', JSON.stringify(analysisHistory));
  } catch {
    analysisHistory.forEach(e=>{ e.imageDataUrl=null; });
    try { localStorage.setItem('dermato_history', JSON.stringify(analysisHistory)); } catch {}
  }
  updateHistoryBadge();
}

function updateHistoryBadge() {
  const b = document.getElementById('history-badge');
  if (!b) return;
  b.textContent = analysisHistory.length;
  b.style.display = analysisHistory.length>0 ? 'inline-flex' : 'none';
}

function renderHistory() {
  const container = document.getElementById('history-list');
  if (!container) return;
  if (!analysisHistory.length) {
    container.innerHTML=`<div class="hist-empty"><span>🔬</span><p>${t('hist_empty')}</p></div>`;
    return;
  }
  container.innerHTML = analysisHistory.map((entry,idx) => {
    const dateStr = new Date(entry.timestamp).toLocaleDateString(currentLang==='tr'?'tr-TR':'en-US',{day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'});
    const topClass= CLASS_DISPLAY[currentLang]?.[entry.vision?.top_class]||entry.vision?.top_class_display||entry.vision?.top_class||'—';
    const conf    = entry.vision?.confidence||0;
    const isMal   = entry.vision?.is_malignant;
    const confCol = isMal?'var(--color-err)':'var(--color-success)';
    const thumb   = entry.imageDataUrl
      ? `<img src="${entry.imageDataUrl}" class="hist-thumb" alt="lezyon">`
      : `<div class="hist-thumb-placeholder" style="background:${isMal?'rgba(239,68,68,0.15)':'rgba(16,185,129,0.15)'}">🔬</div>`;
    const malBadge= isMal?`<span class="hist-mal-badge">⚠ Malign</span>`:'';
    return `<div class="hist-item">
      <div class="hist-timeline-dot" style="background:${confCol}"></div>
      ${thumb}
      <div class="hist-info">
        <div class="hist-top-row">
          <span class="hist-diagnosis">${topClass}</span>
          ${malBadge}
        </div>
        <div class="hist-meta">
          <span class="hist-conf" style="color:${confCol}">%${conf} ${t('conf_text')}</span>
          <span class="hist-date">· ${dateStr}</span>
        </div>
        <div class="hist-filename">📎 ${entry.imageName}</div>
      </div>
      <div class="hist-actions">
        <button class="btn-small" onclick="viewHistoryById(${entry.id})">${t('hist_view_btn')}</button>
        <button class="btn-small btn-danger" onclick="deleteHistoryEntry(${entry.id})">${t('hist_del_btn')}</button>
      </div>
    </div>`;
  }).join('');
}

function deleteHistoryEntry(id) {
  const idx = analysisHistory.findIndex(e => e.id === id);
  if (idx === -1) return;
  analysisHistory.splice(idx, 1);
  localStorage.setItem('dermato_history', JSON.stringify(analysisHistory));
  updateHistoryBadge();
  renderHistory();
  // Refresh similar cases panel so deleted entry disappears immediately
  if (currentTopClass) renderSimilarCases(currentTopClass);
  showToast(currentLang==='tr'?'Kayıt silindi.':'Entry deleted.','info',2000);
}

function clearHistory() {
  showModal(
    currentLang==='tr'?'Geçmişi Temizle':'Clear History',
    currentLang==='tr'?'Tüm analiz geçmişi kalıcı olarak silinecek. Emin misiniz?':'All analysis history will be permanently deleted. Are you sure?',
    'warning',
    () => {
      analysisHistory=[];
      localStorage.setItem('dermato_history',JSON.stringify(analysisHistory));
      updateHistoryBadge();
      renderHistory();
      renderSettings();
      // Refresh similar cases so it goes empty immediately
      if (currentTopClass) renderSimilarCases(currentTopClass);
      showToast(currentLang==='tr'?'Geçmiş temizlendi.':'History cleared.','success');
    }
  );
}

/* ── SETTINGS ── */
function renderSettings() {
  const el = document.getElementById('hist-count-display');
  if (el) el.textContent = `${analysisHistory.length} ${t('set_hist_analyses')}`;
  applyTheme(currentTheme);
}

/* ── QUICK FILL ── */
function saveFormData(data) {
  localStorage.setItem('dermato_last_patient', JSON.stringify(data));
}

function loadLastPatient() {
  try {
    const data = JSON.parse(localStorage.getItem('dermato_last_patient') || 'null');
    // Only proceed if there is at least one non-empty value
    const hasRealData = data && Object.values(data).some(v => v && String(v).trim() !== '' && v !== 'none');
    if (!hasRealData) return; // silently do nothing — no toast, no action

    let loaded = 0;
    if (data.age)      { document.getElementById('inp-age').value      = data.age;    loaded++; }
    if (data.gender)   { document.getElementById('inp-gender').value   = data.gender; loaded++; }
    if (data.loc)      { document.getElementById('inp-loc').value      = data.loc;    loaded++; }
    if (data.dur)      { document.getElementById('inp-dur').value      = data.dur;    loaded++; }
    if (data.symp)     { document.getElementById('inp-symp').value     = data.symp;   loaded++; }
    if (data.skintype) { const el = document.getElementById('inp-skintype'); if (el) { el.value = data.skintype; loaded++; } }
    if (data.allergy)  { const el = document.getElementById('inp-allergy');  if (el) { el.value = data.allergy;  loaded++; } }
    if (data.meds)     { const el = document.getElementById('inp-meds');     if (el) { el.value = data.meds;     loaded++; } }
    if (data.famhist && data.famhist !== 'none') { const el = document.getElementById('inp-famhist'); if (el) { el.value = data.famhist; loaded++; } }
    if (loaded > 0) showToast(currentLang==='tr'?`Son hasta bilgileri yüklendi (${loaded} alan).`:`Last patient loaded (${loaded} fields).`,'success',2000);
  } catch { /* fail silently */ }
}

/* ── ONBOARDING TOUR ── */
function startOnboarding(force=false) {
  if (!force && localStorage.getItem('dermato_tour_done')) return;
  tourStep = 0;
  document.getElementById('onboarding-overlay').classList.remove('hidden');
  showTourStep();
}

function showTourStep() {
  const steps = TOUR_STEPS[currentLang] || TOUR_STEPS.tr;
  const step  = steps[tourStep];
  if (!step) { endOnboarding(); return; }

  document.getElementById('tour-num').textContent   = `${tourStep+1} / ${steps.length}`;
  document.getElementById('tour-title').textContent = step.title;
  document.getElementById('tour-body').textContent  = step.body;
  document.querySelector('.tour-icon-big').textContent = step.icon;

  const progress = document.getElementById('tour-progress');
  if (progress) progress.style.width = `${((tourStep+1)/steps.length)*100}%`;

  const nextBtn = document.getElementById('tour-next-btn');
  if (nextBtn) nextBtn.textContent = tourStep===steps.length-1 ? t('tour_finish') : t('tour_next');
}

function tourNext() {
  const steps = TOUR_STEPS[currentLang] || TOUR_STEPS.tr;
  tourStep++;
  if (tourStep >= steps.length) { endOnboarding(); return; }
  showTourStep();
}

function endOnboarding() {
  document.getElementById('onboarding-overlay').classList.add('hidden');
  localStorage.setItem('dermato_tour_done','1');
}

/* ── KEYBOARD SHORTCUTS ── */
function setupKeyboardShortcuts() {
  document.addEventListener('keydown', e => {
    if (e.key==='Escape') {
      closeModal();
      closeImageZoom();
      const sp = document.getElementById('shortcuts-panel');
      if (sp && !sp.classList.contains('hidden')) sp.classList.add('hidden');
      const oo = document.getElementById('onboarding-overlay');
      if (oo && !oo.classList.contains('hidden')) endOnboarding();
      return;
    }
    if (e.key==='?' && !isInputFocused()) { toggleShortcuts(); return; }
    if (e.ctrlKey && e.key==='Enter') {
      const s2 = document.getElementById('step-2');
      if (s2 && !s2.classList.contains('hidden')) startAnalysis();
      return;
    }
    if (e.key==='Enter' && !e.ctrlKey && !isInputFocused()) {
      const s1 = document.getElementById('step-1');
      if (s1 && !s1.classList.contains('hidden') && currentFile) goToStep(2);
    }
  });
}

function isInputFocused() {
  return ['input','textarea','select'].includes(document.activeElement?.tagName.toLowerCase());
}

function toggleShortcuts() {
  document.getElementById('shortcuts-panel')?.classList.toggle('hidden');
}
