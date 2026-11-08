from sanad.textnorm import detect_language, normalize, to_latin_digits, tokenize


def test_normalize_unifies_arabic_forms():
    assert normalize("إرجاع") == normalize("ارجاع") == "ارجاع"
    assert normalize("مدرسة") == "مدرسه"
    assert normalize("مستشفى") == "مستشفي"


def test_normalize_strips_diacritics_and_tatweel():
    assert normalize("سَنَد") == "سند"
    assert normalize("شكــــرا") == "شكرا"


def test_arabic_indic_digits():
    assert to_latin_digits("ORD-١٠٠٢٣") == "ORD-10023"
    assert normalize("۱۲۳") == "123"


def test_tokenize_drops_stopwords_and_article():
    tokens = tokenize("ازاي أرجّع المنتج بتاعي؟")
    assert "منتج" in tokens
    assert "ازاي" not in tokens and "بتاعي" not in tokens


def test_tokenize_english_lowercases():
    assert tokenize("How do I Return an ITEM?") == ["return", "item"]


def test_detect_language():
    assert detect_language("فين الاوردر بتاعي") == "ar"
    assert detect_language("Where is my order?") == "en"
    assert detect_language("الاوردر ORD-10023 فين") == "ar"
    assert detect_language("12345") == "en"
