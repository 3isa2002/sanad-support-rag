"""Arabic-aware text normalization, tokenization and language detection."""

from __future__ import annotations

import re
import unicodedata

# Harakat, tanween, shadda, sukun, dagger alef and Quranic marks.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[\w]+", re.UNICODE)

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_LETTERS = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        "ک": "ك",
        "ی": "ي",
    }
)

# Definite-article / conjunction prefixes, longest first.
_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال")

STOPWORDS = {
    # English
    "a", "an", "the", "is", "are", "am", "be", "to", "of", "in", "on", "for", "and",
    "or", "i", "my", "me", "you", "your", "it", "this", "that", "do", "does", "can",
    "how", "what", "with", "at", "by", "from", "if", "please", "hi", "hello", "we",
    # Arabic (already normalized form)
    "في", "من", "الي", "علي", "عن", "ان", "او", "و", "يا", "هل", "ما", "ماذا", "انا",
    "انت", "هو", "هي", "ده", "دي", "دا", "اللي", "الذي", "التي", "مع", "كان", "لو",
    "عايز", "عاوز", "عايزه", "عاوزه", "ممكن", "لي", "ليا", "بتاعي", "بتاعتي", "كده",
    "ازاي", "ايه", "فين", "امتي", "مين", "اريد", "كيف", "متي", "اين", "لماذا", "ليه",
    "شكرا", "لا", "مش", "بس", "حاجه", "اعمل", "عندي",
}


def to_latin_digits(text: str) -> str:
    return text.translate(_DIGITS)


def normalize(text: str) -> str:
    """Normalize text for matching: NFKC, lowercase, unify Arabic letter forms,
    strip diacritics and tatweel, convert Arabic-Indic digits."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_LETTERS).translate(_DIGITS)
    return text.lower()


def _strip_prefix(token: str) -> str:
    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            return token[len(prefix):]
    return token


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Normalize and split into tokens; Arabic tokens get light prefix stripping."""
    tokens = []
    for tok in _TOKEN.findall(normalize(text)):
        if drop_stopwords and tok in STOPWORDS:
            continue
        if _ARABIC_LETTER.match(tok):
            tok = _strip_prefix(tok)
        if drop_stopwords and tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def detect_language(text: str) -> str:
    """Return "ar" if Arabic letters make up a meaningful share of the letters, else "en"."""
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    if arabic == 0 and latin == 0:
        return "ar" if re.search(r"[؀-ۿ]", text) else "en"
    return "ar" if arabic / (arabic + latin) >= 0.3 else "en"
