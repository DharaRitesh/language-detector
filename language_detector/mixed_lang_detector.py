"""
mixed_lang_detector.py
------------------------
Adds two capabilities on top of the base ML model:

1. Script detection (Unicode ranges) - instantly tells if text is
   written in Bengali script, Devanagari (Hindi) script, or Latin script.

2. Banglish / Hinglish detection - detects Bengali or Hindi words that
   are typed using English (Latin) letters, e.g. "ami bhalo achi",
   "tum kaisa ho". This is very common in real chats but is NOT
   something the base char n-gram model (trained on native-script text)
   can catch, so we use a keyword-heuristic layer for it.

3. Mixed-language detection - splits text into chunks and checks whether
   different chunks belong to different languages / scripts, flagging
   the text as "Mixed".
"""

import re
import joblib
import os

MODEL_DIR = "model"

# --- Common Romanized Bengali (Banglish) words ---
BANGLA_ROMAN_WORDS = {
    "ami", "tumi", "tui", "apni", "kemon", "acho", "achi", "bhalo",
    "bhalobasi", "ki", "koro", "korchi", "korছি", "khabo", "khaisi",
    "jaবo", "jabo", "jacchi", "kotha", "bolo", "bolছি", "dada", "didi",
    "baba", "মা", "maa", "bari", "bhai", "kemne", "keno", "shob",
    "shobar", "porikkha", "poroshu", "ajke", "kalke", "rat", "dupur",
    "shokal", "sondha", "khub", "onek", "eto", "koto", "taka", "লাগবে",
    "lagbe", "hobe", "hoyeche", "hocche", "korbo", "korbi", "amar",
    "tomar", "tor", "oder", "amader", "tader", "bhalobasha", "mishti",
}

# --- Common Romanized Hindi (Hinglish) words ---
HINDI_ROMAN_WORDS = {
    "main", "mai", "tum", "aap", "kaisa", "kaise", "hai", "ho", "hoon",
    "kya", "kyun", "kyu", "nahi", "nahin", "haan", "acha", "accha",
    "theek", "thik", "bhai", "yaar", "kaam", "karo", "karna", "karta",
    "karti", "kar", "raha", "rahe", "rahi", "gaya", "gayi", "gaye",
    "jaana", "jana", "khana", "khaana", "peena", "paani", "ghar",
    "mera", "meri", "tera", "teri", "uska", "uski", "hum", "humein",
    "tumhe", "tumhara", "abhi", "kal", "aaj", "subah", "shaam", "raat",
    "bahut", "bohot", "thoda", "zyada", "sab", "kuch", "kahan", "kab",
}

BENGALI_UNICODE_RANGE = re.compile(r"[\u0980-\u09FF]")
DEVANAGARI_UNICODE_RANGE = re.compile(r"[\u0900-\u097F]")
LATIN_RANGE = re.compile(r"[A-Za-z]")

# The standard Kaggle "Language Detection" dataset (17 languages) does NOT
# include Bengali as a trained class, even though Hindi (Devanagari script)
# is included. So if we ever hand native Bengali-script text to the ML
# model, it has no correct class to pick and will guess something close
# to random (e.g. Danish, German) with low confidence.
#
# Fix: since Unicode script alone is a 100% reliable signal for Bengali
# (no other language in this project uses that script), we short-circuit
# straight to "Bengali" whenever the dominant script is Bengali, instead
# of trusting the ML model's guess.
SCRIPT_OVERRIDE = {
    "bengali_script": "Bengali",
}


def detect_script(text: str) -> str:
    """Returns the dominant script used in the text."""
    bengali_chars = len(BENGALI_UNICODE_RANGE.findall(text))
    devanagari_chars = len(DEVANAGARI_UNICODE_RANGE.findall(text))
    latin_chars = len(LATIN_RANGE.findall(text))

    counts = {
        "bengali_script": bengali_chars,
        "devanagari_script": devanagari_chars,
        "latin_script": latin_chars,
    }
    dominant = max(counts, key=counts.get)
    if counts[dominant] == 0:
        return "unknown"
    return dominant


def romanized_language_score(text: str):
    """
    Returns (label, score) for romanized Bangla/Hindi keyword matches.
    score = fraction of words that matched a known list.
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return None, 0.0

    bangla_hits = sum(1 for w in words if w in BANGLA_ROMAN_WORDS)
    hindi_hits = sum(1 for w in words if w in HINDI_ROMAN_WORDS)

    bangla_score = bangla_hits / len(words)
    hindi_score = hindi_hits / len(words)

    if bangla_score == 0 and hindi_score == 0:
        return None, 0.0
    if bangla_score >= hindi_score:
        return "Banglish (Romanized Bengali)", bangla_score
    return "Hinglish (Romanized Hindi)", hindi_score


def load_ml_artifacts():
    clf = joblib.load(os.path.join(MODEL_DIR, "lang_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    return clf, vectorizer, le


def predict_language(text, clf, vectorizer, le, top_k=3):
    """Returns list of (language, probability) sorted descending."""
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    ranked = sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def chunk_text(text, chunk_size=4):
    """Splits text into word chunks for per-chunk language checking."""
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [c for c in chunks if c.strip()]


def analyze_text(text, clf, vectorizer, le):
    """
    Full analysis pipeline combining:
      - script detection (with a hard override for Bengali, since it's
        not one of the ML model's trained classes)
      - romanized Bangla/Hindi keyword detection
      - ML model prediction
      - chunk-level mixed-language detection (Latin script only - this is
        where genuine code-mixing / romanization ambiguity happens; running
        it on native scripts the model already confidently recognizes just
        adds noise)

    Returns a dict with all details, including:
      - final_label: the single label to show as the headline result
      - show_confidence: whether the UI should show the top-3 confidence
        breakdown (only True for genuinely ambiguous cases: mixed text or
        romanized Banglish/Hinglish). For a single, clearly-recognized
        language, this stays False so only one result is shown.
    """
    result = {
        "input_text": text,
        "script": detect_script(text),
        "ml_top_prediction": None,
        "ml_top_probs": [],
        "romanized_hint": None,
        "romanized_score": 0.0,
        "is_mixed": False,
        "chunk_predictions": [],
        "final_label": None,
        "show_confidence": False,
    }

    if not text.strip():
        return result

    script = result["script"]

    # ML model prediction
    ranked = predict_language(text, clf, vectorizer, le, top_k=3)
    result["ml_top_prediction"] = ranked[0][0]
    result["ml_top_probs"] = ranked

    # Romanized Bangla/Hindi heuristic (only meaningful for Latin script)
    if script == "latin_script":
        label, score = romanized_language_score(text)
        result["romanized_hint"] = label
        result["romanized_score"] = round(score, 3)

    # Chunk-level mixed-language check - restricted to Latin script text,
    # since that's the only case where the model's word-level guesses are
    # a meaningful signal for genuine language-mixing (native scripts like
    # Bengali/Hindi are already unambiguous from the script alone).
    labels_seen = set()
    if script == "latin_script":
        chunks = chunk_text(text, chunk_size=4)
        for c in chunks:
            pred = predict_language(c, clf, vectorizer, le, top_k=1)[0][0]
            result["chunk_predictions"].append((c, pred))
            labels_seen.add(pred)
        result["is_mixed"] = len(labels_seen) > 1

    # Decide the single final label to display
    script_override = SCRIPT_OVERRIDE.get(script)
    if script_override:
        result["final_label"] = script_override
    elif result["romanized_hint"] and result["romanized_score"] >= 0.15:
        result["final_label"] = result["romanized_hint"]
    elif result["is_mixed"]:
        mixed_langs = " + ".join(sorted(labels_seen))
        result["final_label"] = f"Mixed ({mixed_langs})"
    else:
        result["final_label"] = result["ml_top_prediction"]

    # Only show the multi-language confidence breakdown for genuinely
    # ambiguous cases (mixed text, or romanized Banglish/Hinglish).
    result["show_confidence"] = bool(
        result["is_mixed"] or (result["romanized_hint"] and result["romanized_score"] >= 0.15)
    )

    return result
