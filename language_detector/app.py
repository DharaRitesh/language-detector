"""
app.py
--------
Streamlit web app for the Language Detection project - Modern UI version 3.

Run with:
    streamlit run app.py
    (or: python -m streamlit run app.py)
"""

import os
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

from mixed_lang_detector import load_ml_artifacts, analyze_text

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

GTTS_LANG_CODES = {
    "English": "en", "French": "fr", "Spanish": "es", "Portugeese": "pt",
    "Italian": "it", "Russian": "ru", "Sweedish": "sv", "Malayalam": "ml",
    "Dutch": "nl", "Arabic": "ar", "Turkish": "tr", "German": "de",
    "Tamil": "ta", "Danish": "da", "Kannada": "kn", "Greek": "el",
    "Nepali": "ne", "Hindi": "hi", "Bengali": "bn",
}

LANGUAGE_FLAGS = {
    "English": "🇬🇧", "French": "🇫🇷", "Spanish": "🇪🇸", "Portugeese": "🇵🇹",
    "Italian": "🇮🇹", "Russian": "🇷🇺", "Sweedish": "🇸🇪", "Malayalam": "🇮🇳",
    "Dutch": "🇳🇱", "Arabic": "🇸🇦", "Turkish": "🇹🇷", "German": "🇩🇪",
    "Tamil": "🇮🇳", "Danish": "🇩🇰", "Kannada": "🇮🇳", "Greek": "🇬🇷",
    "Nepali": "🇳🇵", "Hindi": "🇮🇳", "Bengali": "🇧🇩",
}


def raw_html(markup: str):
    """
    Render an HTML/CSS block via st.markdown, stripping the leading
    whitespace from every line first.

    WHY THIS EXISTS: Streamlit's Markdown layer follows CommonMark rules -
    if an HTML tag starts more than 3 spaces in from the left margin, it is
    treated as an INDENTED CODE BLOCK instead of raw HTML (this is what
    caused the "<div> ... </div> shown as literal text" bug). Because our
    HTML is written inside indented Python triple-quoted strings, every
    line naturally starts with several spaces. Stripping each line's
    leading whitespace before rendering avoids that misinterpretation.
    """
    lines = [line.strip() for line in markup.strip("\n").splitlines()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def inject_custom_css():
    raw_html("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background: radial-gradient(circle at 10% 0%, #1b1440 0%, #0F1117 45%),
                        radial-gradient(circle at 90% 20%, #0f2d3d 0%, #0F1117 55%);
        }

        .hero-box {
            display: flex;
            align-items: center;
            gap: 16px;
            background: linear-gradient(120deg, #6C5CE7 0%, #e84393 50%, #00cec9 100%);
            padding: 20px 28px;
            border-radius: 18px;
            margin-bottom: 22px;
            box-shadow: 0 8px 26px rgba(108, 92, 231, 0.35);
        }
        .logo-badge {
            flex-shrink: 0;
            width: 54px; height: 54px;
            border-radius: 14px;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.35);
            display: flex; align-items: center; justify-content: center;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 15px;
            color: white;
            line-height: 1.1;
            text-align: center;
        }
        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 26px;
            font-weight: 700;
            color: white;
            margin: 0;
        }
        .hero-subtitle {
            color: #F1EEFF;
            font-size: 13px;
            margin-top: 2px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #1A1D29;
            border-radius: 16px !important;
            border: 1px solid #2A2E3F !important;
        }

        .card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 15px;
            font-weight: 600;
            color: #E8E9F3;
            margin-bottom: 2px;
        }
        .card-subtitle {
            font-size: 12px;
            color: #8B8FA3;
            margin-bottom: 10px;
        }

        .result-banner {
            background: linear-gradient(120deg, #00cec9 0%, #00b894 100%);
            padding: 18px 22px;
            border-radius: 14px;
            margin-bottom: 14px;
            box-shadow: 0 6px 20px rgba(0, 206, 201, 0.25);
        }
        .result-lang {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 24px;
            font-weight: 700;
            color: #05261f;
        }
        .result-sub { color: #06392f; font-size: 12.5px; margin-top: 2px; opacity: 0.85; }

        .mixed-banner {
            background: linear-gradient(120deg, #fdcb6e 0%, #e17055 100%);
            padding: 12px 18px;
            border-radius: 12px;
            color: #2d1400;
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 14px;
        }

        .prob-row { display: flex; align-items: center; margin-bottom: 9px; }
        .prob-label { width: 130px; font-size: 13px; color: #E8E9F3; font-weight: 500; }
        .prob-track {
            flex-grow: 1; background: #2A2E3F; border-radius: 8px;
            height: 9px; overflow: hidden; margin: 0 10px;
        }
        .prob-fill {
            background: linear-gradient(90deg, #6C5CE7, #e84393);
            height: 100%; border-radius: 8px;
        }
        .prob-value { width: 46px; font-size: 12px; color: #8B8FA3; text-align: right; }

        .info-chip {
            display: inline-block;
            background: #2A2E3F;
            color: #7ee6e0;
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-family: monospace;
        }

        .section-label {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #e84393;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .stButton>button {
            background: linear-gradient(120deg, #6C5CE7, #e84393);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 9px 16px;
            font-weight: 600;
            width: 100%;
        }
        .stButton>button:hover {
            filter: brightness(1.1);
            color: white;
        }

        .stTextArea textarea {
            background: #14171f !important;
            border-radius: 10px !important;
            border: 1px solid #2A2E3F !important;
            color: #E8E9F3 !important;
            font-size: 14.5px !important;
        }

        section[data-testid="stSidebar"] {
            background: #14171f;
            border-right: 1px solid #2A2E3F;
        }
    </style>
    """)


def render_hero():
    raw_html("""
    <div class="hero-box">
        <div class="logo-badge">A<br>অ अ</div>
        <div>
            <div class="hero-title">Language Detection Engine</div>
            <div class="hero-subtitle">17 languages &middot; Banglish/Hinglish aware &middot; 98.7% accuracy</div>
        </div>
    </div>
    """)


def render_probability_bars(ml_top_probs):
    rows = []
    for lang, prob in ml_top_probs:
        flag = LANGUAGE_FLAGS.get(lang, "🌐")
        pct = prob * 100
        rows.append(
            f'<div class="prob-row">'
            f'<div class="prob-label">{flag} {lang}</div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%;"></div></div>'
            f'<div class="prob-value">{pct:.1f}%</div>'
            f'</div>'
        )
    raw_html("".join(rows))


def resolve_effective_language(result: dict) -> str:
    """
    Picks the language name to use for the flag icon and TTS voice.
    Prefers `final_label` when it's a real recognized language name
    (e.g. the Bengali script-override case), and only falls back to the
    raw ML guess when final_label is something like "Mixed (...)" or a
    Banglish/Hinglish hint that isn't a plain language name.
    """
    label = result.get("final_label") or ""
    if label in GTTS_LANG_CODES or label in LANGUAGE_FLAGS:
        return label
    return result.get("ml_top_prediction") or "English"


@st.cache_resource
def get_model():
    return load_ml_artifacts()


def page_live_detection(clf, vectorizer, le):
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Type or paste any text</div>'
                '<div class="card-subtitle">Native scripts, or Romanized Banglish / Hinglish - both work.</div>',
                unsafe_allow_html=True,
            )
            text_input = st.text_area(
                "Text",
                placeholder="Type here to detect the language...",
                height=130,
                label_visibility="collapsed",
            )

        if text_input.strip():
            result = analyze_text(text_input, clf, vectorizer, le)

            st.markdown('<div class="section-label">Translate & Listen</div>', unsafe_allow_html=True)
            with st.container(border=True):
                c1, c2 = st.columns(2)
                target_lang_for_tts = resolve_effective_language(result)
                gtts_code = GTTS_LANG_CODES.get(target_lang_for_tts, "en")

                with c1:
                    if TRANSLATOR_AVAILABLE:
                        if st.button("🔁  Translate to English"):
                            try:
                                translated = GoogleTranslator(source="auto", target="en").translate(text_input)
                                st.info(f"**Translation:** {translated}")
                            except Exception as e:
                                st.error(f"Translation failed (check internet): {e}")
                    else:
                        st.warning("Install `deep-translator` for translation.")

                with c2:
                    if TTS_AVAILABLE:
                        if st.button("🔊  Play Pronunciation"):
                            try:
                                tts = gTTS(text=text_input, lang=gtts_code)
                                audio_path = os.path.join(MODEL_DIR, "tts_output.mp3")
                                tts.save(audio_path)
                                st.audio(open(audio_path, "rb").read(), format="audio/mp3")
                            except Exception as e:
                                st.error(f"TTS failed (check internet): {e}")
                    else:
                        st.warning("Install `gTTS` for text-to-speech.")

    with col_output:
        st.markdown('<div class="section-label">Detection Result</div>', unsafe_allow_html=True)

        if text_input.strip():
            result = analyze_text(text_input, clf, vectorizer, le)
            flag = LANGUAGE_FLAGS.get(resolve_effective_language(result), "🌐")

            raw_html(f"""
            <div class="result-banner">
                <div class="result-lang">{flag} {result['final_label']}</div>
                <div class="result-sub">Detected script: {result['script'].replace('_', ' ').title()}</div>
            </div>
            """)

            if result["is_mixed"]:
                st.markdown(
                    '<div class="mixed-banner">⚠️ This text appears to mix multiple languages.</div>',
                    unsafe_allow_html=True,
                )

            # Only show the confidence breakdown / chunk details for
            # genuinely ambiguous cases (mixed text, or romanized
            # Banglish/Hinglish). A single, clearly-recognized language
            # just shows the one result above - nothing else.
            if result["show_confidence"]:
                with st.container(border=True):
                    st.markdown(
                        '<div class="card-title">Model Confidence</div>'
                        '<div class="card-subtitle">Top predicted languages with probability score</div>',
                        unsafe_allow_html=True,
                    )
                    render_probability_bars(result["ml_top_probs"])

                if result["romanized_hint"]:
                    with st.container(border=True):
                        st.markdown(
                            f'<div class="card-title">Romanized Language Hint</div>'
                            f'<div class="card-subtitle">Detected via common word matching</div>'
                            f'<span class="info-chip">{result["romanized_hint"]} &middot; '
                            f'match {result["romanized_score"]*100:.1f}%</span>',
                            unsafe_allow_html=True,
                        )

                if result["chunk_predictions"]:
                    with st.expander("🔬 Word-chunk level breakdown"):
                        for chunk, pred in result["chunk_predictions"]:
                            st.write(f"`{chunk}` → **{pred}**")
        else:
            with st.container(border=True):
                st.markdown(
                    '<div class="card-subtitle">Start typing on the left to see live '
                    'detection results here.</div>',
                    unsafe_allow_html=True,
                )


def page_model_dashboard():
    st.markdown('<div class="section-label">Evaluation</div>', unsafe_allow_html=True)
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    metrics_path = os.path.join(MODEL_DIR, "metrics.txt")

    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Confusion Matrix</div>'
                '<div class="card-subtitle">Rows = actual language, columns = predicted. '
                'A perfect model shows a solid diagonal line.</div>',
                unsafe_allow_html=True,
            )
            if os.path.exists(cm_path):
                st.image(Image.open(cm_path), use_container_width=True)
            else:
                st.warning("Run `python train_model.py` first to generate this.")

    with col_b:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Classification Report</div>'
                '<div class="card-subtitle">Precision / Recall / F1 per language</div>',
                unsafe_allow_html=True,
            )
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="text")


def page_how_it_works():
    st.markdown('<div class="section-label">Under the hood</div>', unsafe_allow_html=True)

    steps = [
        ("1️⃣ Character n-gram features",
         "Instead of splitting text into words, the model looks at overlapping 1-3 letter "
         "sequences (e.g. 'th', 'the', 'her'). This captures spelling and script patterns "
         "unique to each language, even in short texts."),
        ("2️⃣ Logistic Regression classifier",
         "A lightweight, fast ML model trained on those n-gram features to predict which "
         "of the 17 languages a piece of text belongs to."),
        ("3️⃣ Script detection layer",
         "Checks the Unicode range of each character to instantly know whether the text is "
         "in Bengali script, Devanagari (Hindi) script, Latin letters, etc."),
        ("4️⃣ Banglish / Hinglish detector",
         "If the text is in Latin letters, it's checked against curated lists of common "
         "romanized Bengali/Hindi words (like 'bhalo', 'kemon', 'kaisa', 'hai') to catch text "
         "like 'ami bhalo achi' that the base model can't recognize on its own."),
        ("5️⃣ Mixed-language check",
         "The text is split into small word chunks and each chunk is classified separately. "
         "If different chunks get different language labels, the whole text is flagged as Mixed."),
    ]

    for title, desc in steps:
        with st.container(border=True):
            st.markdown(
                f'<div class="card-title">{title}</div>'
                f'<div class="card-subtitle" style="margin-bottom:0;">{desc}</div>',
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(page_title="Language Detector", page_icon="🌐", layout="wide")
    inject_custom_css()
    render_hero()

    if not os.path.exists(os.path.join(MODEL_DIR, "lang_model.joblib")):
        st.error(
            "Model files not found. Please run `python train_model.py` first "
            "to train and save the model."
        )
        st.stop()

    clf, vectorizer, le = get_model()

    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Live Detection", "Model Dashboard", "How It Works"],
            icons=["search", "bar-chart-line", "info-circle"],
            menu_icon="list",
            default_index=0,
            styles={
                "container": {"background-color": "#14171f"},
                "icon": {"color": "#e84393", "font-size": "16px"},
                "nav-link": {
                    "font-size": "14px",
                    "color": "#C7C9DB",
                    "--hover-color": "#1A1D29",
                },
                "nav-link-selected": {"background-color": "#6C5CE7"},
            },
        )

    if selected == "Live Detection":
        page_live_detection(clf, vectorizer, le)
    elif selected == "Model Dashboard":
        page_model_dashboard()
    else:
        page_how_it_works()


if __name__ == "__main__":
    main()
