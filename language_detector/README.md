# 🌐 Language Detection Project

A language detection system built with **character n-gram TF-IDF + Logistic
Regression**, extended with Banglish/Hinglish (romanized) detection, mixed
language detection, a Streamlit web app, translation, text-to-speech, and a
browser extension demo.

## Project Structure
```
language_detector/
├── data/
│   └── Language Detection.csv     <- put the Kaggle dataset here
├── model/                         <- created after training
│   ├── lang_model.joblib
│   ├── vectorizer.joblib
│   ├── label_encoder.joblib
│   ├── confusion_matrix.png
│   └── metrics.txt
├── train_model.py                 <- Step 1: train the ML model
├── mixed_lang_detector.py         <- Banglish/Hinglish + mixed-lang logic
├── app.py                         <- Streamlit web app (Step 2)
├── api.py                         <- Flask API for the browser extension
├── browser_extension/
│   ├── manifest.json
│   ├── popup.html
│   └── popup.js
└── requirements.txt
```

## Step 1: Setup

```bash
pip install -r requirements.txt
```

## Step 2: Get the dataset

Download the **"Language Detection"** dataset from Kaggle
(search: `Language Detection dataset basilb2s`, ~17 languages, columns
`Text` and `Language`). Place the CSV file at:

```
data/Language Detection.csv
```

> If you skip this step, `train_model.py` will still run using a small
> built-in sample (5 languages, 15 rows) just so you can test the pipeline —
> but accuracy will be meaningless. Use the real dataset for your actual
> project submission.

## Step 3: Train the model

```bash
python train_model.py
```

This will:
- Load the dataset
- Convert text to character n-gram (1–3) TF-IDF features
- Train a Logistic Regression classifier
- Save the model, vectorizer, and label encoder to `model/`
- Generate `model/confusion_matrix.png` and `model/metrics.txt`

## Step 4: Run the Streamlit web app

```bash
streamlit run app.py
```

Features in the app:
- **Live Detection tab**: type any text, see the detected language update
  instantly, including Banglish/Hinglish detection and mixed-language
  warnings
- **Translation**: translate detected text to English (via `deep-translator`,
  free, requires internet)
- **Text-to-Speech**: hear the text pronounced in its detected language
  (via `gTTS`, free, requires internet)
- **Confusion Matrix tab**: visual dashboard of model performance

## Step 5: Run the browser extension demo

1. Start the local API server:
   ```bash
   python api.py
   ```
   This runs at `http://localhost:5000` and must stay running while you use
   the extension.

2. Load the extension in Chrome:
   - Go to `chrome://extensions`
   - Enable **Developer mode** (top right)
   - Click **Load unpacked**
   - Select the `browser_extension/` folder

3. Click the extension icon in your browser toolbar, type text into the
   popup, and see the live detected language (it calls your local API).

## How the Banglish/Hinglish detection works

The base ML model is trained on **native-script** text (Bengali script,
Devanagari script, etc.), so it cannot recognize Bengali or Hindi written
with English letters (e.g. `"ami bhalo achi"`, `"tum kaisa ho"`). To handle
this, `mixed_lang_detector.py` adds:

1. **Script detection** — checks Unicode character ranges to instantly know
   if the text is in Bengali script, Devanagari script, or Latin letters.
2. **Romanized keyword matching** — if the text is in Latin script, it is
   checked against curated lists of common romanized Bengali and Hindi
   words. If enough words match, the text is labeled Banglish or Hinglish.
3. **Chunk-level mixed detection** — the text is split into small word
   chunks, each chunk is classified separately, and if different chunks
   get different language labels, the text is flagged as "Mixed".

## Notes for your project report

- Mention that character-level n-grams (rather than word-level) were chosen
  specifically because they capture spelling/script patterns, which works
  well even for short texts and across scripts.
- The confusion matrix helps identify which languages are commonly confused
  (e.g., similar Latin-script European languages), which is a good talking
  point during your viva/demo.
- The Banglish/Hinglish + mixed-language layer is what makes this project
  different from typical language-detection submissions, since most only
  handle single, native-script language classification.
