"""
train_model.py
----------------
Trains a Language Detection model using:
  - Character n-gram (1-3) TF-IDF features
  - Logistic Regression classifier

Dataset expected at: data/Language Detection.csv
  Download from Kaggle: "Language Detection" dataset (by basilb2s)
  It must have two columns: "Text" and "Language"

If the dataset file is not found, a small built-in sample dataset
is used instead (for quick testing only - accuracy will be low).

Outputs:
  model/lang_model.joblib       -> trained LogisticRegression model
  model/vectorizer.joblib       -> trained TF-IDF vectorizer
  model/label_encoder.joblib    -> label encoder (language names)
  model/confusion_matrix.png    -> confusion matrix heatmap
  model/metrics.txt             -> accuracy + classification report
"""

import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

DATA_PATH = os.path.join("data", "Language Detection.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading dataset from {DATA_PATH} ...")
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["Text", "Language"])
        return df

    print("!! Dataset file not found. Using a small built-in sample dataset "
          "for demo purposes. Download the Kaggle 'Language Detection' "
          "dataset and place it at data/Language Detection.csv for real results.")

    sample = {
        "Text": [
            "Hello, how are you today?",
            "I love programming in Python.",
            "This is a wonderful sunny day.",
            "আমি ভাত খেয়েছি।",
            "তুমি কেমন আছো আজকে?",
            "আমার বাড়ি কলকাতায়।",
            "मुझे हिंदी बहुत पसंद है।",
            "आप कैसे हैं आज?",
            "यह एक अच्छा दिन है।",
            "Bonjour, comment ça va?",
            "J'aime beaucoup la France.",
            "C'est une belle journée.",
            "Hola, ¿cómo estás hoy?",
            "Me encanta aprender español.",
            "Es un día muy bonito.",
        ],
        "Language": [
            "English", "English", "English",
            "Bengali", "Bengali", "Bengali",
            "Hindi", "Hindi", "Hindi",
            "French", "French", "French",
            "Spanish", "Spanish", "Spanish",
        ],
    }
    return pd.DataFrame(sample)


def main():
    df = load_data()
    print(f"Total samples: {len(df)}")
    print("Languages found:", sorted(df["Language"].unique()))

    X_text = df["Text"].astype(str)
    y_text = df["Language"].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    # Character n-gram TF-IDF: analyzer='char' with ngram_range=(1,3)
    # This captures spelling / script patterns which work well
    # across languages regardless of word segmentation.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        max_features=20000,
    )
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(df) > 30 else None
    )

    print("Training Logistic Regression model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Only report on classes that actually appear in this test split
    # (matters for very small/demo datasets; real Kaggle dataset won't hit this)
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = le.inverse_transform(present_labels)
    report = classification_report(
        y_test, y_pred, labels=present_labels,
        target_names=present_names, zero_division=0
    )

    print(f"Accuracy: {acc:.4f}")
    print(report)

    with open(os.path.join(MODEL_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=present_names, yticklabels=present_names
    )
    plt.xlabel("Predicted Language")
    plt.ylabel("Actual Language")
    plt.title("Language Detection - Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Save model artifacts
    joblib.dump(clf, os.path.join(MODEL_DIR, "lang_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    print("\nSaved model artifacts to model/ folder:")
    print("  - lang_model.joblib")
    print("  - vectorizer.joblib")
    print("  - label_encoder.joblib")
    print("  - confusion_matrix.png")
    print("  - metrics.txt")


if __name__ == "__main__":
    main()
