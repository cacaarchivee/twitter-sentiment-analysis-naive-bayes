"""
Naive Bayes Text Classification Pipeline
==========================================
Pipeline klasifikasi sentimen dari teks hasil preprocessing:
  1. Auto-labeling berbasis keyword (negatif / positif / netral)
  2. Split train/test
  3. Feature extraction: CountVectorizer, TF-IDF, N-gram (dibandingkan)
  4. Training Multinomial Naive Bayes & pemilihan model terbaik
  5. Evaluasi (accuracy, precision, recall, F1, confusion matrix)
  6. Cross-validation 5-fold
  7. Top feature per kelas
  8. Simpan model + vectorizer (pickle)

Cara pakai:
    python naive_bayes_classifier.py --input data_preprocessed.csv --outdir results/

Kolom input yang dibutuhkan: 'text_final'
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

NEGATIVE_KEYWORDS = ["tolak", "protes", "kritik", "mundur", "gagal", "buruk", "rusak", "korupsi", "demo", "ribut"]
POSITIVE_KEYWORDS = ["dukung", "setuju", "bagus", "baik", "sukses", "berhasil", "hebat", "mantap", "pro"]


def auto_label(text: str) -> str:
    """Labeling otomatis berbasis kemunculan keyword positif/negatif."""
    text = str(text).lower()
    neg_score = sum(1 for word in NEGATIVE_KEYWORDS if word in text)
    pos_score = sum(1 for word in POSITIVE_KEYWORDS if word in text)

    if neg_score > pos_score:
        return "negatif"
    if pos_score > neg_score:
        return "positif"
    return "netral"


def train_and_compare(X_train, X_test, y_train, y_test):
    """Latih model dengan 3 strategi ekstraksi fitur, kembalikan yang terbaik."""
    vectorizers = {
        "Count": CountVectorizer(max_features=3000, min_df=2, max_df=0.8),
        "TF-IDF": TfidfVectorizer(max_features=3000, min_df=2, max_df=0.8),
        "N-gram": CountVectorizer(max_features=3000, min_df=2, max_df=0.8, ngram_range=(1, 2)),
    }

    results = {}
    for name, vec in vectorizers.items():
        print(f"\nTraining with {name}...")
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"✓ Accuracy: {acc * 100:.2f}%")
        results[name] = {"vectorizer": vec, "model": model, "y_pred": y_pred, "accuracy": acc}

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n🏆 BEST MODEL: {best_name} (Accuracy: {results[best_name]['accuracy'] * 100:.2f}%)")
    return best_name, results[best_name]


def run_pipeline(input_path: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    print("=" * 80)
    print("NAIVE BAYES TEXT CLASSIFICATION PIPELINE")
    print("=" * 80)

    df = pd.read_csv(input_path, sep=";", engine="python")
    print(f"✓ Data loaded: {len(df)} records")

    print("\n[STEP 1] LABELING DATA")
    df["label"] = df["text_final"].apply(auto_label)
    label_dist = df["label"].value_counts()
    print(f"Distribusi Label:\n{label_dist}")

    plt.figure(figsize=(8, 5))
    label_dist.plot(kind="bar", color=["#ff6b6b", "#4ecdc4", "#95e1d3"])
    plt.title("Distribusi Label Dataset", fontsize=14, fontweight="bold")
    plt.xlabel("Label")
    plt.ylabel("Jumlah")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "label_distribution.png"), dpi=150)
    plt.close()

    print("\n[STEP 2] SPLIT DATA")
    X, y = df["text_final"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"✓ Training: {len(X_train)} | Testing: {len(X_test)}")

    print("\n[STEP 3] TRAINING & COMPARING MODELS")
    best_name, best = train_and_compare(X_train, X_test, y_train, y_test)
    final_model, final_vectorizer, y_pred_final = best["model"], best["vectorizer"], best["y_pred"]

    print("\n[STEP 4] EVALUATION")
    accuracy = accuracy_score(y_test, y_pred_final)
    precision = precision_score(y_test, y_pred_final, average="weighted")
    recall = recall_score(y_test, y_pred_final, average="weighted")
    f1 = f1_score(y_test, y_pred_final, average="weighted")

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred_final)}")

    cm = confusion_matrix(y_test, y_pred_final)
    labels = sorted(y_test.unique())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {best_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close()

    print("\n[STEP 5] CROSS-VALIDATION (5-Fold)")
    cv_vectorizer = type(final_vectorizer)(**final_vectorizer.get_params())
    X_cv = cv_vectorizer.fit_transform(X)
    cv_scores = cross_val_score(MultinomialNB(alpha=1.0), X_cv, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f}%)")

    print("\n[STEP 6] TOP FEATURES PER CLASS")
    feature_names = final_vectorizer.get_feature_names_out()
    for idx, label in enumerate(final_model.classes_):
        top_indices = np.argsort(final_model.feature_log_prob_[idx])[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"{label}: {', '.join(top_words)}")

    print("\n[STEP 7] SAVE MODEL & ARTIFACTS")
    with open(os.path.join(outdir, "naive_bayes_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
    with open(os.path.join(outdir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(final_vectorizer, f)
    df.to_csv(os.path.join(outdir, "data_preprocessed_labeled.csv"), index=False, sep=";")

    print(f"✓ Semua artefak tersimpan di folder: {outdir}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"""
Best Model    : {best_name}
Accuracy      : {accuracy * 100:.2f}%
Precision     : {precision * 100:.2f}%
Recall        : {recall * 100:.2f}%
F1-Score      : {f1 * 100:.2f}%
CV Accuracy   : {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f}%)

Dataset       : {len(df)} samples
Training      : {len(X_train)} samples
Testing       : {len(X_test)} samples
Vocabulary    : {len(feature_names)} features
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive Bayes text classification pipeline.")
    parser.add_argument("--input", required=True, help="Path CSV hasil preprocessing (harus punya kolom 'text_final').")
    parser.add_argument("--outdir", default="results", help="Folder untuk menyimpan model, plot, dan hasil.")
    args = parser.parse_args()

    run_pipeline(args.input, args.outdir)


if __name__ == "__main__":
    main()
