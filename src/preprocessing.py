"""
Preprocessing Data Twitter untuk Naive Bayes Classifier
=========================================================
Pipeline preprocessing teks Bahasa Indonesia:
  1. Case folding (lowercase + cleaning mention/hashtag/URL/angka/simbol)
  2. Tokenisasi (NLTK)
  3. Stemming (Sastrawi)
  4. Stopwords removal (NLTK + custom stopwords)
  5. Filter data yang terlalu pendek & simpan hasil akhir

Cara pakai:
    python preprocessing.py --input data_mentah.csv --output data_preprocessed.csv

Kolom input yang dibutuhkan: 'full_text'
"""

import argparse
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

CUSTOM_STOPWORDS = {"nya", "kah", "lah", "pun", "yg", "dgn", "krn", "utk", "dlm"}


def download_nltk_resources() -> None:
    """Download resource NLTK yang dibutuhkan (idempotent)."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


def case_folding(text: str) -> str:
    """Lowercase + hapus mention, hashtag, URL, angka, dan karakter spesial."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"@\w+", "", text)  # hapus mention
    text = re.sub(r"#\w+", "", text)  # hapus hashtag
    text = re.sub(r"http\S+|www\S+", "", text)  # hapus URL
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = re.sub(r"[^\w\s]", "", text)  # hapus karakter spesial
    text = re.sub(r"\s+", " ", text).strip()  # rapikan whitespace

    return text


def tokenize_text(text: str) -> list:
    """Tokenisasi teks menjadi list kata."""
    return word_tokenize(text)


def stem_tokens(tokens: list, stemmer) -> list:
    """Stemming setiap token ke bentuk dasar (Sastrawi)."""
    return [stemmer.stem(token) for token in tokens]


def remove_stopwords(tokens: list, stop_words: set) -> list:
    """Hapus stopwords dari list token."""
    return [token for token in tokens if token not in stop_words]


def run_pipeline(input_path: str, output_path: str, min_tokens: int = 3) -> pd.DataFrame:
    """Jalankan seluruh pipeline preprocessing dan simpan hasilnya ke CSV."""
    download_nltk_resources()

    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)
    df = pd.read_csv(input_path, engine="python", encoding="utf-8-sig")
    print(f"Jumlah data: {len(df)} tweets")
    print(f"Kolom yang tersedia: {df.columns.tolist()}")

    print("\n" + "=" * 80)
    print("CASE FOLDING")
    print("=" * 80)
    df["text_casefolding"] = df["full_text"].apply(case_folding)

    print("\n" + "=" * 80)
    print("TOKENISASI")
    print("=" * 80)
    df["text_tokens"] = df["text_casefolding"].apply(tokenize_text)
    avg_tokens = df["text_tokens"].apply(len).mean()
    print(f"Rata-rata token per tweet: {avg_tokens:.2f}")

    print("\n" + "=" * 80)
    print("STEMMING (Sastrawi)")
    print("=" * 80)
    stemmer = StemmerFactory().create_stemmer()
    df["text_stemmed"] = df["text_tokens"].apply(lambda t: stem_tokens(t, stemmer))

    print("\n" + "=" * 80)
    print("STOPWORDS REMOVAL")
    print("=" * 80)
    stop_words_id = set(stopwords.words("indonesian")) | CUSTOM_STOPWORDS
    df["text_cleaned"] = df["text_stemmed"].apply(lambda t: remove_stopwords(t, stop_words_id))

    print("\n" + "=" * 80)
    print("FINALISASI")
    print("=" * 80)
    df["text_final"] = df["text_cleaned"].apply(lambda x: " ".join(x))

    df_filtered = df[df["text_cleaned"].apply(len) >= min_tokens].copy()
    print(f"Data sebelum filter: {len(df)}")
    print(f"Data setelah filter (min {min_tokens} token): {len(df_filtered)}")

    df_filtered.to_csv(output_path, index=False, sep=";")
    print(f"\n✓ Preprocessing selesai. Hasil disimpan ke: {output_path}")

    return df_filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing data Twitter untuk Naive Bayes.")
    parser.add_argument("--input", required=True, help="Path CSV mentah (harus punya kolom 'full_text').")
    parser.add_argument("--output", default="data_preprocessed.csv", help="Path CSV hasil preprocessing.")
    parser.add_argument("--min-tokens", type=int, default=3, help="Minimum jumlah token agar baris dipertahankan.")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.min_tokens)


if __name__ == "__main__":
    main()
