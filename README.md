# Analisis Sentimen Twitter dengan Naive Bayes

Proyek ini adalah pipeline lengkap untuk melakukan **analisis sentimen data Twitter berbahasa Indonesia**, mulai dari preprocessing teks mentah hingga klasifikasi menggunakan **Multinomial Naive Bayes**.

## 🧩 Alur Proyek

```
data mentah (.csv)
      │
      ▼
[1] preprocessing.py
      │  - case folding (lowercase, hapus mention/hashtag/URL/angka/simbol)
      │  - tokenisasi (NLTK)
      │  - stemming (Sastrawi)
      │  - stopwords removal
      ▼
data_preprocessed.csv
      │
      ▼
[2] naive_bayes_classifier.py
      │  - auto-labeling (positif / negatif / netral) berbasis keyword
      │  - split train/test
      │  - ekstraksi fitur: CountVectorizer vs TF-IDF vs N-gram
      │  - training & pemilihan model terbaik
      │  - evaluasi (accuracy, precision, recall, F1, confusion matrix)
      │  - cross-validation 5-fold
      ▼
model + vectorizer (.pkl) + laporan evaluasi
```

## 📁 Struktur Folder

```
.
├── src/
│   ├── preprocessing.py          # Tahap 1: pembersihan & normalisasi teks
│   └── naive_bayes_classifier.py # Tahap 2: training & evaluasi model
├── requirements.txt
└── README.md
```

## ⚙️ Instalasi

```bash
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO
pip install -r requirements.txt
```

## 🚀 Cara Menjalankan

**1. Preprocessing data mentah**

Data input minimal harus punya kolom `full_text`.

```bash
python src/preprocessing.py --input data_mentah.csv --output data_preprocessed.csv
```

**2. Training & evaluasi model**

```bash
python src/naive_bayes_classifier.py --input data_preprocessed.csv --outdir results/
```

Output yang dihasilkan di folder `results/`:
- `naive_bayes_model.pkl` — model terlatih
- `vectorizer.pkl` — vectorizer (Count/TF-IDF/N-gram) terbaik
- `label_distribution.png` — grafik distribusi label
- `confusion_matrix.png` — confusion matrix model terbaik
- `data_preprocessed_labeled.csv` — data lengkap dengan label

## 🛠️ Teknologi

- **Python** — pandas, numpy
- **NLP**: NLTK (tokenisasi, stopwords), Sastrawi (stemming Bahasa Indonesia)
- **Machine Learning**: scikit-learn (`MultinomialNB`, `CountVectorizer`, `TfidfVectorizer`)
- **Visualisasi**: matplotlib, seaborn

## 📊 Metodologi

Karena dataset tidak memiliki label sentimen bawaan, label dibuat otomatis dengan pendekatan **keyword-based labeling** (kata-kata yang mengindikasikan sentimen positif/negatif). Tiga strategi ekstraksi fitur dibandingkan (Count, TF-IDF, N-gram) dan model dengan akurasi tertinggi dipilih sebagai model final, lalu divalidasi lebih lanjut dengan 5-fold cross-validation.

> Catatan: pendekatan labeling berbasis keyword ini sederhana dan cocok untuk keperluan eksplorasi/portofolio. Untuk keperluan produksi, disarankan menggunakan data berlabel manual atau model labeling yang lebih robust.

## ✍️ Penulis

Dibuat sebagai bagian dari proyek pembelajaran Natural Language Processing & Machine Learning.
