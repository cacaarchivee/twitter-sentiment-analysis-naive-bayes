# Twitter Sentiment Analysis with Naive Bayes

This project is a complete pipeline for performing **sentiment analysis on Indonesian-language Twitter data**, from raw text preprocessing to classification using **Multinomial Naive Bayes**.

## 🧩 Project Flow

```
raw data (.csv)
      │
      ▼
[1] preprocessing.py
      │  - case folding (lowercase, remove mentions/hashtags/URLs/numbers/symbols)
      │  - tokenization (NLTK)
      │  - stemming (Sastrawi)
      │  - stopwords removal
      ▼
data_preprocessed.csv
      │
      ▼
[2] naive_bayes_classifier.py
      │  - keyword-based auto-labeling (positive / negative / neutral)
      │  - train/test split
      │  - feature extraction: CountVectorizer vs TF-IDF vs N-gram
      │  - training & best-model selection
      │  - evaluation (accuracy, precision, recall, F1, confusion matrix)
      │  - 5-fold cross-validation
      ▼
model + vectorizer (.pkl) + evaluation report
```

## 📁 Folder Structure

```
.
├── src/
│   ├── preprocessing.py          # Stage 1: text cleaning & normalization
│   └── naive_bayes_classifier.py # Stage 2: model training & evaluation
├── img/                          # Documentation screenshots
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/USERNAME/REPO-NAME.git
cd REPO-NAME
pip install -r requirements.txt
```

## 🚀 How to Run

**1. Preprocess raw data**

The input data must at least have a `full_text` column.

```bash
python src/preprocessing.py --input raw_data.csv --output data_preprocessed.csv
```

**2. Train & evaluate the model**

```bash
python src/naive_bayes_classifier.py --input data_preprocessed.csv --outdir results/
```

Output generated in the `results/` folder:
- `naive_bayes_model.pkl` — trained model
- `vectorizer.pkl` — best vectorizer (Count/TF-IDF/N-gram)
- `label_distribution.png` — label distribution chart
- `confusion_matrix.png` — confusion matrix of the best model
- `data_preprocessed_labeled.csv` — full data with labels

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **NLP**: NLTK (tokenization, stopwords), Sastrawi (Indonesian stemming)
- **Machine Learning**: scikit-learn (`MultinomialNB`, `CountVectorizer`, `TfidfVectorizer`)
- **Visualization**: matplotlib, seaborn

## 📊 Methodology

Since the dataset doesn't come with built-in sentiment labels, labels are generated automatically using a **keyword-based labeling** approach (words indicating positive/negative sentiment). Three feature extraction strategies are compared (Count, TF-IDF, N-gram), and the model with the highest accuracy is selected as the final model, then further validated with 5-fold cross-validation.

> Note: this keyword-based labeling approach is simple and suitable for exploration/portfolio purposes. For production use, manually labeled data or a more robust labeling model is recommended.

## 📸 Documentation

Below are screenshots of each pipeline stage running in Google Colab.

### Stage 1 — Preprocessing (`preprocessing.py`)

| | |
|---|---|
| ![Data upload & initial info](img/python1.png) **1. Data upload & initial info**<br>Upload the raw CSV file, install & import libraries, check data count (459 tweets) and the `full_text` column. | ![Case folding](img/python2.png) **2. Case folding**<br>Sample raw text and the result after case folding (lowercase, removal of symbols/URLs/numbers). |
| ![Tokenization & stemming](img/python3.png) **3. Tokenization & stemming**<br>Tokenization results (13,706 tokens, average 29.86 tokens/tweet) and stemming using Sastrawi. | ![Final result preview](img/python4.png) **4. Final result preview**<br>Final data after filtering (455 tweets), complete with `text_casefolding`, `text_tokens`, `text_stemmed`, `text_cleaned`, `text_final` columns. |

![Preprocessing complete](img/python5.png)
**5. Preprocessing complete** — comparison of original vs. final text for 3 sample records, ready to be downloaded as `data_preprocessed.csv`.

### Stage 2 — Naive Bayes Classification (`naive_bayes_classifier.py`)

| | |
|---|---|
| ![Data loading & labeling](img/python6.png) **6. Data loading & labeling**<br>Upload preprocessed data (455 records) and keyword-based auto-labeling: negative (322), neutral (121), positive (12). | ![Label distribution](img/python7.png) **7. Label distribution**<br>Visualization of record counts per sentiment class. |
| ![Training & evaluation](img/python8.png) **8. Training & evaluation**<br>Comparison of Count vs. TF-IDF vs. N-gram — the best model (Count) achieves 83.52% accuracy with per-class precision/recall/F1-score. | ![Confusion matrix](img/python9.png) **9. Confusion matrix**<br>Confusion matrix of the best model (Count vectorizer) on the test set. |
| ![Cross-validation & prediction](img/python10.png) **10. Cross-validation & prediction**<br>5-fold cross-validation results (80.66% ± 5.23%), top features per class, and sample predictions on new text. | ![Results summary](img/python11.png) **11. Results summary**<br>Saved model & vectorizer (`.pkl`), plus final summary: 455 data samples, 364 training, 91 testing, 695 vocabulary features. |

## ✍️ Author

Created as part of a Natural Language Processing & Machine Learning learning project.