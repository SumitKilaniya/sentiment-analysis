
# Sentiment Analysis on Movie Reviews

A natural language processing project that classifies movie reviews as **positive** or **negative** using classic NLP preprocessing and machine learning models (Naive Bayes and Linear SVM) trained on TF-IDF features.

## Overview

This project implements an end-to-end text classification pipeline:

1. Load and explore a labeled movie review dataset
2. Clean and preprocess raw review text (HTML removal, lowercasing, tokenization, stopword removal, lemmatization)
3. Convert text into numerical features using TF-IDF vectorization
4. Train and evaluate two classifiers — **Multinomial Naive Bayes** and **Linear SVM**
5. Use the trained model to predict sentiment on new, unseen reviews

## Project Structure

```
.
├── sentiment_analysis.ipynb   # Main notebook with the full pipeline
├── requirements.txt           # Python dependencies
├── data/
│   └── reviews.csv            # Dataset (not included — see Dataset section)
└── README.md
```

## Dataset

The notebook expects a CSV file at `../data/reviews.csv` (relative to the notebook) with at least the following columns:

| Column      | Description                                  |
|-------------|-----------------------------------------------|
| `review`    | Raw text of the movie review                  |
| `sentiment` | Label — `positive` or `negative`              |

Place your dataset at `data/reviews.csv` relative to the project root before running the notebook.

## Installation

1. Clone or download this repository.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter:
   ```bash
   jupyter notebook sentiment_analysis.ipynb
   ```

The notebook downloads the required NLTK corpora (`punkt`, `stopwords`, `wordnet`, `omw-1.4`) automatically on first run.

## Methodology

### Text Preprocessing
- Strip HTML tags and lowercase all text
- Tokenize using NLTK's `word_tokenize`
- Remove stopwords, while explicitly keeping negation words (`not`, `no`, `nor`) to preserve sentiment-bearing context
- Lemmatize tokens with `WordNetLemmatizer`

### Feature Extraction
- **TF-IDF Vectorization** (`TfidfVectorizer`, top 5,000 features) converts cleaned text into numerical vectors

### Modeling
Two classifiers are trained on an 80/20 stratified train-test split:

| Model                  | Description                              |
|-------------------------|-------------------------------------------|
| Multinomial Naive Bayes | Fast probabilistic baseline classifier    |
| Linear SVM              | Higher-capacity linear classifier         |

### Evaluation
Each model is evaluated using:
- Accuracy
- Precision (positive class)
- Recall (positive class)
- Full classification report (precision, recall, F1-score per class)

## Usage

After training, the notebook exposes a helper function to classify new text:

```python
predict_review("This movie was absolutely amazing, I loved every minute.")
# → 'positive'

predict_review("This was a terrible waste of time and the acting was awful.")
# → 'negative'
```

The function applies the same preprocessing pipeline (cleaning, tokenization, stopword removal, lemmatization) and TF-IDF transformation used during training, then predicts using the trained SVM model.

## Dependencies

- `pandas` — data loading and manipulation
- `nltk` — tokenization, stopwords, lemmatization
- `scikit-learn` — TF-IDF vectorization, model training, and evaluation

See `requirements.txt` for the full list.

## Future Improvements

- Add cross-validation and hyperparameter tuning (e.g., `GridSearchCV`)
- Experiment with word embeddings (Word2Vec, GloVe) or transformer-based models (BERT)
- Add a confusion matrix and visualizations for model comparison
- Package the trained model and vectorizer for reuse (e.g., via `joblib`)
- Build a simple CLI or web interface for real-time predictions

## License

This project is available for educational and research purposes. Add a license of your choice (e.g., MIT) if distributing publicly.
