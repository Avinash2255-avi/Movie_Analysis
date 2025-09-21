# train_model.py
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import nltk

# Optional: NLTK downloads used if you expand preprocessing (not strictly required here)
# nltk.download('punkt')

DATA_PATH = "IMDB Dataset.csv"   # ensure this file exists in backend/
MODEL_PATH = "sentiment_model.joblib"

def clean_text(s: str) -> str:
    """Lightweight text cleanup: remove HTML tags, non-alphanumerics, lowercase."""
    s = re.sub(r"<.*?>", " ", s)                 # remove HTML tags
    s = re.sub(r"[^a-zA-Z0-9\s']", " ", s)       # keep alphanumerics and apostrophes
    s = re.sub(r"\s+", " ", s)                   # collapse whitespace
    return s.strip().lower()

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)  # columns: review, sentiment

    # Quick sanity check
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise RuntimeError("CSV must contain 'review' and 'sentiment' columns")

    # Clean text (this is fast)
    print("Cleaning text (this may take a short while)...")
    df['clean'] = df['review'].astype(str).map(clean_text)

    # Map labels: pos -> 1, neg -> 0
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['clean']
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Build pipeline: TF-IDF -> Logistic Regression
    print("Building pipeline and training model...")
    pipeline = make_pipeline(
        TfidfVectorizer(
            max_features=30000,     # use up to 30k features; adjust if memory is tight
            ngram_range=(1,2),      # unigrams + bigrams
            stop_words='english',
            min_df=3
        ),
        LogisticRegression(
            C=4.0,
            max_iter=1000,
            solver='saga',          # good for large sparse data; requires scikit-learn >=0.22
            n_jobs=-1
        )
    )

    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    acc = metrics.accuracy_score(y_test, preds)
    f1 = metrics.f1_score(y_test, preds)
    print(f"Accuracy on test set: {acc:.4f}")
    print(f"F1-score on test set: {f1:.4f}")

    # Save model pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Quick sanity example
    examples = [
        "The film was fantastic and I loved it",
        "It was awful and I hated the movie"
    ]
    print("Example predictions:")
    for ex in examples:
        pred = pipeline.predict([clean_text(ex)])[0]
        conf = pipeline.predict_proba([clean_text(ex)])[0].max()
        label = "Positive" if pred == 1 else "Negative"
        print(f"  {ex!r} -> {label} ({conf:.2%})")

if __name__ == "__main__":
    main()
