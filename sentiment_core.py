"""
sentiment_core.py - core NLP logic (training + prediction)
==========================================================
Classifies a customer review as POSITIVE, NEGATIVE or NEUTRAL using a
supervised model trained on public labelled datasets.

Pipeline (scikit-learn)
-----------------------
    raw text -> clean_text() -> TF-IDF (word 1-2 grams + character 3-5 grams)
             -> Logistic Regression (multinomial, class-balanced)

* TF-IDF turns text into numeric features by weighting words/phrases by how
  informative they are. Word bigrams capture phrases like "not good"; character
  n-grams make the model robust to typos and slang ("gr8", "worstt").
* Logistic Regression learns, from labelled examples, which features point to
  which class, and outputs a probability for each of the three classes.
* class_weight="balanced" stops the model from favouring the majority class
  (the airline data is ~63% negative).

Public API
----------
    train_and_save()          -> trains on ./data, evaluates, saves model.joblib
    SentimentClassifier()     -> loads model.joblib
        .predict(text)        -> dict with label, confidence, per-class probs

Honest note: no ML model is "unmistakable" (100% correct). `predict()` returns
the probability of each class and a `low_confidence` flag when the top two
classes are close, so uncertain reviews are visible instead of hidden.
"""
from __future__ import annotations

import html
import pathlib
import re
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

BASE = pathlib.Path(__file__).parent
DATA_DIR = BASE / "data"
MODEL_PATH = BASE / "model.joblib"
LABELS = ["negative", "neutral", "positive"]
MAX_CHARS = 5000
#: If the best class has probability below this, the prediction is flagged.
LOW_CONFIDENCE_THRESHOLD = 0.55
RANDOM_STATE = 42

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise a review: unescape HTML, drop tags/URLs/@mentions, lowercase,
    collapse whitespace. Punctuation is kept (e.g. '!' and ':)' carry signal)."""
    text = html.unescape(str(text))
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
DATA_URLS = {
    "tweets.csv": "https://raw.githubusercontent.com/richardcsuwandi/airline-sentiment/master/tweets.csv",
    "movie_reviews.zip": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/movie_reviews.zip",
}


def ensure_data() -> None:
    """Download the public datasets into ./data if they are not already there,
    so training works even on a fresh copy of the project."""
    import urllib.request
    import zipfile
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in DATA_URLS.items():
        dest = DATA_DIR / name
        if not dest.exists():
            print(f"Downloading {name} ...")
            dest.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    if not (DATA_DIR / "movie_reviews").exists():
        zipfile.ZipFile(DATA_DIR / "movie_reviews.zip").extractall(DATA_DIR)

def load_airline() -> pd.DataFrame:
    """Twitter US Airline Sentiment: 3 classes, 14,640 rows."""
    df = pd.read_csv(DATA_DIR / "tweets.csv", usecols=["text", "airline_sentiment"])
    return df.rename(columns={"airline_sentiment": "label"})


def load_movie_reviews() -> pd.DataFrame:
    """NLTK movie_reviews: positive/negative only, 2,000 longer reviews."""
    rows = []
    for label, folder in (("positive", "pos"), ("negative", "neg")):
        for f in sorted((DATA_DIR / "movie_reviews" / folder).glob("*.txt")):
            rows.append({"text": f.read_text(errors="ignore"), "label": label})
    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_and_save(use_movie_reviews: bool = True) -> Tuple[Pipeline, str]:
    """Train, evaluate on a held-out 20% of the airline data, save the model.

    The held-out test set is never seen during training, so the reported
    accuracy is an honest estimate for unseen reviews *of that kind*.
    Returns (fitted_pipeline, report_text).
    """
    ensure_data()
    airline = load_airline()
    airline["text"] = airline["text"].map(clean_text)
    train_df, test_df = train_test_split(
        airline, test_size=0.2, stratify=airline["label"], random_state=RANDOM_STATE)

    if use_movie_reviews:
        movies = load_movie_reviews()
        movies["text"] = movies["text"].map(clean_text)
        train_df = pd.concat([train_df, movies], ignore_index=True)

    model = build_pipeline().fit(train_df["text"], train_df["label"])

    pred = model.predict(test_df["text"])
    report = classification_report(test_df["label"], pred, labels=LABELS, digits=3)
    cm = pd.DataFrame(confusion_matrix(test_df["label"], pred, labels=LABELS),
                      index=[f"true_{l}" for l in LABELS],
                      columns=[f"pred_{l}" for l in LABELS])
    text = (f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}\n\n"
            f"{report}\nConfusion matrix:\n{cm}\n")
    joblib.dump(model, MODEL_PATH)
    return model, text


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
class SentimentClassifier:
    """Loads the trained model once; call .predict() for each new review."""

    def __init__(self, path: pathlib.Path = MODEL_PATH) -> None:
        try:
            self.model: Pipeline = joblib.load(path)
        except Exception as exc:  # missing file, or saved by another scikit-learn version
            print(f"No usable saved model ({type(exc).__name__}); training a new one "
                  "(takes about a minute, one time only) ...")
            self.model, report = train_and_save()
            print(report)

    def predict(self, review: str) -> Dict:
        """Classify one review.

        Returns:
            label          'positive' | 'negative' | 'neutral'
            confidence     probability of the chosen label (0-1)
            probabilities  {label: probability} for all three classes
            low_confidence True if confidence < LOW_CONFIDENCE_THRESHOLD
        Raises ValueError for empty / non-string / over-long input.
        """
        if not isinstance(review, str) or not clean_text(review):
            raise ValueError("Please enter a non-empty review.")
        if len(review) > MAX_CHARS:
            raise ValueError(f"Review too long (max {MAX_CHARS} characters).")
        probs = self.model.predict_proba([clean_text(review)])[0]
        classes: List[str] = list(self.model.classes_)
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "label": classes[best],
            "confidence": round(float(probs[best]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
            "low_confidence": bool(probs[best] < LOW_CONFIDENCE_THRESHOLD),
        }

    def predict_many(self, reviews: List[str]) -> List[str]:
        """Classify many reviews at once (fast, vectorised).

        Returns one label per input, in the same order. Blank / non-string
        entries get the label 'n/a'. Over-long reviews are truncated to
        MAX_CHARS instead of being rejected, so one bad row never stops a file.
        """
        cleaned = [clean_text(r[:MAX_CHARS]) if isinstance(r, str) else "" for r in reviews]
        labels = ["n/a"] * len(cleaned)
        valid = [i for i, c in enumerate(cleaned) if c]
        if valid:
            predictions = self.model.predict([cleaned[i] for i in valid])
            for i, label in zip(valid, predictions):
                labels[i] = str(label)
        return labels
