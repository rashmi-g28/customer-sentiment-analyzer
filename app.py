"""Minimal demo website:  python app.py  ->  http://127.0.0.1:5000

Two inputs are supported:
  1. a single review typed into the text box  -> returns just the label
  2. a CSV file containing many reviews       -> returns a label for every row
"""
import io

import pandas as pd
from flask import Flask, jsonify, render_template, request

from sentiment_core import SentimentClassifier

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # reject uploads over 5 MB
classifier = SentimentClassifier()                    # load the trained model once

MAX_ROWS = 10000
# Column names (lower-case) that are treated as the review text.
REVIEW_COLUMNS = ("review", "reviews", "review_text", "reviewtext", "text",
                  "comment", "comments", "feedback", "content", "body", "description")


def pick_review_column(df: pd.DataFrame):
    """Choose the column holding the review text: a column with a well-known
    name if there is one, otherwise the column with the longest average text."""
    for col in df.columns:
        if str(col).strip().lower() in REVIEW_COLUMNS:
            return col
    return max(df.columns, key=lambda c: df[c].str.len().mean())


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    """Single review -> {"label": "..."} only."""
    review = (request.get_json(silent=True) or {}).get("review", "")
    try:
        return jsonify(label=classifier.predict(review)["label"])
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.post("/api/predict_csv")
def predict_csv():
    """CSV upload -> {"column": "...", "results": [{"review": ..., "sentiment": ...}, ...]}"""
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="No file uploaded."), 400
    raw = file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return jsonify(error="Could not read this file as a CSV."), 400
    if df.empty or len(df.columns) == 0:
        return jsonify(error="The CSV has no rows."), 400
    if len(df) > MAX_ROWS:
        return jsonify(error=f"The CSV has {len(df)} rows; the maximum is {MAX_ROWS}."), 400

    column = pick_review_column(df)
    reviews = df[column].tolist()
    labels = classifier.predict_many(reviews)
    return jsonify(
        column=str(column),
        results=[{"review": r, "sentiment": s} for r, s in zip(reviews, labels)],
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large (maximum 5 MB)."), 413


if __name__ == "__main__":
    app.run(debug=False)
