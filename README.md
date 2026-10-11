# Customer Review Sentiment Demo

Classifies a review as positive / negative / neutral with a model trained on
public datasets (Twitter US Airline Sentiment + NLTK movie_reviews).

## Run (3 commands, from inside this folder)
```
pip install -r requirements.txt
python train.py      # downloads data, trains (~1 min), saves model.joblib
python app.py        # then open http://127.0.0.1:5000 in your browser
```
On Windows use `py` instead of `python` if `python` is not recognised.

* Open the page through the URL above. Double-clicking templates/index.html will NOT work.
* If model.joblib is missing or was made by a different scikit-learn version,
  `app.py` trains a fresh one automatically on first start.

## Files
* sentiment_core.py - all NLP logic: cleaning, training, prediction
* train.py          - trains and prints accuracy report
* app.py + templates/index.html - the demo website

## Using the page
* Type one review and press Analyze -> shows only the label.
* Or choose a CSV file (needs a header row; the column named review/text/comment/feedback... is used) -> table + download button. Try sample.csv.
