"""Train the model:  python train.py   (run `python fetch_data.py` first)"""
from sentiment_core import train_and_save, MODEL_PATH

_, report = train_and_save()
print(report)
print("Saved model to", MODEL_PATH)
