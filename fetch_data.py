"""Optional: download the datasets now (train.py also does this automatically)."""
from sentiment_core import ensure_data

ensure_data()
print("Datasets ready in ./data")
