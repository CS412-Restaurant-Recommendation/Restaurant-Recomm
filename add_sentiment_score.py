import pandas as pd
from pathlib import Path

from sentiment_analysis import predict_sentiment

# ===========
# 1. LOAD DATA
# ===========
data_dir = Path("Dataset")
input_path = data_dir / "filtered_yelp_review.json"
output_path = data_dir / "filtered_yelp_review_with_sentiment.json"

# Adjust nrows or chunksize if needed
df = pd.read_json(input_path, lines=True, nrows=200)

print("Loaded reviews:", df.shape)

# ===========
# 2. COMPUTE SENTIMENT SCORE
#    (add a new column with model prediction)
# ===========
def safe_predict(text):
    if not isinstance(text, str) or not text.strip():
        return None
    return predict_sentiment(text)

df["sentiment_score"] = df["text"].apply(safe_predict)

print("Example rows with sentiment:")
print(df[["text", "sentiment_score"]].head())

# ===========
# 3. SAVE TO NEW JSONL FILE
#    same format (one JSON object per line) + new column
# ===========
df.to_json(output_path, orient="records", lines=True)

print(f"Saved scored reviews to: {output_path}")
