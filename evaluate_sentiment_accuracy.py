import pandas as pd

from sentiment_analysis import predict_sentiment

# ===========
# 1. LOAD DATA
# ===========
file_path = "Dataset/filtered_yelp_review.json"
df = pd.read_json(file_path, lines=True, nrows=1000)

print("Loaded reviews:", df.shape)

# Keep only needed columns
df = df[["text", "stars"]].dropna()

# ===========
# 2. RUN MODEL ON DATA
# ===========
correct = 0
total = len(df)

for idx, row in df.iterrows():
    text = row["text"]
    true_rating = row["stars"]

    pred_rating = predict_sentiment(text)

    # If model rating is within ±1 of true rating → correct
    if abs(pred_rating - true_rating) <= 0:
        correct += 1

    # Optional progress output every 1000 rows
    if idx % 1000 == 0:
        print(f"Processed {idx}/{total}")

# ===========
# 3. ACCURACY
# ===========
accuracy = correct / total
print("\n==============================")
print(f"Accuracy (±1 star tolerance): {accuracy:.4f}")
print("==============================\n")