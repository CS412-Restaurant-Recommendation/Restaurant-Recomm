import pandas as pd

# Load Yelp JSON lines file
df = pd.read_json("Dataset/filtered_yelp_business.json", lines=True)

# Ensure numeric columns
df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

df = df.dropna(subset=["stars", "review_count"])

# Weighted rating
C = df["stars"].mean()
m = 300

v = df["review_count"]
R = df["stars"]

df["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

top20 = df.sort_values("weighted_rating", ascending=False).head(20)

print(top20[["name", "stars", "review_count", "weighted_rating", "categories"]])