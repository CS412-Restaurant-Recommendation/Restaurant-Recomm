# file: new_user_recommendation.py

import pandas as pd
from pathlib import Path

data_dir = Path("Dataset")

def get_new_user_recommendations(top_n: int = 20):
    # Load Yelp JSON lines file
    df = pd.read_json(data_dir / "filtered_yelp_business.json", lines=True)

    # Ensure numeric columns
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars", "review_count"])

    # Weighted rating (IMDB-style)
    C = df["stars"].mean()
    m = 300  # minimum review count threshold; tune if needed

    v = df["review_count"]
    R = df["stars"]

    df["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

    top = df.sort_values("weighted_rating", ascending=False).head(top_n)

    # Return only the fields you care about as list[dict]
    return top[["business_id", "name", "stars", "review_count", "weighted_rating", "categories"]].to_dict(orient="records")
