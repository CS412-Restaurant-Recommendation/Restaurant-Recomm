# file: new_user_recommendation.py

import pandas as pd
from pathlib import Path

# Resolve relative to this file so imports work from any working directory
data_dir = Path(__file__).resolve().parent / "Dataset"

# Load processed business data once (same table recommend.py uses)
business = pd.read_pickle(data_dir / "business_df.pkl")

# Ensure numeric columns
business["review_count"] = pd.to_numeric(business["review_count"], errors="coerce")
business["stars"] = pd.to_numeric(business["stars"], errors="coerce")
business = business.dropna(subset=["stars", "review_count"])

# Weighted rating (IMDB-style), computed once over all businesses
C = business["stars"].mean()
m = 300  # minimum review count threshold; tune if needed

v = business["review_count"]
R = business["stars"]

business["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C


def filter_by_categories(df, categories):
    """
    Keep rows whose category_list contains at least one of the given categories
    (case-insensitive exact match, e.g. ["italian", "indian"]).
    """
    wanted = {c.strip().lower() for c in categories if c and c.strip()}
    if not wanted:
        return df
    mask = df["category_list"].apply(
        lambda cats: isinstance(cats, list) and not wanted.isdisjoint(cats)
    )
    return df[mask]


def get_new_user_recommendations(top_n: int = 20, city=None, categories=None, exclude_biz_ids=None):
    df = business
    if city:
        df = df[df["city"].str.lower() == city.strip().lower()]
    if categories:
        df = filter_by_categories(df, categories)
    if exclude_biz_ids:
        df = df[~df["business_id"].isin(exclude_biz_ids)]

    top = df.sort_values("weighted_rating", ascending=False).head(top_n)

    # Return only the fields you care about as list[dict]
    return top[["business_id", "name", "city", "stars", "review_count", "weighted_rating", "categories"]].to_dict(orient="records")
