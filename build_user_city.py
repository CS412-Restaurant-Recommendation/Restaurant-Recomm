# file: build_user_city.py
import pandas as pd
from pathlib import Path

data_dir = Path("Dataset")

business = pd.read_pickle(data_dir / "business_df.pkl")
reviews = pd.read_pickle(data_dir / "reviews_df.pkl")

# Ensure columns exist
business = business[["business_id", "city"]].dropna(subset=["business_id", "city"])
reviews = reviews[["user_id", "business_id", "stars"]].dropna(subset=["user_id", "business_id"])

# Join to attach city to each review
rev_with_city = reviews.merge(business, on="business_id", how="inner")
print("rev_with_city rows:", rev_with_city.shape[0])

if rev_with_city.empty:
    print("Joined reviews have no rows; cannot infer user cities.")
else:
    user_city = (
        rev_with_city
        .groupby(["user_id", "city"])
        .size()
        .reset_index(name="count")
    )

    # Most frequent city per user
    user_city = (
        user_city.sort_values(["user_id", "count"], ascending=[True, False])
        .drop_duplicates(subset=["user_id"], keep="first")
        .reset_index(drop=True)
    )

    user_city_map = dict(zip(user_city["user_id"], user_city["city"]))

    user_city.to_pickle(data_dir / "user_city_df.pkl")
    pd.Series(user_city_map).to_pickle(data_dir / "user_city_map.pkl")

    print("Saved user -> city mapping for", len(user_city_map), "users.")
