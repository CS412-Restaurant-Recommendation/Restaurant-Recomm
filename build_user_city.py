# file: build_user_city.py
import pandas as pd
from pathlib import Path

data_dir = Path("Dataset")
biz_path = data_dir / "filtered_yelp_business.json"
rev_path = data_dir / "filtered_yelp_review.json"

business = pd.read_json(biz_path, lines=True)
reviews = pd.read_json(rev_path, lines=True)

# Keep only needed columns
business = business[["business_id", "city"]].dropna(subset=["business_id", "city"])
reviews = reviews[["user_id", "business_id", "stars"]].dropna(subset=["user_id", "business_id"])

# Join to attach city to each review
rev_with_city = reviews.merge(business, on="business_id", how="inner")

# For each user, count reviews per city and pick the most frequent city
user_city = (
    rev_with_city
    .groupby(["user_id", "city"])
    .size()
    .reset_index(name="count")
)

# For each user_id, keep the city with max count
user_city = (
    user_city.sort_values(["user_id", "count"], ascending=[True, False])
    .drop_duplicates(subset=["user_id"], keep="first")
    .reset_index(drop=True)
)

# user_city: columns ['user_id', 'city', 'count']
user_city_map = dict(zip(user_city["user_id"], user_city["city"]))

# Save mapping
user_city.to_pickle(data_dir / "user_city_df.pkl")
pd.Series(user_city_map).to_pickle(data_dir / "user_city_map.pkl")

print("Saved user -> city mapping for", len(user_city_map), "users.")
