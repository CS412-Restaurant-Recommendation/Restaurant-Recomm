# file: build_profiles.py
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data_dir = Path("Dataset")
business_path = data_dir / "filtered_yelp_business.json"
review_path = data_dir / "filtered_yelp_review.json"

# ===========
# 1. LOAD DATA
# ===========
business = pd.read_json(business_path, lines=True)
reviews = pd.read_json(review_path, lines=True)

# Keep only needed columns
business = business[["business_id", "name", "stars", "review_count", "categories"]]
reviews = reviews[["user_id", "business_id", "stars"]]

# Drop broken rows
business = business.dropna(subset=["business_id", "categories"])
reviews = reviews.dropna(subset=["user_id", "business_id", "stars"])

# ===========
# 2. PROCESS CATEGORIES → MULTI-LABEL
# ===========
def split_categories(cat_str):
    if not isinstance(cat_str, str):
        return []
    return [c.strip().lower() for c in cat_str.split(",")]

business["category_list"] = business["categories"].apply(split_categories)

mlb = MultiLabelBinarizer()
cat_matrix = mlb.fit_transform(business["category_list"])
cat_feature_names = mlb.classes_

# Turn into DataFrame
cat_df = pd.DataFrame(cat_matrix, columns=cat_feature_names, index=business.index)

# ===========
# 3. RESTAURANT FEATURE MATRIX
#    (categories only; stars/review_count used later for re-ranking)
# ===========
business_features = pd.concat([business[["business_id"]], cat_df], axis=1)

# Map business_id → row index for fast lookup
bizid_to_idx = {bid: i for i, bid in enumerate(business_features["business_id"])}

# Save artifacts for later use (optional, or keep in memory if running once)
business.to_pickle(data_dir / "business_df.pkl")
business_features.to_pickle(data_dir / "business_features.pkl")
pd.Series(bizid_to_idx).to_pickle(data_dir / "bizid_to_idx.pkl")
pd.Series(cat_feature_names).to_pickle(data_dir / "cat_feature_names.pkl")
reviews.to_pickle(data_dir / "reviews_df.pkl")

print("Profiles built and saved.")