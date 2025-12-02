# file: recommend.py

import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("Dataset")

# ===============
# 1. LOAD OBJECTS
# ===============

# Business data (same for train/test)
business = pd.read_pickle(data_dir / "business_df.pkl")

# TRAIN reviews only
reviews = pd.read_pickle(data_dir / "reviews_train.pkl")

# User -> city mapping built from TRAIN reviews only
user_city_map = pd.read_pickle(data_dir / "user_city_map.pkl").to_dict()

# Category features (built from all businesses, but user profiles use TRAIN reviews)
business_features = pd.read_pickle(data_dir / "business_features.pkl")
business_features = business_features.set_index("business_id")

# Sanity check
required_biz_cols = {"business_id", "name", "stars", "review_count", "categories", "city"}
required_rev_cols = {"user_id", "business_id", "stars"}

missing_biz = required_biz_cols - set(business.columns)
missing_rev = required_rev_cols - set(reviews.columns)

if missing_biz:
    raise ValueError(f"business_df.pkl missing columns: {missing_biz}")
if missing_rev:
    raise ValueError(f"reviews_train.pkl missing columns: {missing_rev}")

# ===============================
# 2. USER CATEGORY PROFILE (TRAIN)
# ===============================

def build_user_profile(user_id, min_positive_stars=4):
    """
    Build a normalized category preference vector for a user
    from TRAIN reviews only (reviews with stars >= min_positive_stars).
    """
    user_reviews = reviews[reviews["user_id"] == user_id]
    pos_reviews = user_reviews[user_reviews["stars"] >= min_positive_stars]
    if pos_reviews.empty:
        return None

    pos_biz_ids = pos_reviews["business_id"].unique()
    cat_mat = business_features.loc[
        business_features.index.intersection(pos_biz_ids)
    ]
    if cat_mat.empty:
        return None

    profile = cat_mat.mean(axis=0).values.astype(float)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return None
    return profile / norm

def add_category_similarity(candidates, user_profile):
    """
    Add a 'cat_sim' column to candidates, which is cosine similarity
    between each business's category vector and the user profile.
    If no profile exists, all cat_sim are set to 0.
    """
    candidates = candidates.copy()
    if user_profile is None:
        candidates["cat_sim"] = 0.0
        return candidates

    cand_feats = business_features.loc[
        business_features.index.intersection(candidates["business_id"])
    ]
    # Align order with candidates
    cand_feats = cand_feats.reindex(candidates["business_id"])

    feats = cand_feats.values.astype(float)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats_norm = feats / norms

    sims = feats_norm @ user_profile
    candidates["cat_sim"] = sims
    return candidates

# ==============================================
# 3. CATEGORY-FIRST, THEN POPULARITY RECOMMENDER
# ==============================================

def recommend_for_user(user_id, top_n=10, min_review_count=1):
    """
    Recommend restaurants for a user based on:

    1. User's inferred city from TRAIN reviews (user_city_map).
    2. Restaurants in that city the user has not already rated in TRAIN.
    3. Primary sort: category similarity (cat_sim, descending).
    4. Tie-breakers: review_count (descending), then stars (descending).

    All personalization is based only on TRAIN data.
    """

    # 1. City filter from TRAIN-based user_city_map
    user_city = user_city_map.get(user_id)
    if user_city is None:
        print("No inferred city for this user; cannot filter by location.")
        return pd.DataFrame()

    city_biz = business[business["city"] == user_city].copy()
    if city_biz.empty:
        print("No restaurants found in user city:", user_city)
        return pd.DataFrame()

    # 2. Exclude businesses the user already rated in TRAIN
    rated_biz_ids = reviews[reviews["user_id"] == user_id]["business_id"].unique()
    city_biz = city_biz[~city_biz["business_id"].isin(rated_biz_ids)]
    if city_biz.empty:
        print("No unseen restaurants left in user city:", user_city)
        return pd.DataFrame()

    # 3. Popularity threshold
    city_biz = city_biz[city_biz["review_count"] >= min_review_count]
    if city_biz.empty:
        print("No restaurants with at least", min_review_count, "reviews in user city:", user_city)
        return pd.DataFrame()

    # 4. Category profile (TRAIN only) and similarity
    user_profile = build_user_profile(user_id)
    city_biz = add_category_similarity(city_biz, user_profile)

    # 5. Sort: category similarity first, then review_count, then stars
    city_biz = city_biz.sort_values(
        by=["cat_sim", "review_count", "stars"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # 6. Take top_n
    result = city_biz[
        ["business_id", "name", "city", "stars", "review_count", "categories", "cat_sim"]
    ].head(top_n).reset_index(drop=True)

    return result

# =================
# 4. EXAMPLE USAGE
# =================

if __name__ == "__main__":
    # Replace with a real user_id from your TRAIN data
    example_user = "j14WgRoU_-2ZE1aw1dXrJg"
    recs = recommend_for_user(example_user, top_n=10, min_review_count=10)
    print(recs)
