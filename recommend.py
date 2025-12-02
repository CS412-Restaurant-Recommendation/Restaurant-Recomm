# file: recommend.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

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

# Precomputed CF artifacts (built offline from TRAIN only)
# user_item_cf.pkl: user-item rating matrix (reduced)
# user_index_cf.pkl: mapping user_id -> row index in user_item
try:
    user_item = pd.read_pickle(data_dir / "user_item_cf.pkl")
    user_index = pd.read_pickle(data_dir / "user_index_cf.pkl").to_dict()
    # NEW: replace NaNs with 0 for CF math
    user_item = user_item.fillna(0.0)
except FileNotFoundError:
    # Fallback: CF disabled if artifacts not present
    user_item = pd.DataFrame()
    user_index = {}

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

# =========================================
# 2a. USER-BASED COLLABORATIVE FILTERING
# =========================================

def compute_cf_scores_for_user(user_id, candidate_biz_ids, k_neighbors=30):
    """
    Compute a collaborative-filtering score (cf_score) for each candidate business
    for a given user_id, using user-based CF over the precomputed TRAIN subset.

    - Uses cosine similarity between the target user and all other users.
    - Aggregates neighbors' ratings as a weighted average for each candidate.
    - Returns a DataFrame with columns: business_id, cf_score.
    """
    # If matrix is empty or user not represented, return zeros
    if user_item.empty or user_id not in user_index:
        return pd.DataFrame({
            "business_id": list(candidate_biz_ids),
            "cf_score": np.zeros(len(candidate_biz_ids), dtype=float)
        })

    # Extract target user vector
    target_row_idx = user_index[user_id]
    target_vec = user_item.values[target_row_idx:target_row_idx+1]  # shape (1, n_items)

    # Ensure no NaNs (defensive; user_item already filled)
    target_vec = np.nan_to_num(target_vec, copy=False)
    all_users_mat = np.nan_to_num(user_item.values, copy=False)

    # Compute similarity with all users (reduced set)
    sims = cosine_similarity(target_vec, all_users_mat)[0]

    sim_series = pd.Series(sims, index=user_item.index)
    # Drop self
    sim_series = sim_series.drop(labels=[user_id], errors="ignore")
    # Keep top-k neighbors with positive similarity
    sim_series = sim_series[sim_series > 0].nlargest(k_neighbors)

    if sim_series.empty:
        return pd.DataFrame({
            "business_id": list(candidate_biz_ids),
            "cf_score": np.zeros(len(candidate_biz_ids), dtype=float)
        })

    # Neighbor ratings submatrix
    neighbor_ratings = user_item.loc[sim_series.index]
    sims_values = sim_series.values.reshape(-1, 1)  # column vector

    cf_scores = []
    for bid in candidate_biz_ids:
        if bid not in neighbor_ratings.columns:
            cf_scores.append(0.0)
            continue

        col = neighbor_ratings[bid].values.reshape(-1, 1)  # neighbor ratings
        mask = ~np.isnan(col)
        if not mask.any():
            cf_scores.append(0.0)
            continue

        weights = sims_values[mask]
        vals = col[mask]
        denom = weights.sum()
        if denom == 0:
            cf_scores.append(0.0)
        else:
            cf_scores.append(float((weights * vals).sum() / denom))

    return pd.DataFrame({
        "business_id": list(candidate_biz_ids),
        "cf_score": cf_scores
    })

# ==============================================
# 3. CATEGORY + CF + POPULARITY RECOMMENDER
# ==============================================

def recommend_for_user(user_id, top_n=10, min_review_count=1):
    """
    Recommend restaurants for a user based on:
    1. User's inferred city from TRAIN reviews (user_city_map).
    2. Restaurants in that city the user has not already rated in TRAIN.
    3. Primary sort: collaborative filtering score (cf_score, descending).
    4. Tie-breakers: category similarity (cat_sim, descending),
       then review_count (descending), then stars (descending).
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

    # 5. User-based collaborative filtering score for these candidates
    candidate_biz_ids = city_biz["business_id"].tolist()
    cf_df = compute_cf_scores_for_user(user_id, candidate_biz_ids)
    city_biz = city_biz.merge(cf_df, on="business_id", how="left")
    city_biz["cf_score"] = city_biz["cf_score"].fillna(0.0)

    # 6. Sort: CF score, then category similarity, then review_count, then stars
    city_biz = city_biz.sort_values(
        by=["cf_score", "cat_sim", "review_count", "stars"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    # 7. Take top_n
    result = city_biz[
        ["business_id", "name", "city", "stars", "review_count", "categories", "cat_sim", "cf_score"]
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
