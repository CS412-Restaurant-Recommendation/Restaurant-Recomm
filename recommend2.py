# file: recommend.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

data_dir = Path("Dataset")

# ===============
# 1. LOAD OBJECTS
# ===============

business = pd.read_pickle(data_dir / "business_df.pkl")
reviews = pd.read_pickle(data_dir / "reviews_train.pkl")
user_city_map = pd.read_pickle(data_dir / "user_city_map.pkl").to_dict()

business_features = pd.read_pickle(data_dir / "business_features.pkl")
business_features = business_features.set_index("business_id")

try:
    user_item = pd.read_pickle(data_dir / "user_item_cf.pkl")
    user_index = pd.read_pickle(data_dir / "user_index_cf.pkl").to_dict()
    user_item = user_item.fillna(0.0)
except FileNotFoundError:
    user_item = pd.DataFrame()
    user_index = {}

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
    candidates = candidates.copy()
    if user_profile is None:
        candidates["cat_sim"] = 0.0
        return candidates

    cand_feats = business_features.loc[
        business_features.index.intersection(candidates["business_id"])
    ]
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

    if user_item.empty or user_id not in user_index:
        return pd.DataFrame({
            "business_id": list(candidate_biz_ids),
            "cf_score": np.zeros(len(candidate_biz_ids), dtype=float)
        })

    target_row_idx = user_index[user_id]
    target_vec = user_item.values[target_row_idx:target_row_idx+1]
    target_vec = np.nan_to_num(target_vec, copy=False)
    all_users_mat = np.nan_to_num(user_item.values, copy=False)

    sims = cosine_similarity(target_vec, all_users_mat)[0]

    sim_series = pd.Series(sims, index=user_item.index)
    sim_series = sim_series.drop(labels=[user_id], errors="ignore")
    sim_series = sim_series[sim_series > 0].nlargest(k_neighbors)

    if sim_series.empty:
        return pd.DataFrame({
            "business_id": list(candidate_biz_ids),
            "cf_score": np.zeros(len(candidate_biz_ids), dtype=float)
        })

    neighbor_ratings = user_item.loc[sim_series.index]
    sims_values = sim_series.values.reshape(-1, 1)

    cf_scores = []
    for bid in candidate_biz_ids:
        if bid not in neighbor_ratings.columns:
            cf_scores.append(0.0)
            continue

        col = neighbor_ratings[bid].values.reshape(-1, 1)
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
# 3. CATEGORY + CF + POPULARITY + HYBRID SCORE
# ==============================================

def recommend_for_user(user_id, top_n=10, min_review_count=1):

    user_city = user_city_map.get(user_id)
    if user_city is None:
        print("No inferred city for this user; cannot filter by location.")
        return pd.DataFrame()

    city_biz = business[business["city"] == user_city].copy()
    if city_biz.empty:
        print("No restaurants found in user city:", user_city)
        return pd.DataFrame()

    rated_biz_ids = reviews[reviews["user_id"] == user_id]["business_id"].unique()
    city_biz = city_biz[~city_biz["business_id"].isin(rated_biz_ids)]
    if city_biz.empty:
        print("No unseen restaurants left in user city:", user_city)
        return pd.DataFrame()

    city_biz = city_biz[city_biz["review_count"] >= min_review_count]
    if city_biz.empty:
        print("No restaurants with at least", min_review_count, "reviews in user city:", user_city)
        return pd.DataFrame()

    # Category similarity
    user_profile = build_user_profile(user_id)
    city_biz = add_category_similarity(city_biz, user_profile)

    # CF score
    candidate_biz_ids = city_biz["business_id"].tolist()
    cf_df = compute_cf_scores_for_user(user_id, candidate_biz_ids)
    city_biz = city_biz.merge(cf_df, on="business_id", how="left")
    city_biz["cf_score"] = city_biz["cf_score"].fillna(0.0)

    # =======================
    # NORMALIZATION FOR HYBRID
    # =======================
    city_biz["norm_rc"] = city_biz["review_count"] / city_biz["review_count"].max()
    city_biz["norm_stars"] = city_biz["stars"] / 5.0

    # =======================
    # HYBRID SCORE
    # =======================
    city_biz["hybrid_score"] = (
        0.5 * city_biz["cf_score"] +
        0.3 * city_biz["cat_sim"] +
        0.1 * city_biz["norm_rc"] +
        0.1 * city_biz["norm_stars"]
    )

    # Sort by hybrid score
    city_biz = city_biz.sort_values(
        by=["hybrid_score"],
        ascending=False
    ).reset_index(drop=True)

    # Output
    result = city_biz[
        ["business_id", "name", "city", "stars", "review_count", "categories",
         "cat_sim", "cf_score", "hybrid_score"]
    ].head(top_n)

    return result


# =================
# 4. EXAMPLE USAGE
# =================

if __name__ == "__main__":
    example_user = "j14WgRoU_-2ZE1aw1dXrJg"
    recs = recommend_for_user(example_user, top_n=10, min_review_count=10)
    print(recs)
