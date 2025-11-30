# file: recommender.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

data_dir = Path("Dataset")

# ===========
# 1. LOAD PRECOMPUTED OBJECTS
# ===========
business = pd.read_pickle(data_dir / "business_df.pkl")
business_features = pd.read_pickle(data_dir / "business_features.pkl")
bizid_to_idx = pd.read_pickle(data_dir / "bizid_to_idx.pkl").to_dict()
reviews = pd.read_pickle(data_dir / "reviews_df.pkl")

# Extract category feature matrix (exclude business_id column)
cat_feature_cols = [c for c in business_features.columns if c != "business_id"]
item_cat_matrix = business_features[cat_feature_cols].values  # shape: [n_items, n_cats]

# ===========
# 2. BUILD USER PROFILE (CATEGORY VECTOR)
# ===========
def build_user_profile(user_id, rating_threshold=4.0):
    """
    Aggregate categories of businesses the user rated >= threshold,
    returning a normalized preference vector over categories.
    """
    user_reviews = reviews[(reviews["user_id"] == user_id) & (reviews["stars"] >= rating_threshold)]
    if user_reviews.empty:
        return None  # cold-start user

    # Collect indices of businesses this user liked
    liked_biz_ids = user_reviews["business_id"].unique()
    liked_indices = [bizid_to_idx[b] for b in liked_biz_ids if b in bizid_to_idx]

    if not liked_indices:
        return None

    # Average the category vectors of liked businesses
    liked_matrix = item_cat_matrix[liked_indices]
    user_vec = liked_matrix.mean(axis=0)

    # Normalize
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec

# ===========
# 3. SCORING FUNCTION
#    content score × popularity factor
# ===========
def score_items_for_user(user_id, rating_threshold=4.0, min_review_count=5):
    user_vec = build_user_profile(user_id, rating_threshold=rating_threshold)
    if user_vec is None:
        return None  # handle cold-start separately

    # Cosine similarity between user profile and all restaurants
    content_scores = cosine_similarity(user_vec.reshape(1, -1), item_cat_matrix)[0]  # shape: [n_items]

    # Popularity / quality factor from business stars and review_count
    biz_stars = business["stars"].values  # global average ratings
    biz_reviews = business["review_count"].values.astype(float)

    # Normalize stars to [0,1] (assuming 1–5)
    stars_norm = (biz_stars - 1.0) / (5.0 - 1.0)

    # Normalize log(review_count+1) to [0,1]
    log_counts = np.log1p(biz_reviews)
    log_counts_norm = (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min() + 1e-8)

    # Popularity weight
    popularity = 0.7 * stars_norm + 0.3 * log_counts_norm

    # Final score: content * (1 + alpha * popularity)
    alpha = 1.0
    final_scores = content_scores * (1.0 + alpha * popularity)

    return final_scores

# ===========
# 4. TOP-N RECOMMEND FUNCTION
# ===========
def recommend_for_user(user_id, top_n=10, rating_threshold=4.0):
    scores = score_items_for_user(user_id, rating_threshold=rating_threshold)
    if scores is None:
        print("Cold-start user: no high-rated history.")
        return pd.DataFrame()

    # Exclude businesses the user has already rated
    rated_biz_ids = reviews[reviews["user_id"] == user_id]["business_id"].unique()
    rated_mask = business["business_id"].isin(rated_biz_ids)

    candidate_indices = np.where(~rated_mask.values)[0]
    candidate_scores = scores[candidate_indices]

    # Top-N indices among candidates
    if len(candidate_indices) == 0:
        print("No candidate restaurants left.")
        return pd.DataFrame()

    top_n = min(top_n, len(candidate_indices))
    top_idx_local = np.argpartition(-candidate_scores, top_n - 1)[:top_n]
    top_idx = candidate_indices[top_idx_local]

    # Build result DataFrame
    result = business.iloc[top_idx][["business_id", "name", "stars", "review_count", "categories"]].copy()
    result["score"] = scores[top_idx]

    # Sort by score descending
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result

# ===========
# 5. EXAMPLE USAGE
# ===========
if __name__ == "__main__":
    example_user = "j14WgRoU_-2ZE1aw1dXrJg"  # replace with a real user_id from your data
    recs = recommend_for_user(example_user, top_n=10, rating_threshold=4.0)
    print(recs)
