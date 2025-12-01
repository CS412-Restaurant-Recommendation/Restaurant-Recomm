# file: evaluate_recommender.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

data_dir = Path("Dataset")

# ===========
# 1. LOAD OBJECTS
# ===========
business = pd.read_pickle(data_dir / "business_df.pkl")
business_features = pd.read_pickle(data_dir / "business_features.pkl")
bizid_to_idx = pd.read_pickle(data_dir / "bizid_to_idx.pkl").to_dict()
reviews_full = pd.read_pickle(data_dir / "reviews_df.pkl")
user_city_map = pd.read_pickle(data_dir / "user_city_map.pkl").to_dict()

# Category feature matrix
cat_feature_cols = [c for c in business_features.columns if c != "business_id"]
item_cat_matrix = business_features[cat_feature_cols].values  # [n_items, n_cats]


# ===========
# 2. USER PROFILE (same as in unified recommender)
# ===========
def build_user_profile_from_reviews(user_reviews, rating_threshold=4.0):
    """
    Build user category profile from this user's TRAIN reviews only.
    user_reviews: DataFrame with ['user_id', 'business_id', 'stars'].
    """
    liked = user_reviews[user_reviews["stars"] >= rating_threshold]
    if liked.empty:
        return None

    liked_biz_ids = liked["business_id"].unique()
    liked_indices = [bizid_to_idx[b] for b in liked_biz_ids if b in bizid_to_idx]

    if not liked_indices:
        return None

    liked_matrix = item_cat_matrix[liked_indices]
    user_vec = liked_matrix.mean(axis=0)

    # Normalize
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec


# ===========
# 3. RECOMMENDATION LOGIC (mirrors unified recommend_for_user)
# ===========
def recommend_for_user_eval(user_id,
                            train_reviews,
                            top_n=50,
                            rating_threshold=3.0,
                            min_review_count=1,
                            alpha_popularity=1.0):
    """
    Combined model for evaluation:
    - Build user profile from TRAIN reviews.
    - Recommend restaurants in user's city, excluding TRAIN restaurants.
    - Score by category similarity boosted by popularity (review_count, stars).
    Returns a ranked list of business_ids.
    """
    # Get inferred city
    user_city = user_city_map.get(user_id)
    if user_city is None:
        return []

    # Build user profile from TRAIN only
    user_vec = build_user_profile_from_reviews(
        train_reviews,
        rating_threshold=rating_threshold,
    )
    if user_vec is None:
        return []

    # Content scores (similarity to all restaurants)
    content_scores = cosine_similarity(
        user_vec.reshape(1, -1),
        item_cat_matrix
    )[0]  # [n_items]

    # Popularity / quality factor
    biz_stars = business["stars"].values.astype(float)
    biz_reviews = business["review_count"].values.astype(float)

    # Normalize stars to [0,1]
    stars_norm = (biz_stars - 1.0) / (5.0 - 1.0)

    # Normalize log(review_count+1) to [0,1]
    log_counts = np.log1p(biz_reviews)
    log_counts_norm = (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min() + 1e-8)

    # Popularity weight — heavier on review_count
    popularity = 0.3 * stars_norm + 0.7 * log_counts_norm

    # Combined score
    final_scores = content_scores * (1.0 + alpha_popularity * popularity)

    # Exclude TRAIN restaurants (already visited in train)
    train_biz_ids = set(train_reviews["business_id"].unique())
    rated_mask = business["business_id"].isin(train_biz_ids)

    # Restrict to same city
    same_city_mask = business["city"] == user_city

    # Optional: minimum number of reviews
    enough_reviews_mask = business["review_count"] >= min_review_count

    candidate_mask = (~rated_mask) & same_city_mask & enough_reviews_mask
    candidate_indices = np.where(candidate_mask.values)[0]

    if len(candidate_indices) == 0:
        return []

    candidate_scores = final_scores[candidate_indices]

    # Top-N candidates
    top_n = min(top_n, len(candidate_indices))
    top_idx_local = np.argpartition(-candidate_scores, top_n - 1)[:top_n]
    top_idx = candidate_indices[top_idx_local]

    # Sort properly
    top_scores = final_scores[top_idx]
    order = np.argsort(-top_scores)
    top_idx = top_idx[order]

    ranked_biz_ids = business.iloc[top_idx]["business_id"].tolist()
    return ranked_biz_ids


# ===========
# 4. METRICS
# ===========
def precision_recall_hit_at_k(ranked_items, relevant_items, k):
    """
    ranked_items: list of business_ids sorted by score.
    relevant_items: set of business_ids from test (restaurants user visited).
    """
    if len(ranked_items) == 0:
        return 0.0, 0.0, 0.0

    if k > len(ranked_items):
        k = len(ranked_items)

    top_k = ranked_items[:k]
    top_k_set = set(top_k)

    hits = len(top_k_set & relevant_items)
    hit = 1.0 if hits > 0 else 0.0
    precision = hits / float(k) if k > 0 else 0.0
    recall = hits / float(len(relevant_items)) if len(relevant_items) > 0 else 0.0

    return precision, recall, hit


# ===========
# 5. EVALUATION LOOP
# ===========
def evaluate_model(rating_threshold_like=4.0,
                   k=50,
                   min_user_reviews=5,
                   max_users=1000,
                   min_review_count_item=1,
                   alpha_popularity=1.0,
                   random_state=42):
    """
    Evaluate the unified (category + city + popularity) recommender.

    - For each user with >= min_user_reviews:
      * Split their reviews 75% train / 25% test.
      * Build profile from train.
      * Recommend top-K restaurants in their city, unseen in train.
      * Relevant items = all restaurants in test (visited).
    """
    users = reviews_full["user_id"].value_counts()
    eligible_users = users[users >= min_user_reviews].index

    rng = np.random.default_rng(random_state)
    if len(eligible_users) > max_users:
        eligible_users = rng.choice(eligible_users, size=max_users, replace=False)

    precisions, recalls, hits = [], [], []
    n_users_eval = 0

    for user_id in eligible_users:
        user_reviews = reviews_full[reviews_full["user_id"] == user_id]

        # 75/25 split
        train_reviews, test_reviews = train_test_split(
            user_reviews,
            test_size=0.25,
            random_state=random_state,
            shuffle=True,
        )

        if test_reviews.empty:
            continue

        # Relevant items: all restaurants in test set
        relevant_biz_ids = set(test_reviews["business_id"].unique())
        if not relevant_biz_ids:
            continue

        ranked_biz_ids = recommend_for_user_eval(
            user_id=user_id,
            train_reviews=train_reviews,
            top_n=k,
            rating_threshold=rating_threshold_like,
            min_review_count=min_review_count_item,
            alpha_popularity=alpha_popularity,
        )

        if not ranked_biz_ids:
            continue

        prec, rec, hit = precision_recall_hit_at_k(ranked_biz_ids, relevant_biz_ids, k)
        precisions.append(prec)
        recalls.append(rec)
        hits.append(hit)
        n_users_eval += 1

    if n_users_eval == 0:
        print("No users available for evaluation with current settings.")
        return

    print(f"Evaluated users: {n_users_eval}")
    print(f"Hit@{k}:       {np.mean(hits):.4f}")
    print(f"Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Recall@{k}:    {np.mean(recalls):.4f}")


if __name__ == "__main__":
    # Adjust parameters as needed
    evaluate_model(
        rating_threshold_like=3.0,   # how you define "liked" when building the profile
        k=50,                        # size of recommendation list evaluated
        min_user_reviews=5,
        max_users=1000,
        min_review_count_item=10,    # only recommend restaurants with >= 10 reviews
        alpha_popularity=1.0,        # how strongly popularity boosts the content score
        random_state=42,
    )