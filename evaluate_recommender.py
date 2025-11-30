# file: evaluate_recommender.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

data_dir = Path("Dataset")

# ===========
# 1. LOAD PRECOMPUTED OBJECTS
# ===========
business = pd.read_pickle(data_dir / "business_df.pkl")
business_features = pd.read_pickle(data_dir / "business_features.pkl")
bizid_to_idx = pd.read_pickle(data_dir / "bizid_to_idx.pkl").to_dict()
reviews_full = pd.read_pickle(data_dir / "reviews_df.pkl")

# Extract category feature matrix (exclude business_id column)
cat_feature_cols = [c for c in business_features.columns if c != "business_id"]
item_cat_matrix = business_features[cat_feature_cols].values  # [n_items, n_cats]


# ===========
# 2. USER PROFILE
# ===========
def build_user_profile_from_reviews(user_reviews, rating_threshold=4.0):
    """
    Build user profile from a DataFrame of this user's TRAIN reviews only.
    user_reviews: DataFrame with columns ['user_id', 'business_id', 'stars'].
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

    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec


def score_items_for_user_vec(user_vec):
    """Cosine similarity between user_vec and all restaurants."""
    return cosine_similarity(user_vec.reshape(1, -1), item_cat_matrix)[0]


# ===========
# 3. RANKING METRICS
# ===========
def precision_recall_hit_at_k(ranked_items, relevant_items, k):
    """
    ranked_items: list/array of business_ids sorted by score (best first).
    relevant_items: set of business_ids the user liked in TEST.
    """
    top_k = ranked_items[:k]
    top_k_set = set(top_k)

    hits = len(top_k_set & relevant_items)
    hit = 1.0 if hits > 0 else 0.0

    precision = hits / float(k) if k > 0 else 0.0
    recall = hits / float(len(relevant_items)) if len(relevant_items) > 0 else 0.0

    return precision, recall, hit


# ===========
# 4. OFFLINE EVALUATION
# ===========
def evaluate_model(rating_threshold_like=4.0, k=10, min_user_reviews=5, max_users=500, random_state=42):
    users = reviews_full["user_id"].value_counts()
    eligible_users = users[users >= min_user_reviews].index

    rng = np.random.default_rng(random_state)
    if len(eligible_users) > max_users:
        eligible_users = rng.choice(eligible_users, size=max_users, replace=False)


    precisions, recalls, hits = [], [], []
    n_users_eval = 0

    for user_id in eligible_users:
        user_reviews = reviews_full[reviews_full["user_id"] == user_id]

        # 75/25 split on this user's interactions
        train_reviews, test_reviews = train_test_split(
            user_reviews,
            test_size=0.25,
            random_state=42,
            shuffle=True,
        )

        # Build user profile from TRAIN only
        user_vec = build_user_profile_from_reviews(
            train_reviews,
            rating_threshold=rating_threshold_like,
        )
        if user_vec is None:
            continue  # skip users with no liked items in train

        # Identify relevant items in TEST (liked restaurants)
        test_liked = test_reviews[test_reviews["stars"] >= rating_threshold_like]
        relevant_biz_ids = set(test_liked["business_id"].unique())
        if not relevant_biz_ids:
            continue  # nothing to evaluate for this user

        # Score all restaurants
        scores = score_items_for_user_vec(user_vec)

        # Exclude items seen in TRAIN (we want to recommend unseen ones)
        seen_train_biz = set(train_reviews["business_id"].unique())
        seen_mask = business["business_id"].isin(seen_train_biz)
        candidate_indices = np.where(~seen_mask.values)[0]

        if len(candidate_indices) == 0:
            continue

        candidate_scores = scores[candidate_indices]
        # Sort candidates by score descending
        sorted_idx_local = np.argsort(-candidate_scores)
        ranked_indices = candidate_indices[sorted_idx_local]

        ranked_biz_ids = business.iloc[ranked_indices]["business_id"].tolist()

        # Compute metrics for this user
        prec, rec, hit = precision_recall_hit_at_k(
            ranked_biz_ids, relevant_biz_ids, k
        )

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
    # You can tweak these:
    evaluate_model(
        rating_threshold_like=4.0,  # what counts as "liked"
        k=10,                       # top-K cutoff
        min_user_reviews=5,         # only evaluate users with at least 5 reviews
    )