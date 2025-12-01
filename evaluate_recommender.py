# file: evaluate_recommender.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path("Dataset")

# ===========
# 1. LOAD OBJECTS
# ===========
business = pd.read_pickle(data_dir / "business_df.pkl")
reviews_full = pd.read_pickle(data_dir / "reviews_df.pkl")
user_city_map = pd.read_pickle(data_dir / "user_city_map.pkl").to_dict()

# Ensure necessary columns exist
assert "business_id" in business.columns
assert "city" in business.columns
assert {"user_id", "business_id", "stars"}.issubset(reviews_full.columns)


# ===========
# 2. POPULARITY-BASED RECOMMENDER (same idea as in recommend.py)
# ===========
def recommend_popular_in_city_for_eval(user_id, train_reviews, top_n=50, min_review_count=1):
    """
    Popularity-based recommendations for evaluation.

    - user_id: current user
    - train_reviews: this user's TRAIN reviews (used to exclude already-known restaurants)
    - top_n: how many items to recommend
    - min_review_count: minimum number of reviews for a restaurant to be a candidate
    """
    user_city = user_city_map.get(user_id)
    if user_city is None:
        return []

    # Restaurants in this city
    same_city_mask = business["city"] == user_city
    city_biz = business[same_city_mask].copy()
    if city_biz.empty:
        return []

    # Exclude restaurants seen in TRAIN
    seen_train_biz_ids = train_reviews["business_id"].unique()
    city_biz = city_biz[~city_biz["business_id"].isin(seen_train_biz_ids)]
    if city_biz.empty:
        return []

    # Filter by min_review_count (popularity filter)
    city_biz = city_biz[city_biz["review_count"] >= min_review_count]
    if city_biz.empty:
        return []

    # Sort by popularity (review_count), then rating as tie-breaker
    city_biz = city_biz.sort_values(
        by=["review_count", "stars"],
        ascending=[False, False],
    )

    # Return top_n business_ids as recommendation list
    rec_ids = city_biz["business_id"].head(top_n).tolist()
    return rec_ids


# ===========
# 3. METRICS
# ===========
def precision_recall_hit_at_k(ranked_items, relevant_items, k):
    """
    ranked_items: list of business_ids, sorted by rec quality.
    relevant_items: set of business_ids from test (visited by user).
    """
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
# 4. EVALUATION LOOP
# ===========
def evaluate_popularity_model(k=50, min_user_reviews=5, max_users=1000,
                              min_review_count_item=1, random_state=42):
    """
    Evaluate popularity-based recommender:
    - Train/test split: 75% train, 25% test per user.
    - Recommendations: most popular unseen restaurants in user's city.
    - Relevant items: all restaurants in test (restaurants the user visited/rated).
    """

    users = reviews_full["user_id"].value_counts()
    eligible_users = users[users >= min_user_reviews].index

    # Sample subset of users for speed
    rng = np.random.default_rng(random_state)
    if len(eligible_users) > max_users:
        eligible_users = rng.choice(eligible_users, size=max_users, replace=False)

    precisions, recalls, hits = [], [], []
    n_users_eval = 0

    for user_id in eligible_users:
        user_reviews = reviews_full[reviews_full["user_id"] == user_id]

        # 75/25 split for this user's interactions
        train_reviews, test_reviews = train_test_split(
            user_reviews,
            test_size=0.25,
            random_state=random_state,
            shuffle=True,
        )

        if test_reviews.empty:
            continue

        # Relevant items: all restaurants in test (user visited/rated)
        relevant_biz_ids = set(test_reviews["business_id"].unique())
        if not relevant_biz_ids:
            continue

        # Get popularity-based recommendations for this user
        rec_ids = recommend_popular_in_city_for_eval(
            user_id=user_id,
            train_reviews=train_reviews,
            top_n=k,  # we’ll evaluate at K using this list
            min_review_count=min_review_count_item,
        )

        if not rec_ids:
            continue

        prec, rec, hit = precision_recall_hit_at_k(rec_ids, relevant_biz_ids, k)
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
    # Example: evaluate top-50 popular restaurants per user
    evaluate_popularity_model(
        k=50,
        min_user_reviews=5,
        max_users=1000,
        min_review_count_item=1,  # or 5 / 10 to only consider more popular places
        random_state=42,
    )
