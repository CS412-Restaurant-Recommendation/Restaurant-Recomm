# file: eval_recommender.py

import pandas as pd
from pathlib import Path
from recommend import recommend_for_user  # uses TRAIN-only data

data_dir = Path("Dataset")

# Load held-out TEST reviews (30%)
test_reviews = pd.read_pickle(data_dir / "reviews_test.pkl")

TOP_N = 50
MIN_REVIEW_COUNT = 10  # keep consistent with recommend_for_user calls

def evaluate_hit_at_n(top_n=TOP_N, min_review_count=MIN_REVIEW_COUNT, min_test_items=1):
    """
    For each user with at least `min_test_items` interactions in the TEST set:
      - Get their ground-truth test businesses
      - Get top-N recommendations from recommend_for_user (TRAIN-based)
      - Check if there is at least one overlap (hit) between recs and test items

    Returns:
      hit_at_n (float): fraction of users with at least one hit in top-N
      total_users (int): number of users evaluated
    """

    # Users who appear in the TEST set
    users = test_reviews["user_id"].unique()
    
    hits = 0
    total_users = 0

    users = users[:500]

    for uid in users:
        user_test = test_reviews[test_reviews["user_id"] == uid]
        if len(user_test) < min_test_items:
            continue

        test_items = set(user_test["business_id"])
        if not test_items:
            continue

        # Get recommendations using TRAIN-only recommender
        recs = recommend_for_user(uid, top_n=top_n, min_review_count=min_review_count)
        if recs is None or recs.empty:
            continue

        rec_items = set(recs["business_id"])

        # Hit if any test item appears in top-N recs
        if test_items & rec_items:
            hits += 1

        total_users += 1

    hit_at_n = hits / total_users if total_users > 0 else 0.0
    return hit_at_n, total_users

if __name__ == "__main__":
    hit, n_users = evaluate_hit_at_n()
    print(f"Evaluated users: {n_users}")
    print(f"Hit@{TOP_N}: {hit:.4f}")
