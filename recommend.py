# file: recommend.py

import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from new_user_recommendation import get_new_user_recommendations, filter_by_categories

# Resolve relative to this file so imports work from any working directory
data_dir = Path(__file__).resolve().parent / "Dataset"

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

@lru_cache(maxsize=256)
def get_neighbors(user_id, k_neighbors=30):
    """
    Return the top-k most similar users to user_id (cosine similarity over the
    precomputed TRAIN user-item matrix), keeping only positive similarities.

    - Returns a Series indexed by neighbor user_id with similarity values,
      or an empty Series if the user is not in the CF matrix.
    - Cached per user, since the similarity against all users is the expensive
      step. Callers must not mutate the returned Series.
    """
    if user_item.empty or user_id not in user_index:
        return pd.Series(dtype=float)

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
    return sim_series[sim_series > 0].nlargest(k_neighbors)


def compute_cf_scores_for_user(user_id, candidate_biz_ids, k_neighbors=30):
    """
    Compute a collaborative-filtering score (cf_score) for each candidate business
    for a given user_id, using user-based CF over the precomputed TRAIN subset.

    - Uses cosine similarity between the target user and all other users.
    - Aggregates neighbors' ratings as a weighted average for each candidate.
    - Returns a DataFrame with columns: business_id, cf_score.
    """
    sim_series = get_neighbors(user_id, k_neighbors)
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

def recommend_for_user(user_id, top_n=10, min_review_count=1, categories=None, exclude_biz_ids=None):
    """
    Recommend restaurants for a user based on:
    1. User's inferred city from TRAIN reviews (user_city_map).
    2. Restaurants in that city the user has not already rated in TRAIN.
    3. Primary sort: collaborative filtering score (cf_score, descending).
    4. Tie-breakers: category similarity (cat_sim, descending),
       then review_count (descending), then stars (descending).
    All personalization is based only on TRAIN data.

    Optional filters:
    - categories: keep only restaurants with at least one of these categories
      (case-insensitive exact match, e.g. ["italian", "pizza"]).
    - exclude_biz_ids: business_ids to leave out (e.g. already shown in chat).
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

    # 3a. Optional category filter and exclusions
    if categories:
        city_biz = filter_by_categories(city_biz, categories)
    if exclude_biz_ids:
        city_biz = city_biz[~city_biz["business_id"].isin(exclude_biz_ids)]
    if city_biz.empty:
        print("No restaurants matching the requested filters in user city:", user_city)
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


def get_recommendations_with_meta(
    user_id: str,
    top_n: int = 10,
    min_review_count: int = 1,
    categories=None,
    exclude_biz_ids=None,
):
    """
    Like get_recommendations, but also reports which path produced the results,
    so callers (e.g., the chat agent) can explain them correctly.

    Returns a dict:
      source: "personalized" or "cold_start"
      fallback_reason: None, "unknown_user" or "no_personalized_results"
      city: the user's inferred home city (None for unknown users)
      recommendations: list of dicts (JSON-serializable)
    """
    # Check if this user has an inferred city; if not, treat as new user
    user_city = user_city_map.get(user_id)
    if user_city is None:
        # cold-start path
        recs = get_new_user_recommendations(
            top_n=top_n, categories=categories, exclude_biz_ids=exclude_biz_ids
        )
        return {"source": "cold_start", "fallback_reason": "unknown_user",
                "city": None, "recommendations": recs}

    # existing-user path
    df = recommend_for_user(
        user_id=user_id,
        top_n=top_n,
        min_review_count=min_review_count,
        categories=categories,
        exclude_biz_ids=exclude_biz_ids,
    )

    # If nothing came back (e.g., filtered out), also fall back to cold-start
    if df.empty:
        recs = get_new_user_recommendations(
            top_n=top_n, categories=categories, exclude_biz_ids=exclude_biz_ids
        )
        return {"source": "cold_start", "fallback_reason": "no_personalized_results",
                "city": user_city, "recommendations": recs}

    return {"source": "personalized", "fallback_reason": None,
            "city": user_city, "recommendations": df.to_dict(orient="records")}


def get_recommendations(user_id: str, top_n: int = 10, min_review_count: int = 1):
    """
    Wrapper function for external callers (e.g., FastAPI).
    Returns a list of dicts (JSON-serializable).
    For unknown users (no city mapping), falls back to global new-user recommendations.
    """
    return get_recommendations_with_meta(
        user_id=user_id, top_n=top_n, min_review_count=min_review_count
    )["recommendations"]

# ==============================================
# 4. EXPLANATIONS
# ==============================================

def explain_recommendation(user_id, business_id, k_neighbors=30, max_examples=3):
    """
    Explain why business_id was (or would be) recommended to user_id, using the
    same signals the ranker uses.

    Returns a dict with:
      - restaurant: basic info about the business
      - collaborative: how the user's top-k similar users rated it
      - content: categories shared with the user's taste profile, plus
        restaurants the user rated 4+ stars that share those categories
    Returns {"error": ...} if business_id is unknown.
    """
    biz_rows = business[business["business_id"] == business_id]
    if biz_rows.empty:
        return {"error": f"Unknown business_id: {business_id}"}
    biz = biz_rows.iloc[0]

    result = {
        "user_id": user_id,
        "user_known": user_id in user_city_map,
        "restaurant": {
            "business_id": business_id,
            "name": biz["name"],
            "city": biz["city"],
            "stars": float(biz["stars"]),
            "review_count": int(biz["review_count"]),
            "categories": biz["categories"],
        },
    }

    # Collaborative evidence. user_item has unrated cells filled with 0,
    # so only ratings > 0 count as "rated".
    neighbors = get_neighbors(user_id, k_neighbors)
    collaborative = {
        "similar_users_considered": int(len(neighbors)),
        "similar_users_who_rated": 0,
        "avg_rating_from_similar_users": None,
        "avg_similarity_of_raters": None,
        "cf_score": float(compute_cf_scores_for_user(user_id, [business_id], k_neighbors)["cf_score"].iloc[0]),
    }
    if not neighbors.empty and business_id in user_item.columns:
        ratings = user_item.loc[neighbors.index, business_id]
        rated = ratings[ratings > 0]
        if not rated.empty:
            weights = neighbors.loc[rated.index]
            collaborative["similar_users_who_rated"] = int(len(rated))
            collaborative["avg_rating_from_similar_users"] = round(
                float((rated * weights).sum() / weights.sum()), 2
            )
            collaborative["avg_similarity_of_raters"] = round(float(weights.mean()), 3)
    result["collaborative"] = collaborative

    # Content evidence: overlap between the business's categories and the user's profile
    content = {
        "category_similarity": 0.0,
        "shared_categories": [],
        "liked_restaurants_with_shared_categories": [],
    }
    user_profile = build_user_profile(user_id)
    if user_profile is not None and business_id in business_features.index:
        biz_vec = business_features.loc[business_id].values.astype(float)
        norm = np.linalg.norm(biz_vec)
        if norm > 0:
            content["category_similarity"] = round(float((biz_vec / norm) @ user_profile), 3)

        shared_mask = (biz_vec > 0) & (user_profile > 0)
        shared = pd.Series(user_profile[shared_mask], index=business_features.columns[shared_mask])
        shared_cats = shared.sort_values(ascending=False).index.tolist()
        content["shared_categories"] = shared_cats

        if shared_cats:
            liked = reviews[(reviews["user_id"] == user_id) & (reviews["stars"] >= 4)]
            liked = liked.merge(
                business[["business_id", "name", "city", "categories", "category_list"]],
                on="business_id",
            )
            liked = liked[liked["business_id"] != business_id]
            liked = filter_by_categories(liked, shared_cats)
            liked = liked.sort_values("stars", ascending=False).head(max_examples)
            content["liked_restaurants_with_shared_categories"] = [
                {"name": row["name"], "city": row["city"], "your_rating": int(row["stars"]),
                 "categories": row["categories"]}
                for _, row in liked.iterrows()
            ]
    result["content"] = content

    return result

# ==============================================
# 5. CONTENT-ONLY HELPERS (cold start, similar items, taste)
# ==============================================

def build_profile_from_categories(categories):
    """
    Build a normalized category preference vector from a list of category names
    (e.g. stated preferences from a new user). Same shape as build_user_profile.
    Returns None if none of the categories are known.
    """
    wanted = {c.strip().lower() for c in categories if c and c.strip()}
    profile = business_features.columns.isin(wanted).astype(float)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return None
    return profile / norm


def recommend_from_profile(profile, city=None, top_n=10, categories=None,
                           min_review_count=1, exclude_biz_ids=None):
    """
    Content-only recommendations for a category profile vector (e.g. from
    build_profile_from_categories). Optionally restricted to a city.
    Sort: category similarity, then review_count, then stars.
    """
    candidates = business
    if city:
        candidates = candidates[candidates["city"].str.lower() == city.strip().lower()]
    candidates = candidates[candidates["review_count"] >= min_review_count]
    if categories:
        candidates = filter_by_categories(candidates, categories)
    if exclude_biz_ids:
        candidates = candidates[~candidates["business_id"].isin(exclude_biz_ids)]
    if candidates.empty:
        return pd.DataFrame()

    candidates = add_category_similarity(candidates, profile)
    candidates = candidates.sort_values(
        by=["cat_sim", "review_count", "stars"],
        ascending=[False, False, False],
    )
    return candidates[
        ["business_id", "name", "city", "stars", "review_count", "categories", "cat_sim"]
    ].head(top_n).reset_index(drop=True)


def similar_restaurants(business_id, top_n=5, same_city=True):
    """
    Restaurants most similar to business_id by category vector (cosine).
    Sort: category similarity, then stars, then review_count.
    Returns an empty DataFrame if business_id is unknown.
    """
    if business_id not in business_features.index:
        return pd.DataFrame()

    target_vec = business_features.loc[business_id].values.astype(float)
    norm = np.linalg.norm(target_vec)
    if norm == 0:
        return pd.DataFrame()

    candidates = business[business["business_id"] != business_id]
    if same_city:
        target_city = business.loc[business["business_id"] == business_id, "city"].iloc[0]
        candidates = candidates[candidates["city"] == target_city]
    if candidates.empty:
        return pd.DataFrame()

    candidates = add_category_similarity(candidates, target_vec / norm)
    # Places sharing no categories are not "similar", even if few remain
    candidates = candidates[candidates["cat_sim"] > 0]
    candidates = candidates.sort_values(
        by=["cat_sim", "stars", "review_count"],
        ascending=[False, False, False],
    )
    return candidates[
        ["business_id", "name", "city", "stars", "review_count", "categories", "cat_sim"]
    ].head(top_n).reset_index(drop=True)


def get_user_taste(user_id, top_categories=10, top_restaurants=10):
    """
    Summarize a user's taste from TRAIN reviews: home city, review stats,
    strongest categories in their profile, and their highest-rated restaurants.
    Returns None if the user has no TRAIN reviews.
    """
    user_reviews = reviews[reviews["user_id"] == user_id]
    if user_reviews.empty:
        return None

    taste = {
        "user_id": user_id,
        "home_city": user_city_map.get(user_id),
        "review_count": int(len(user_reviews)),
        "avg_stars_given": round(float(user_reviews["stars"].mean()), 2),
        "top_categories": [],
        "top_rated_restaurants": [],
    }

    user_profile = build_user_profile(user_id)
    if user_profile is not None:
        weights = pd.Series(user_profile, index=business_features.columns)
        weights = weights[weights > 0].sort_values(ascending=False).head(top_categories)
        taste["top_categories"] = [
            {"category": cat, "weight": round(float(w), 3)} for cat, w in weights.items()
        ]

    history = user_reviews.merge(
        business[["business_id", "name", "city", "categories", "review_count"]],
        on="business_id",
    )
    history = history.sort_values(["stars", "review_count"], ascending=[False, False]).head(top_restaurants)
    taste["top_rated_restaurants"] = [
        {"business_id": row["business_id"], "name": row["name"], "city": row["city"],
         "your_rating": int(row["stars"]), "categories": row["categories"]}
        for _, row in history.iterrows()
    ]
    return taste


@lru_cache(maxsize=1)
def _users_by_review_count():
    """TRAIN review counts for users who have both a CF row and a home city."""
    counts = reviews["user_id"].value_counts()
    eligible = [uid for uid in counts.index if uid in user_index and uid in user_city_map]
    return counts.loc[eligible]


def get_sample_users(n=20):
    """
    Return n known users with the richest history (most TRAIN reviews), for
    picking a user ID in a demo UI.
    """
    counts = _users_by_review_count().head(n)
    return [
        {"user_id": uid, "home_city": user_city_map.get(uid), "review_count": int(cnt)}
        for uid, cnt in counts.items()
    ]

