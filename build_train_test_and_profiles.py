# file: build_train_test_and_profiles.py

import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

data_dir = Path("Dataset")
business_path = data_dir / "filtered_yelp_business(new).json"
review_path = data_dir / "filtered_yelp_review.json"

# -------------------------
# 1. MEMORY-SAFE LOAD BUSINESS DATA
# -------------------------
print("Loading business data...")
filtered_business_records = []

with open(business_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        # Keep only records with business_id and categories
        if record.get("business_id") and record.get("categories"):
            filtered_business_records.append({
                "business_id": record.get("business_id"),
                "name": record.get("name"),
                "stars": record.get("stars"),
                "review_count": record.get("review_count"),
                "categories": record.get("categories"),
                "city": record.get("city") if "city" in record else None  # include city
            })

business = pd.DataFrame(filtered_business_records)
print(f"Total businesses loaded: {len(business)}")

# -------------------------
# 1a. KEEP ONLY TOP-K BUSINESSES BY REVIEW COUNT
# -------------------------
TOP_K = 1000  # You can adjust this
business = business.sort_values("review_count", ascending=False).head(TOP_K).reset_index(drop=True)
print(f"Businesses after top-{TOP_K} filtering: {len(business)}")

# -------------------------
# 1b. MEMORY-SAFE LOAD REVIEWS
# -------------------------
reviews_records = []
print("Loading review data...")
with open(review_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        if record.get("user_id") and record.get("business_id") and record.get("stars") is not None:
            reviews_records.append({
                "user_id": record.get("user_id"),
                "business_id": record.get("business_id"),
                "stars": record.get("stars"),
                "text": record.get("text")  # <-- added review text
            })

reviews = pd.DataFrame(reviews_records)
print(f"Total reviews loaded: {len(reviews)}")

# Remove reviews with empty text
reviews = reviews[reviews['text'].str.strip().astype(bool)].reset_index(drop=True)

# -------------------------
# Filter reviews to the selected businesses
# -------------------------
keep_biz_ids = set(business["business_id"])
reviews = reviews[reviews["business_id"].isin(keep_biz_ids)].reset_index(drop=True)
print(f"Reviews after filtering by business: {len(reviews)}")

# -------------------------
# 1c. DROP USERS WITH VERY FEW REVIEWS
# -------------------------
MIN_USER_REVIEWS = 5
user_counts = reviews["user_id"].value_counts()
keep_users = set(user_counts[user_counts >= MIN_USER_REVIEWS].index)
reviews = reviews[reviews["user_id"].isin(keep_users)].reset_index(drop=True)
print(f"Reviews after filtering by user (min {MIN_USER_REVIEWS}): {len(reviews)}")
print(f"Unique users: {reviews['user_id'].nunique()}")

if reviews.empty:
    raise ValueError("No reviews left after filtering. Check your input files!")

# -------------------------
# 2. TRAIN/TEST SPLIT PER USER
# -------------------------
def train_test_split_per_user(df, train_ratio=0.7, min_interactions=5):
    df = df.sort_values(["user_id"]).reset_index(drop=True)
    train_list, test_list = [], []

    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_interactions:
            train_list.append(grp)
            continue

        n_train = int(len(grp) * train_ratio)
        train_list.append(grp.iloc[:n_train])
        if n_train < len(grp):
            test_list.append(grp.iloc[n_train:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True) if test_list else pd.DataFrame(columns=df.columns)
    return train_df, test_df

train_reviews, test_reviews = train_test_split_per_user(reviews, train_ratio=0.7)
print(f"Train reviews: {len(train_reviews)}, Test reviews: {len(test_reviews)}")

# -------------------------
# 3. PROCESS CATEGORIES
# -------------------------
def split_categories(cat_str):
    if not isinstance(cat_str, str):
        return []
    return [c.strip().lower() for c in cat_str.split(",")]

business["category_list"] = business["categories"].apply(split_categories)
mlb = MultiLabelBinarizer()
cat_matrix = mlb.fit_transform(business["category_list"])
cat_feature_names = mlb.classes_

cat_df = pd.DataFrame(cat_matrix, columns=cat_feature_names, index=business.index)
business_features = pd.concat([business[["business_id"]], cat_df], axis=1)

bizid_to_idx = {bid: i for i, bid in enumerate(business_features["business_id"])}

# -------------------------
# 4. SAVE ARTIFACTS
# -------------------------
business.to_pickle(data_dir / "business_df.pkl")
business_features.to_pickle(data_dir / "business_features.pkl")
pd.Series(bizid_to_idx).to_pickle(data_dir / "bizid_to_idx.pkl")
pd.Series(cat_feature_names).to_pickle(data_dir / "cat_feature_names.pkl")
train_reviews.to_pickle(data_dir / "reviews_train.pkl")
test_reviews.to_pickle(data_dir / "reviews_test.pkl")

print("Saved all artifacts successfully.")




