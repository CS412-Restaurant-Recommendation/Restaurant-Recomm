# file: build_train_test_and_profiles.py

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

data_dir = Path("Dataset")

business_path = data_dir / "filtered_yelp_business.json"
review_path = data_dir / "filtered_yelp_review.json"

# 1. LOAD DATA
business = pd.read_json(business_path, lines=True)
reviews = pd.read_json(review_path, lines=True)

# Keep needed columns from business, including city
business = business[["business_id", "name", "stars", "review_count", "categories", "city"]]
reviews = reviews[["user_id", "business_id", "stars"]]

# Drop broken rows
business = business.dropna(subset=["business_id", "categories", "city"])
reviews = reviews.dropna(subset=["user_id", "business_id", "stars"])

# 1a. KEEP ONLY TOP-K POPULAR RESTAURANTS PER CITY
TOP_K_PER_CITY = 100  # tune this (e.g., 100, 300, 500)

business = (
    business.sort_values(["city", "review_count"], ascending=[True, False])
            .groupby("city")
            .head(TOP_K_PER_CITY)
            .reset_index(drop=True)
)

# 1b. FILTER REVIEWS TO THESE RESTAURANTS ONLY
keep_biz_ids = set(business["business_id"])
reviews = reviews[reviews["business_id"].isin(keep_biz_ids)].reset_index(drop=True)

# 1c. OPTIONAL: DROP USERS WITH VERY FEW REMAINING REVIEWS
MIN_USER_REVIEWS = 2  # or 3/5 if you want denser users
user_counts = reviews["user_id"].value_counts()
keep_users = set(user_counts[user_counts >= MIN_USER_REVIEWS].index)
reviews = reviews[reviews["user_id"].isin(keep_users)].reset_index(drop=True)

print("Businesses after top-K per city:", len(business))
print("Reviews after filtering:", len(reviews))
print("Users after filtering:", reviews["user_id"].nunique())

# 2. PER-USER TRAIN/TEST SPLIT (70/30)
def train_test_split_per_user(df, train_ratio=0.7, min_interactions=5):
    df = df.sort_values(["user_id"]).reset_index(drop=True)
    train_list = []
    test_list = []

    for uid, grp in df.groupby("user_id"):
        if len(grp) < min_interactions:
            # keep very sparse users entirely in train to avoid tiny test sets
            train_list.append(grp)
            continue

        n_train = int(len(grp) * train_ratio)
        if n_train == 0:
            # fallback: everything to train
            train_list.append(grp)
            continue

        if n_train == len(grp):
            train_list.append(grp)
            continue

        train_list.append(grp.iloc[:n_train])
        test_list.append(grp.iloc[n_train:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = (
        pd.concat(test_list).reset_index(drop=True)
        if test_list
        else pd.DataFrame(columns=df.columns)
    )
    return train_df, test_df

train_reviews, test_reviews = train_test_split_per_user(reviews, train_ratio=0.7)
print("Train reviews:", len(train_reviews), "Test reviews:", len(test_reviews))

# 3. PROCESS CATEGORIES (BUSINESS FEATURES – SAME FOR TRAIN/TEST)
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

# 4. SAVE TRAIN-ONLY AND SHARED ARTIFACTS

# Business info is shared for train/test, since items themselves are not “leaking”.
business.to_pickle(data_dir / "business_df.pkl")
business_features.to_pickle(data_dir / "business_features.pkl")
pd.Series(bizid_to_idx).to_pickle(data_dir / "bizid_to_idx.pkl")
pd.Series(cat_feature_names).to_pickle(data_dir / "cat_feature_names.pkl")

# Save reviews split
train_reviews.to_pickle(data_dir / "reviews_train.pkl")
test_reviews.to_pickle(data_dir / "reviews_test.pkl")

print("Saved business_df.pkl, business_features.pkl, and train/test review splits.")
