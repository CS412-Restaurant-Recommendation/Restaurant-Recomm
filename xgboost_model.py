# xgboost_model.py
import pandas as pd
from common_utils import preprocess_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb  # type: ignore

# Paths to pickled datasets
train_path = "Dataset/reviews_train.pkl"
test_path = "Dataset/reviews_test.pkl"

# Preprocess reviews (labels will be -1, 0, 1)
X_train, X_test, y_train, y_test = preprocess_reviews(train_path, test_path)

# Map -1,0,1 → 0,1,2 for XGBoost (required, XGBoost does not accept negative labels)
y_train_xgb = y_train.map({-1:0, 0:1, 1:2})
y_test_xgb  = y_test.map({-1:0, 0:1, 1:2})

# Vectorize text
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,  # 3 classes: negative, neutral, positive
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train_vec, y_train_xgb)

# Predict
y_pred_xgb = xgb_model.predict(X_test_vec)

# Map predictions back to original -1,0,1 labels for consistency
inverse_map = {0:-1, 1:0, 2:1}
y_pred = pd.Series(y_pred_xgb).map(inverse_map)

# Evaluate
target_names = ["negative", "neutral", "positive"]
print("XGBoost Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, labels=[-1,0,1], target_names=target_names))


