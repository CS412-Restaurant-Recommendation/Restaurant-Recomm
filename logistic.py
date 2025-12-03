# logistic_regression_model.py
from common_utils import preprocess_reviews  # sentiment_map not needed now
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Paths to pickled datasets
train_path = "Dataset/reviews_train.pkl"
test_path = "Dataset/reviews_test.pkl"

# Preprocess reviews (labels will be -1, 0, 1)
X_train, X_test, y_train, y_test = preprocess_reviews(train_path, test_path)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train logistic regression
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train_vec, y_train)

# Predict
y_pred = lr_model.predict(X_test_vec)

# Evaluate
target_names = ["negative", "neutral", "positive"]  # corresponds to -1,0,1
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, labels=[-1,0,1], target_names=target_names))


