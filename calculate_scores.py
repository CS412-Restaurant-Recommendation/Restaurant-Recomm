# calculate_scores.py
import pandas as pd
from common_utils import preprocess_reviews
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_sentiment_and_hybrid_scores(train_path, test_path, alpha=0.5, beta=0.5):
    """
    Train Logistic Regression on training data, predict sentiment on test data,
    and calculate sentiment score and hybrid score.
    
    Args:
        train_path (str): path to training pickle file
        test_path (str): path to test pickle file
        alpha (float): weight for normalized star rating
        beta (float): weight for normalized sentiment score
        
    Returns:
        pd.DataFrame: test dataframe with sentiment_score and hybrid_score
    """
    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_reviews(train_path, test_path)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=500)
    lr_model.fit(X_train_vec, y_train)
    
    # Predict sentiment labels (-1,0,1)
    y_pred = lr_model.predict(X_test_vec)
    
    # Load test dataframe
    df_test = pd.read_pickle(test_path)
    df_test['pred_label'] = y_pred
    
    # Calculate normalized sentiment score (0 to 1)
    df_test['sentiment_score'] = (df_test['pred_label'] + 1) / 2
    
    # Calculate normalized star rating (0 to 1)
    df_test['star_score'] = (df_test['stars'] - 1) / 4
    
    # Calculate hybrid score
    df_test['hybrid_score'] = alpha * df_test['star_score'] + beta * df_test['sentiment_score']
    
    return df_test[['text', 'stars', 'pred_label', 'sentiment_score', 'star_score', 'hybrid_score']]

# Example usage
if __name__ == "__main__":
    train_path = "Dataset/reviews_train.pkl"
    test_path = "Dataset/reviews_test.pkl"
    
    df_scores = calculate_sentiment_and_hybrid_scores(train_path, test_path)
    print(df_scores.head())
