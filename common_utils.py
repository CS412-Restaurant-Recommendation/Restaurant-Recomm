# common_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def map_star_to_sentiment(star):
    if star >= 4:
        return "positive"
    elif star == 3:
        return "neutral"
    else:  # 1 or 2
        return "negative"

# Map sentiments to numeric labels (-1, 0, 1)
sentiment_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

def preprocess_reviews(train_path, test_path):
    train_reviews = pd.read_pickle(train_path)
    test_reviews = pd.read_pickle(test_path)

    # Map stars to sentiment labels
    train_reviews['sentiment'] = train_reviews['stars'].apply(map_star_to_sentiment)
    test_reviews['sentiment'] = test_reviews['stars'].apply(map_star_to_sentiment)

    # Map sentiment labels to numeric values
    train_reviews['label'] = train_reviews['sentiment'].map(sentiment_map)
    test_reviews['label'] = test_reviews['sentiment'].map(sentiment_map)

    X_train = train_reviews['text']
    y_train = train_reviews['label']
    X_test = test_reviews['text']
    y_test = test_reviews['label']

    return X_train, X_test, y_train, y_test
