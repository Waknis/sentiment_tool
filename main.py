import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Text', 'Score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def extract_features(texts, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)
        return vectorizer, vectorizer.fit_transform(texts)
    return vectorizer.transform(texts)


def train_model(X, y):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model


def classify_sentiment(score):
    if score > 3:
        return 'positive'
    elif score < 3:
        return 'negative'
    else:
        return 'neutral'


def main():
    # Load data
    df = load_data('Reviews.csv')

    # Preprocess the text data
    df['processed_text'] = df['Text'].apply(preprocess_text)

    # Classify sentiments
    df['sentiment'] = df['Score'].apply(classify_sentiment)

    # Encode sentiment labels
    le = LabelEncoder()
    df['encoded_sentiment'] = le.fit_transform(df['sentiment'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['encoded_sentiment'], test_size=0.2, random_state=42
    )

    # Extract features
    vectorizer, X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test, vectorizer)

    # Train the model
    model = train_model(X_train_features, y_train)

    # Evaluate the model
    predictions = model.predict(X_test_features)
    print(classification_report(y_test, predictions, target_names=le.classes_))

    # Example of using the model for prediction
    new_reviews = [
        "This product is amazing! I love it.",
        "It's okay, but not great.",
        "Terrible product, don't buy it."
    ]
    new_reviews_processed = [preprocess_text(review) for review in new_reviews]
    new_reviews_features = extract_features(new_reviews_processed, vectorizer)
    new_predictions = model.predict(new_reviews_features)

    print("\nPredictions for new reviews:")
    for review, prediction in zip(new_reviews, new_predictions):
        sentiment = le.inverse_transform([prediction])[0]
        print(f"Review: {review}")
        print(f"Predicted sentiment: {sentiment}\n")


if __name__ == "__main__":
    main()