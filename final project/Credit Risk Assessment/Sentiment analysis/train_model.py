import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Configuration
NEWSAPI_KEY = "6d69be83469b42939e27626f3b453383"

def fetch_training_data():
    """Fetch news articles for training from different categories"""
    categories = ['business', 'technology', 'sports', 'entertainment', 'health']
    all_articles = []
    
    for category in categories:
        url = f"https://newsapi.org/v2/top-headlines?category={category}&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            for article in data.get('articles', []):
                article_info = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'category': category
                }
                all_articles.append(article_info)
    
    return pd.DataFrame(all_articles)

def preprocess_text(text):
    """Preprocess the text data"""
    if not isinstance(text, str):
        return ''
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def get_sentiment_label(text):
    """Get sentiment label using VADER"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    # Define sentiment thresholds
    if scores['compound'] >= 0.05:
        return 1  # Positive
    elif scores['compound'] <= -0.05:
        return 0  # Negative
    else:
        return 0  # Neutral is considered negative for news

def prepare_data(df):
    """Prepare the data for training"""
    # Combine title and description
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    
    # Remove rows where text is empty or only contains whitespace
    df = df[df['text'].str.strip() != '']
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Get sentiment labels using VADER
    df['sentiment'] = df['text'].apply(get_sentiment_label)
    
    return df

def train_model():
    """Train the sentiment analysis model"""
    # Fetch and prepare data
    print("Fetching training data...")
    df = fetch_training_data()
    df = prepare_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['sentiment'],
        test_size=0.2,
        random_state=42
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the model
    print("Training the model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Save the model and vectorizer
    print("\nSaving the model and vectorizer...")
    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    
    print("\nModel training completed!")

if __name__ == "__main__":
    train_model() 