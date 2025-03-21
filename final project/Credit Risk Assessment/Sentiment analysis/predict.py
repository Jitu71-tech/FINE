import joblib
import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuration
NEWSAPI_KEY = "6d69be83469b42939e27626f3b453383"

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

def fetch_news(query):
    """Fetch news articles for the given query"""
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        articles = []
        for article in data.get('articles', []):
            article_info = {
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', '')
            }
            articles.append(article_info)
        return pd.DataFrame(articles)
    else:
        print(f"Error fetching news: {response.status_code}")
        return pd.DataFrame()

def analyze_sentiment(query):
    """Analyze sentiment of news articles for the given query"""
    # Load the trained model and vectorizer
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    # Fetch news articles
    print(f"Fetching news articles for query: {query}")
    df = fetch_news(query)
    
    if df.empty:
        print("No articles found.")
        return
    
    # Prepare the data
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    
    # Remove rows where text is empty or only contains whitespace
    df = df[df['text'].str.strip() != '']
    
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Transform the text data
    X = vectorizer.transform(df['processed_text'])
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Add predictions to the dataframe
    df['sentiment'] = predictions
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Good News' if x == 1 else 'Bad News')
    df['confidence'] = probabilities.max(axis=1)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    df['sentiment_label'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title(f'Sentiment Analysis of News Articles for "{query}"')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png')
    plt.close()
    
    # Print results
    print("\nSentiment Analysis Results:")
    print(f"Total articles analyzed: {len(df)}")
    print(f"Good News: {sum(df['sentiment'] == 1)}")
    print(f"Bad News: {sum(df['sentiment'] == 0)}")
    
    # Print detailed results
    print("\nDetailed Analysis:")
    for _, row in df.iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Sentiment: {row['sentiment_label']}")
        print(f"Confidence: {row['confidence']:.2%}")
        print("-" * 80)
    
    # Save detailed results to CSV
    df.to_csv('sentiment_results.csv', index=False)
    print("\nDetailed results saved to 'sentiment_results.csv'")

if __name__ == "__main__":
    query = input("Enter your search query: ")
    analyze_sentiment(query) 