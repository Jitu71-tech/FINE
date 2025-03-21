import requests
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from data_cleaning import clean_news_data, analyze_text_statistics

def fetch_news(query):
    NEWSAPI_KEY = "6d69be83469b42939e27626f3b453383"
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data['articles'])
    else:
        print(f"Error fetching news: {response.status_code}")
        return pd.DataFrame()

def analyze_sentiment(query):
    """Analyze sentiment of news articles"""
    # Load model and vectorizer
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    # Fetch and clean news data
    df = fetch_news(query)
    if df.empty:
        print("No articles found for the given query.")
        return
    
    # Clean the data
    df_cleaned = clean_news_data(df)
    
    X = vectorizer.transform(df_cleaned['text_cleaned'])
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    df_cleaned['sentiment'] = predictions
    df_cleaned['sentiment_label'] = df_cleaned['sentiment'].map({1: 'Good News', 0: 'Bad News'})
    df_cleaned['confidence'] = probabilities.max(axis=1)
    
    stats = analyze_text_statistics(df_cleaned, 'text_cleaned')
    print("\nAnalysis Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Print detailed results
    print("\nDetailed Results:")
    for _, row in df_cleaned.iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Sentiment: {row['sentiment_label']}")
        print(f"Confidence: {row['confidence']:.2%}")
    
    df_cleaned.to_csv('sentiment_results.csv', index=False)

    plt.figure(figsize=(8, 6))
    sentiment_counts = df_cleaned['sentiment_label'].value_counts()
    sentiment_counts.plot(kind='bar')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png')
    plt.close()

if __name__ == "__main__":
    query = input("Enter your search query: ")
    analyze_sentiment(query)