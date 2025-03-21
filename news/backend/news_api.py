import requests
import json
from datetime import datetime, timedelta

API_KEY = "87bdb1e5-bcc2-43e8-9b96-d3f7a956ec61"

def get_news_articles():
    # Get news from the last 24 hours for real-time relevance
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"https://newsapi.org/v2/everything?q=finance OR stock market OR investing&from={yesterday}&sortBy=publishedAt&language=en&pageSize=20&apiKey={API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = json.loads(response.text)
        articles = data.get("articles", [])
        
        # Process and clean the articles
        processed_articles = []
        for article in articles:
            if article.get("title") and article.get("description"):
                processed_article = {
                    "title": article["title"],
                    "description": article["description"],
                    "url": article.get("url", ""),
                    "publishedAt": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "urlToImage": article.get("urlToImage", ""),
                    "sentiment": analyze_sentiment(article["title"] + " " + article["description"])
                }
                processed_articles.append(processed_article)
        
        return processed_articles[:20]  # Ensure we return max 20 articles
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def analyze_sentiment(text):
    # Simple sentiment analysis based on keywords
    positive_words = ['surge', 'rise', 'gain', 'up', 'positive', 'growth', 'profit', 'success']
    negative_words = ['fall', 'drop', 'down', 'negative', 'loss', 'decline', 'risk', 'concern']
    
    text = text.lower()
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    return "neutral"

def filter_news_articles(articles, keywords):
    filtered_articles = []
    for article in articles:
        for keyword in keywords:
            if (keyword.lower() in article["title"].lower() or 
                (article.get("description") and keyword.lower() in article["description"].lower())):
                filtered_articles.append(article)
                break
    return filtered_articles

def get_curated_news_feed(keywords=None):
    articles = get_news_articles()
    if keywords:
        return filter_news_articles(articles, keywords)
    return articles