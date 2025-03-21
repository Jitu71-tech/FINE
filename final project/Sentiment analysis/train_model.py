import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from data_cleaning import clean_news_data, TextCleaner
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_and_prepare_data(csv_path, excel_path):
    """Load and prepare the training data from both CSV and Excel files"""
    # Read the CSV file with appropriate encoding
    df_csv = pd.read_csv(csv_path, encoding='latin1')
    
    # Read the Excel file
    df_excel = pd.read_excel(excel_path)
    
    # Clean the data
    cleaner = TextCleaner()
    
    # Clean the text column for both datasets
    df_csv = cleaner.clean_dataframe(df_csv, ['text'])
    df_excel = cleaner.clean_dataframe(df_excel, ['Caption'])
    
    # Convert sentiment to binary (1 for positive, 0 for negative/neutral)
    sentiment_map = {
        'positive': 1,
        'negative': 0,
        'neutral': 0
    }
    df_csv['sentiment'] = df_csv['sentiment'].map(sentiment_map)
    df_excel['sentiment'] = df_excel['LABEL'].map(sentiment_map)
    
    # Combine the datasets
    df_combined = pd.concat([
        df_csv[['text_cleaned', 'sentiment']],
        df_excel[['Caption_cleaned', 'sentiment']].rename(columns={'Caption_cleaned': 'text_cleaned'})
    ], ignore_index=True)
    
    # Remove any rows with NaN values
    df_combined = df_combined.dropna(subset=['text_cleaned', 'sentiment'])
    
    # Remove duplicates
    df_combined = df_combined.drop_duplicates(subset=['text_cleaned'])
    
    return df_combined

def train_model(df):
    """Train the sentiment analysis model"""
    # Prepare features and target
    X = df['text_cleaned']
    y = df['sentiment']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and fit the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print some example predictions
    print("\nExample Predictions:")
    sample_indices = np.random.randint(0, len(X_test), 5)
    for idx in sample_indices:
        text = X_test.iloc[idx]
        true_sentiment = y_test.iloc[idx]
        pred_sentiment = model.predict(vectorizer.transform([text]))[0]
        print(f"\nText: {text}")
        print(f"True Sentiment: {'Positive' if true_sentiment == 1 else 'Negative/Neutral'}")
        print(f"Predicted Sentiment: {'Positive' if pred_sentiment == 1 else 'Negative/Neutral'}")
    
    return model, vectorizer

def save_model(model, vectorizer):
    """Save the trained model and vectorizer"""
    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("\nModel and vectorizer saved successfully!")

def main():
    # Load and prepare data
    print("Loading data...")
    df = load_and_prepare_data('test.csv.xls', 'labeledText.xlsx')
    
    if df is None:
        print("Failed to load data. Please check your data format.")
        return
    
    print(f"\nData loaded successfully! Shape: {df.shape}")
    print(f"Positive samples: {sum(df['sentiment'] == 1)}")
    print(f"Negative/Neutral samples: {sum(df['sentiment'] == 0)}")
    
    # Train the model
    print("\nTraining model...")
    model, vectorizer = train_model(df)
    
    # Save the model
    save_model(model, vectorizer)

if __name__ == "__main__":
    main()