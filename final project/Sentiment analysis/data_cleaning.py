import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import unicodedata

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Comprehensive text cleaning function
        """
        if not isinstance(text, str):
            return ''
            
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Remove Unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
        # 3. Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # 4. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 5. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. Tokenize
        tokens = word_tokenize(text)
        
        # 7. Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # 8. Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # 9. Remove short words (less than 2 characters)
        tokens = [token for token in tokens if len(token) > 1]
        
        # 10. Join tokens back into text
        return ' '.join(tokens)

    def clean_dataframe(self, df, text_columns):
        """
        Clean multiple text columns in a dataframe
        """
        df_cleaned = df.copy()
        
        for col in text_columns:
            if col in df_cleaned.columns:
                # Create new column with cleaned text
                df_cleaned[f'{col}_cleaned'] = df_cleaned[col].apply(self.clean_text)
                
                # Remove rows with empty cleaned text
                df_cleaned = df_cleaned[df_cleaned[f'{col}_cleaned'].str.strip() != '']
        
        return df_cleaned

    def remove_duplicates(self, df, columns):
        """
        Remove duplicate entries based on specified columns
        """
        return df.drop_duplicates(subset=columns)

    def handle_missing_values(self, df, columns, method='fill'):
        """
        Handle missing values in specified columns
        """
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df_cleaned.columns:
                if method == 'fill':
                    df_cleaned[col] = df_cleaned[col].fillna('')
                elif method == 'drop':
                    df_cleaned = df_cleaned.dropna(subset=[col])
        
        return df_cleaned

def clean_news_data(df):
    """
    Specific function for cleaning news data
    """
    cleaner = TextCleaner()
    
    # 1. Handle missing values
    df = cleaner.handle_missing_values(df, ['title', 'description'])
    
    # 2. Combine title and description
    df['text'] = df['title'] + ' ' + df['description']
    
    # 3. Clean the combined text
    df = cleaner.clean_dataframe(df, ['text'])
    
    # 4. Remove duplicates
    df = cleaner.remove_duplicates(df, ['text_cleaned'])
    
    # 5. Remove very short articles
    df = df[df['text_cleaned'].str.split().str.len() > 5]
    
    return df

def analyze_text_statistics(df, text_column):
    """
    Analyze text statistics after cleaning
    """
    stats = {
        'total_articles': len(df),
        'avg_words_per_article': df[text_column].str.split().str.len().mean(),
        'max_words': df[text_column].str.split().str.len().max(),
        'min_words': df[text_column].str.split().str.len().min(),
        'unique_words': len(set(' '.join(df[text_column]).split())),
        'avg_word_length': df[text_column].str.split().apply(lambda x: sum(len(word) for word in x) / len(x)).mean()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    import requests
    
    # Fetch sample data
    NEWSAPI_KEY = "6d69be83469b42939e27626f3b453383"
    url = f"https://newsapi.org/v2/top-headlines?category=technology&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['articles'])
        
        # Clean the data
        df_cleaned = clean_news_data(df)
        
        # Analyze statistics
        stats = analyze_text_statistics(df_cleaned, 'text_cleaned')
        
        print("\nData Cleaning Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        
        print("\nSample of cleaned text:")
        print(df_cleaned['text_cleaned'].head()) 