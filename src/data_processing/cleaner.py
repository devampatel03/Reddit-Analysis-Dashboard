import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextCleaner:
    def __init__(self):
        try:
            required_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
            for resource in required_resources:
                try:
                    nltk.find('corpora/' + resource)
                except LookupError:
                    print(f"Downloading {resource}...")
                    nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK resources: {str(e)}")
            raise

        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error initializing text cleaning components: {str(e)}")
            raise

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        try:
            text = str(text).lower()
            
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
            text = re.sub(r'[^\w\s]', '', text)
            
            tokens = word_tokenize(text)
            
            tokens = [self.lemmatizer.lemmatize(token) 
                     for token in tokens 
                     if token not in self.stop_words]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return text 