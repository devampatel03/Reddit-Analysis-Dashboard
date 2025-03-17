

from textblob import TextBlob
import pandas as pd
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        pass
    
    def analyze_texts(self, texts):
        """Analyze sentiment of texts using TextBlob"""
        results = []
        
        for text in texts:
            try:
                blob = TextBlob(str(text) if text else "")
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    label = "POSITIVE"
                elif polarity < -0.1:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                
                results.append({
                    'label': label,
                    'score': abs(polarity),
                    'polarity': polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                results.append({
                    'label': "NEUTRAL",
                    'score': 0.5,
                    'polarity': 0.0,
                    'subjectivity': 0.5
                })
        
        return pd.DataFrame(results)