import pandas as pd
import json
from pathlib import Path

def load_social_media_data():
    """
    Load and combine social media data from different sources
    """
    data_path = Path("data/raw")
    
    def load_json_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return pd.json_normalize(json.load(f))
    
    # Load different data sources
    twitter_data = load_json_data(data_path / "twitter_data.json")
    reddit_data = load_json_data(data_path / "reddit_data.json")
    
    # Standardize columns and combine data
    twitter_data['platform'] = 'twitter'
    reddit_data['platform'] = 'reddit'
    
    combined_data = pd.concat([twitter_data, reddit_data], ignore_index=True)
    return combined_data