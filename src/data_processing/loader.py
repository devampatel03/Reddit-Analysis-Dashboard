import pandas as pd
import json
from pathlib import Path

def load_social_media_data():
    """Load and process Reddit data from JSONL file"""
    data_path = Path("data/raw/reddit_data.jsonl")
    
    posts = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                post = json.loads(line)
                processed_post = {
                    'id': post['data']['id'],
                    'subreddit': post['data']['subreddit'],
                    'title': post['data']['title'],
                    'text': post['data']['selftext'],
                    'author': post['data']['author'],
                    'created_utc': pd.to_datetime(post['data']['created_utc'], unit='s'),
                    'score': post['data']['score'],
                    'num_comments': post['data']['num_comments'],
                    'upvote_ratio': post['data'].get('upvote_ratio', None)
                }
                posts.append(processed_post)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line: {e}")
                continue
    
    df = pd.DataFrame(posts)
    return df