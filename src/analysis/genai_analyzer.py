import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv

class GenAIAnalyzer:
    def __init__(self):
        load_dotenv()
        
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def generate_trend_summary(self, df, time_series_data):
        """Generate natural language summary of time series trends using Gemini"""
        total_posts = len(df)
        peak_date = time_series_data.idxmax()
        peak_posts = time_series_data.max()
        avg_posts = time_series_data.mean()
        
        prompt = f"""
        Analyze this Reddit data and provide insights in a clear, concise manner:
        
        Data Overview:
        - Total posts: {total_posts}
        - Date range: {df['created_utc'].min().date()} to {df['created_utc'].max().date()}
        - Peak activity: {peak_date.date()} with {int(peak_posts)} posts
        - Average daily posts: {avg_posts:.1f}
        - Top subreddits: {', '.join(df['subreddit'].value_counts().nlargest(3).index)}
        
        Please provide:
        1. Key trends and patterns
        2. Notable insights about peak activity
        3. Analysis of subreddit distribution
        4. Recommendations for further investigation
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def answer_query(self, query, df, context=None):
        """Answer questions about trends, topics, and narratives using Gemini"""
        topic_trends = {
            'posts_by_subreddit': df.groupby('subreddit').size().to_dict(),
            'temporal_trends': df.groupby(df['created_utc'].dt.date).size().to_dict(),
            'top_scoring_themes': df.nlargest(5, 'score')[['title', 'score', 'subreddit']].to_dict('records'),
            'recent_trends': df.sort_values('created_utc', ascending=False).head(5)[['title', 'subreddit']].to_dict('records')
        }

        enhanced_context = f"""
        Detailed Reddit Data Analysis Context:
        
        1. Overall Statistics:
        - Total posts analyzed: {len(df)}
        - Unique subreddits: {df['subreddit'].nunique()}
        - Time period: {df['created_utc'].min().date()} to {df['created_utc'].max().date()}
        
        2. Top Active Subreddits (with post counts):
        {', '.join(f'{sub}: {count}' for sub, count in sorted(topic_trends['posts_by_subreddit'].items(), key=lambda x: x[1], reverse=True)[:5])}
        
        3. Current Trending Topics:
        {', '.join(post['title'][:100] + f" (r/{post['subreddit']})" for post in topic_trends['recent_trends'])}
        
        4. Most Impactful Content:
        {', '.join(f"'{post['title'][:50]}...' (Score: {post['score']})" for post in topic_trends['top_scoring_themes'])}
        
        5. Activity Patterns:
        - Peak posting day: {max(topic_trends['temporal_trends'].items(), key=lambda x: x[1])[0]}
        - Average daily posts: {df.groupby(df['created_utc'].dt.date).size().mean():.1f}
        """

        prompt = f"""
        You are analyzing Reddit data trends and patterns. Using only the provided context below, 
        answer the following question with specific details and data-driven insights.

        Context:
        {enhanced_context}

        Question: {query}

        Please provide:
        1. A direct answer to the question
        2. Supporting data from the context
        3. Any relevant trends or patterns
        4. Specific examples if applicable

        Structure your response in a clear, concise manner using bullet points or short paragraphs.
        Only use information from the provided context.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}\nPlease try rephrasing your question."
        
    def analyze_topics(self, df, subreddit=None):
        if subreddit:
            data = df[df['subreddit'] == subreddit]
        else:
            data = df

        recent_posts = data.nlargest(10, 'created_utc')[['title', 'text', 'score']]

        prompt = f"""
        Analyze these recent Reddit posts and identify key themes and topics:

        Posts:
        {recent_posts[['title', 'score']].to_string()}

        Please provide:
        1. Main themes and topics
        2. Emerging trends
        3. Notable patterns in user engagement
        4. Any significant correlations between topics and scores
        """

        response = self.model.generate_content(prompt)
        return response.text