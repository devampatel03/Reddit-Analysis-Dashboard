
# Reddit Data Analysis Dashboard

## Project Overview
This project implements a data analysis dashboard for Reddit content, focusing on:
1. Topic modeling and visualization
2. Sentiment analysis
3. Network analysis of subreddit interactions
4. Time series analysis with AI-powered insights

## Technical Architecture

### Core Components
1. **Data Processing**
   - Data loading and cleaning
   - Text preprocessing
   - Feature extraction

2. **Analysis Modules**
   - Topic Modeling
   - Sentiment Analysis
   - Network Analysis
   - Time Series Analysis

3. **AI Integration**
   - Google Gemini for trend analysis
   - Natural language insights
   - Query-based data exploration

### File Structure
```plaintext
python-data-science/
├── src/
│   ├── analysis/
│   │   ├── topic_modelling.py
│   │   ├── sentiment_analysis.py
│   │   ├── network_analysis.py
│   │   ├── genai_analyzer.py
│   │
│   ├── data_processing/
│   │   ├── loader.py
│   │   └── cleaner.py
│   └── dashboard/
│       └── app.py
├── requirements.txt
└── README.md
```


## Implementation Details

### Setup Instructions
1. **Environment Setup**
```batch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configuration**
Create .env file:
````text
GOOGLE_API_KEY=your_gemini_api_key_here
````

3. **Running the Dashboard**
```batch
streamlit run run_dashboard.py
```

## Key Functionalities

## Key Insights

### 1. Time Series Analysis
This feature provides insights into Reddit activity over time, helping identify trends, patterns, and anomalies.

#### Insights Provided:

- **Daily/Weekly/Monthly Post Activity**:
  - **Insight**: Understand the overall activity level on Reddit over time.
  - **Details**: Identify trends, seasonality, and anomalies in posting behavior.
  - **Visualization**: Line chart showing post counts over time.

- **Peak Activity Times (Hourly)**:
  - **Insight**: Determine the most active hours of the day for Reddit discussions.
  - **Details**: Optimize posting schedules to maximize visibility and engagement.
  - **Visualization**: Bar chart showing post counts by hour.

- **Activity by Day of the Week**:
  - **Insight**: Identify which days of the week have the highest and lowest engagement.
  - **Details**: Tailor content strategy based on weekday-specific trends.
  - **Visualization**: Bar chart showing post counts by day of the week.

- **AI-Generated Trend Analysis**:
  - **Insight**: Obtain a natural language summary of key trends and patterns.
  - **Details**: Quickly grasp the main takeaways from the time series data.
  - **Visualization**: Text summary generated by Google's Gemini model.

- **Interactive Q&A**:
  - **Insight**: Answer specific questions about the data using AI.
  - **Details**: Explore trends, topics, and patterns through natural language queries.
  - **Visualization**: Chatbot interface powered by Google's Gemini model.

![alt text](<Screenshot 2025-03-17 002835.png>)
![alt text](<Screenshot 2025-03-17 003039.png>)
![alt text](<Screenshot 2025-03-17 003858.png>)


### 2. Topic Analysis
This feature identifies and visualizes the main topics discussed on Reddit, providing insights into content themes and trends.

#### Insights Provided:

- **Topic Distribution Pie Chart**:
  - **Insight**: Visualize the proportion of documents belonging to each identified topic.
  - **Details**: Understand the relative importance of different themes in the dataset.
  - **Visualization**: Pie chart showing topic distribution.

- **Topic Details (Expandable Sections)**:
  - **Insight**: Explore representative documents for each topic.
  - **Details**: Understand the content and context of each topic.
  - **Visualization**: Expandable sections with top posts for each topic.

- **3D Interactive Topic Explorer**:
  - **Insight**: Visualize relationships between documents in a 3D space.
  - **Details**: Identify clusters of related content and explore topic boundaries.
  - **Visualization**: 3D scatter plot of topic embeddings.

- **Topic Evolution Over Time**:
  - **Insight**: Track how topics change over time.
  - **Details**: Identify emerging trends and declining interests.
  - **Visualization**: Line chart showing topic frequency over time.

  ![alt text](<Screenshot 2025-03-17 002850.png>)
  ![alt text](<Screenshot 2025-03-17 002927.png>)

### 3. Sentiment Analysis
This feature analyzes the sentiment of Reddit posts, providing insights into community mood and emotional trends.

#### Insights Provided:

- **Overall Sentiment Distribution**:
  - **Insight**: Understand the general sentiment of Reddit posts.
  - **Details**: Identify the proportion of positive, negative, and neutral posts.
  - **Visualization**: Pie chart showing sentiment distribution.


- **Sentiment by Subreddit**:
  - **Insight**: Compare sentiment across different subreddits.
  - **Details**: Identify which communities have the most positive or negative discussions.
  - **Visualization**: Bar chart comparing sentiment scores across subreddits.

![alt text](<Screenshot 2025-03-17 002949.png>)


### 4. Network Analysis
This feature visualizes relationships between subreddits, providing insights into community structure and influence.

#### Insights Provided:

- **Subreddit Interaction Network**:
  - **Insight**: Visualize relationships between subreddits.
  - **Details**: Identify which communities interact most frequently.
  - **Visualization**: Network graph showing subreddit connections.

- **Top Influential Subreddits**:
  - **Insight**: Identify the most influential subreddits in the network.
  - **Details**: Understand which communities have the most significant impact.
  - **Visualization**: Bar chart showing influence scores for each subreddit.

- **Network Anomalies Distribution**:
  - **Insight**: Identify unusual patterns in the network.
  - **Details**: Detect isolated communities, high-degree nodes, and dense clusters.
  - **Visualization**: Pie chart showing the distribution of network anomalies.

- **Geographic Distribution of Content**:
  - **Insight**: Understand the geographical distribution of content.
  - **Details**: Identify which locations have the most activity.
  - **Visualization**: Scatter plot on a map showing content distribution by location.

![alt text](<Screenshot 2025-03-17 003011.png>)
![alt text](<Screenshot 2025-03-17 003024.png>)


## License
MIT License

