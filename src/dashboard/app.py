import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import networkx as nx
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from src.analysis.topic_modelling import TopicModeler


root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.data_processing.loader import load_social_media_data
from src.data_processing.cleaner import TextCleaner
from src.analysis.topic_modelling import TopicModeler
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.analysis.network_analysis import NetworkAnalyzer
from src.analysis.genai_analyzer import GenAIAnalyzer

def create_network_visualizations(metrics, G, influential_nodes, anomalies):
    # Create network graph visualization
    pos = nx.spring_layout(G)
    
    # Create node traces with different colors based on influence
    node_x, node_y = [], []
    node_sizes, node_colors, node_texts = [], [], []
    max_influence = max(influential_nodes.values())
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        influence = influential_nodes.get(node, 0)
        node_sizes.append(20 + (influence/max_influence) * 50)  # Size based on influence
        node_colors.append(influence/max_influence)  # Color intensity based on influence
        node_texts.append(f"Subreddit: {node}<br>Influence Score: {influence:.3f}")
    
    # Create edges
    edge_x, edge_y = [], []
    edge_weights = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2].get('weight', 1))
    
    # Create network plot
    fig_network = go.Figure()
    
    # Add edges
    fig_network.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig_network.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_texts,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_sizes,
            color=node_colors,
            colorbar=dict(
                title='Influence Score',
                thickness=15,
                x=0.9
            )
        )
    ))
    
    fig_network.update_layout(
        title='Subreddit Interaction Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    # Create influence bar chart
    fig_influence = px.bar(
        x=list(influential_nodes.keys()),
        y=list(influential_nodes.values()),
        title='Top Influential Subreddits',
        labels={'x': 'Subreddit', 'y': 'Influence Score'},
        color=list(influential_nodes.values()),
        color_continuous_scale='Viridis'
    )
    
    # Create anomaly visualization
    anomaly_data = pd.DataFrame([
        {'type': 'High Degree Nodes', 'count': len(anomalies['high_degree_nodes'])},
        {'type': 'Isolated Nodes', 'count': len(anomalies['isolated_nodes'])},
        {'type': 'Dense Clusters', 'count': len(anomalies['dense_clusters'])}
    ])
    
    fig_anomalies = px.pie(
        anomaly_data,
        values='count',
        names='type',
        title='Network Anomalies Distribution',
        hole=0.4
    )
    
    return fig_network, fig_influence, fig_anomalies


def main():
    st.set_page_config(
        page_title="Reddit Analysis Dashboard",
        layout="wide"
    )
    
    st.title("Reddit Analysis Dashboard")
    
    # Load data
    with st.spinner('Loading data...'):
        try:
            df = load_social_media_data()
            if df.empty:
                st.error("No data loaded. Please check your data file.")
                return
            st.success(f"Loaded {len(df)} posts")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    # Clean text with error handling
    try:
        cleaner = TextCleaner()
        df['cleaned_text'] = df['text'].fillna('').apply(cleaner.clean_text)
    except Exception as e:
        st.error(f"Error cleaning text: {str(e)}")
        return
    

    st.sidebar.title("Filters")
    
    # Date range filter
    min_date = df['created_utc'].min().date()
    max_date = df['created_utc'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    subreddits = sorted(df['subreddit'].unique())
    selected_subreddits = st.sidebar.multiselect(
        "Select Subreddits",
        subreddits,
        default=subreddits[:5]
    )
    
    # Filter data
    mask = (
        (df['created_utc'].dt.date >= date_range[0]) & 
        (df['created_utc'].dt.date <= date_range[1]) &
        (df['subreddit'].isin(selected_subreddits))
    )
    filtered_df = df[mask]
    
    # Display basic stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", len(filtered_df))
    col2.metric("Average Score", f"{filtered_df['score'].mean():.1f}")
    col3.metric("Total Comments", filtered_df['num_comments'].sum())
    
    daily_activity = filtered_df.set_index('created_utc').resample('D').size()



    st.subheader("Post Activity Over Time")
    daily_posts = filtered_df.set_index('created_utc').resample('D').size()
    fig = px.line(
        daily_posts, 
        title='Daily Post Count',
        labels={'created_utc': 'Date', 'value': 'Number of Posts'}
    )
    st.plotly_chart(fig, use_container_width=True)



    st.subheader("Top Posts")
    top_posts = filtered_df.nlargest(5, 'score')[['title', 'score', 'num_comments', 'subreddit']]
    st.dataframe(top_posts)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Topic Analysis", 
        "Sentiment Analysis", 
        "Network Analysis",
        "Time Series Analysis"
    ])
    
    with tab1:
        st.subheader("Topic Analysis")


        # Generate topics and visualizations
        topic_modeler = TopicModeler(n_topics=5)
        topics_df = topic_modeler.generate_topics(filtered_df['cleaned_text'])
        viz_data = topic_modeler.generate_topic_visualization(filtered_df['cleaned_text'])
        
        # 1. Topic Distribution Overview
        st.subheader("Topic Distribution")
        fig_dist = px.pie(
            pd.DataFrame(viz_data['topic_sizes']),
            values='docs',
            names='topic',
            title='Distribution of Documents Across Topics'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # 2. Topic Details
        st.subheader("Topic Details")
        for summary in viz_data['topic_summaries']:
            with st.expander(f"Topic {summary['topic_id']} ({summary['size']} documents)"):
                st.write("#### Top Documents:")
                for i, doc in enumerate(summary['top_documents'], 1):
                    st.write(f"{i}. {doc[:200]}...")
                st.write(f"Proportion of total documents: {summary['proportion']:.2%}")
        
        # 3. Interactive Topic Explorer
        st.subheader("Interactive Topic Explorer")
        embeddings_3d, metadata = topic_modeler.generate_embeddings(filtered_df['cleaned_text'])
        
        #  3D scatter plot
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=[m['topic'] for m in metadata],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Topic")
            ),
            text=[f"Topic: {m['topic']}<br>{m['text']}" for m in metadata],
            hoverinfo='text'
        )])

        fig_3d.update_layout(
            title='Topic Clusters in 3D Space',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # 4. Topic Evolution Over Time
        if 'created_utc' in filtered_df.columns:
            st.subheader("Topic Evolution Over Time")
            filtered_df['topic'] = [m['topic'] for m in metadata]
            filtered_df['date'] = pd.to_datetime(filtered_df['created_utc']).dt.date
            
            topic_evolution = filtered_df.groupby(['date', 'topic']).size().unstack(fill_value=0)
            
            fig_evolution = px.line(
                topic_evolution,
                title='Topic Evolution Over Time',
                labels={'value': 'Number of Documents', 'date': 'Date'},
            )
            st.plotly_chart(fig_evolution, use_container_width=True)

    
    with tab2:
        st.subheader("Sentiment Analysis")
        sentiment_analyzer = SentimentAnalyzer()
        
        with st.spinner('Analyzing sentiments...'):
            sentiments = sentiment_analyzer.analyze_texts(filtered_df['cleaned_text'].tolist())
        
        #  sentiment distribution
        fig_sentiment = px.histogram(
            sentiments, 
            x='label',
            title='Sentiment Distribution',
            color='label',
            labels={'label': 'Sentiment', 'count': 'Number of Posts'},
            template='plotly_white'
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        #  average sentiment scores by subreddit
        filtered_df['sentiment'] = sentiments['label']
        filtered_df['sentiment_score'] = sentiments['score']
        
        sentiment_stats = filtered_df.groupby('subreddit').agg({
            'sentiment': lambda x: x.value_counts().to_dict(),
            'sentiment_score': 'mean'
        }).round(3)
        
        st.subheader("Sentiment Analysis by Subreddit")
        st.dataframe(sentiment_stats)
        
        #  average sentiment scores by subreddit
        fig_subreddit = px.bar(
            sentiment_stats.reset_index(),
            x='subreddit',
            y='sentiment_score',
            title='Average Sentiment Score by Subreddit',
            labels={'sentiment_score': 'Average Sentiment Score', 'subreddit': 'Subreddit'},
            template='plotly_white'
        )
        st.plotly_chart(fig_subreddit, use_container_width=True)

    with tab3:
        st.subheader("Network Analysis")
        network_analyzer = NetworkAnalyzer()
    
        with st.spinner('Analyzing network structure...'):
            metrics, G = network_analyzer.build_interaction_network(filtered_df)
            geo_data = network_analyzer.analyze_geographical_distribution(filtered_df)
            influential_nodes = network_analyzer.get_influential_nodes()
            anomalies = network_analyzer.detect_anomalies()
        
            fig_network, fig_influence, fig_anomalies = create_network_visualizations(
                metrics, G, influential_nodes, anomalies
            )
    
    #  network visualization
        st.plotly_chart(fig_network, use_container_width=True)
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("Influence Analysis")
            st.plotly_chart(fig_influence, use_container_width=True)
        
            st.metric("Network Density", f"{metrics['basic_metrics']['density']:.3f}")
            st.metric("Average Clustering", f"{metrics['basic_metrics']['average_clustering']:.3f}")
    
        with col2:
            st.subheader("Anomaly Detection")
            st.plotly_chart(fig_anomalies, use_container_width=True)
        
            if metrics['basic_metrics']['average_shortest_path']:
                st.metric("Avg Path Length", f"{metrics['basic_metrics']['average_shortest_path']:.2f}")
            st.metric("Modularity", f"{metrics['community_metrics']['modularity']:.3f}")

        #  geographical distribution
        st.subheader("Geographical Distribution of Mentions")
        fig = px.scatter_mapbox(
            geo_data,
            lat='lat',
            lon='lon',
            size='count',
            hover_name='location',
            color='count',
            title='Geographic Distribution of Content',
            mapbox_style='carto-positron',
            zoom=1
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Time Series Analysis")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        total_days = (filtered_df['created_utc'].max() - filtered_df['created_utc'].min()).days
        avg_daily_posts = len(filtered_df) / total_days
        peak_day = daily_activity.idxmax()
        
        metric_col1.metric("Analysis Period", f"{total_days} days")
        metric_col2.metric("Avg Daily Posts", f"{avg_daily_posts:.1f}")
        metric_col3.metric("Peak Day", peak_day.strftime('%Y-%m-%d'))
        
        st.subheader("Activity Timeline")
        timeline_type = st.radio(
            "Select Timeline View",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True
        )
        
        if timeline_type == "Weekly":
            activity_data = filtered_df.set_index('created_utc').resample('W').size()
            period = "Week"
        elif timeline_type == "Monthly":
            activity_data = filtered_df.set_index('created_utc').resample('M').size()
            period = "Month"
        else:
            activity_data = daily_activity
            period = "Day"
        
        #  time series plot
        fig_time = px.line(
            activity_data,
            title=f'Post Activity by {period}',
            labels={'value': 'Number of Posts', 'created_utc': 'Date'},
            template='plotly_white'
        )
        fig_time.update_traces(mode='lines+markers')
        st.plotly_chart(fig_time, use_container_width=True)
        
        #  GenAIAnalyzer
        genai_analyzer = GenAIAnalyzer()
        
        # Generate and display trend summary 
        with st.spinner("Analyzing trends..."):
            trend_summary = genai_analyzer.generate_trend_summary(filtered_df, activity_data)
            st.info("📊 AI-Generated Trend Analysis")
            st.markdown(trend_summary)
        
        # Activity patterns 
        st.subheader("Activity Patterns")
        pattern_col1, pattern_col2 = st.columns(2)
        
        with pattern_col1:
            #  hourly activity plot
            peak_times = filtered_df['created_utc'].dt.hour.value_counts().sort_index()
            fig_hours = px.bar(
                peak_times,
                title='Post Activity by Hour (24h)',
                labels={'index': 'Hour of Day', 'value': 'Number of Posts'},
                template='plotly_white'
            )
            st.plotly_chart(fig_hours, use_container_width=True)
        
        with pattern_col2:
            dow_activity = filtered_df['created_utc'].dt.day_name().value_counts()
            fig_dow = px.bar(
                dow_activity,
                title='Post Activity by Day of Week',
                labels={'index': 'Day of Week', 'value': 'Number of Posts'},
                template='plotly_white'
            )
            st.plotly_chart(fig_dow, use_container_width=True)
        
        #  Q&A section
        st.subheader("💬 Ask AI About the Trends")
        
        # # Add preset questions
        # preset_question = st.selectbox(
        #     "Select a preset question or type your own below:",
        #     [
        #         "What are the peak activity patterns?",
        #         "What trends do you see in the data?",
        #         "When is the best time to post?",
        #         "What are the weekly patterns?",
        #         "Custom question"
        #     ]
        # )
        
        # Handle custom or preset questions
        # if preset_question == "Custom question":



        user_question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the most active times of day?"
        )
       
        
        if user_question:
            with st.spinner("Analyzing..."):
                answer = genai_analyzer.answer_query(user_question, filtered_df)
                st.success("🤖 Analysis Results")
                st.markdown(answer)
        
        

        

if __name__ == "__main__":
    main()