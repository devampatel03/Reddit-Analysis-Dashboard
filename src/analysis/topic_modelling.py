from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import json

class TopicModeler:
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.95,
            min_df=2
        )
        self.nmf = NMF(n_components=n_topics, random_state=42)
        self.svd = TruncatedSVD(n_components=3, random_state=42)
        self.tsne = TSNE(n_components=3, random_state=42)
        self.scaler = MinMaxScaler()
        
    def generate_topics(self, texts):
        """Generate topics using TF-IDF and NMF"""
        dtm = self.vectorizer.fit_transform(texts)
        
        topic_matrix = self.nmf.fit_transform(dtm)
        
        terms = self.vectorizer.get_feature_names_out()
        
        topics = []
        for topic_idx in range(self.n_topics):
            top_term_indices = self.nmf.components_[topic_idx].argsort()[:-10:-1]
            top_terms = [terms[i] for i in top_term_indices]
            
            top_doc_indices = topic_matrix[:, topic_idx].argsort()[-5:][::-1]
            top_docs = [texts[i][:100] for i in top_doc_indices]
            
            topics.append({
                'topic_id': topic_idx,
                'terms': ', '.join(top_terms),
                'representative_docs': '\n'.join(top_docs),
                'weight': np.sum(topic_matrix[:, topic_idx]),
                'top_docs': top_docs
            })
            
        return pd.DataFrame(topics)
    
    def generate_embeddings(self, texts, output_dir='static/projector'):
        """Generate document embeddings for visualization"""
        dtm = self.vectorizer.fit_transform(texts)
        
        embeddings = self.svd.fit_transform(dtm)
        
        embeddings_3d = self.tsne.fit_transform(embeddings)
        
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings_3d)
        
        topic_matrix = self.nmf.fit_transform(dtm)
        topic_assignments = topic_matrix.argmax(axis=1)
        
        metadata = []
        for idx, (text, topic) in enumerate(zip(texts, topic_assignments)):
            metadata.append({
                'index': idx,
                'text': text[:100],
                'topic': int(topic)
            })
        
        with open(os.path.join(output_dir, 'metadata.tsv'), 'w', encoding='utf-8') as f:
            f.write('Index\tText\tTopic\n')
            for item in metadata:
                f.write(f"{item['index']}\t{item['text']}\t{item['topic']}\n")
        
        return embeddings_3d, metadata
    
    def generate_topic_visualization(self, texts, min_topic_size=5):
        """Generate interactive topic visualizations"""
        dtm = self.vectorizer.fit_transform(texts)
        
        topic_matrix = self.nmf.fit_transform(dtm)
        topic_assignments = topic_matrix.argmax(axis=1)
        
        viz_data = {
            'topic_sizes': [],
            'topic_clusters': [],
            'topic_summaries': []
        }
        
        for topic_idx in range(self.n_topics):
            # documents for this topic
            topic_docs = [text for text, assignment in zip(texts, topic_assignments) 
                        if assignment == topic_idx]
            
            topic_size = len(topic_docs)
            viz_data['topic_sizes'].append({
                'topic': f'Topic {topic_idx}',
                'size': topic_size,
                'docs': len(topic_docs)
            })
            
            top_docs = sorted(
                [(score[topic_idx], text) 
                 for score, text in zip(topic_matrix, texts)],
                reverse=True
            )[:5]
            
            #  topic summary
            summary = {
                'topic_id': topic_idx,
                'size': topic_size,
                'top_documents': [doc for _, doc in top_docs],
                'proportion': topic_size / len(texts)
            }
            viz_data['topic_summaries'].append(summary)
        
        return viz_data