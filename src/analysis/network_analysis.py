import networkx as nx
import pandas as pd
import community
import pycountry
import geopy.geocoders
from collections import defaultdict
import numpy as np

class NetworkAnalyzer:
    def __init__(self):
        self.G = nx.Graph()
        self.geolocator = geopy.geocoders.Nominatim(user_agent="social_media_analysis")
        
    def build_interaction_network(self, df):
        author_subreddits = df.groupby('author')['subreddit'].unique()
        
        edge_weights = defaultdict(int)
        for author, subreddits in author_subreddits.items():
            for i in range(len(subreddits)):
                for j in range(i + 1, len(subreddits)):
                    edge = tuple(sorted([subreddits[i], subreddits[j]]))
                    edge_weights[edge] += 1
        
        for (source, target), weight in edge_weights.items():
            self.G.add_edge(source, target, weight=weight)
        
        metrics = {
            'basic_metrics': {
                'num_nodes': self.G.number_of_nodes(),
                'num_edges': self.G.number_of_edges(),
                'density': nx.density(self.G),
                'average_clustering': nx.average_clustering(self.G),
                'average_shortest_path': nx.average_shortest_path_length(self.G) if nx.is_connected(self.G) else None
            },
            'centrality_metrics': {
                'degree': nx.degree_centrality(self.G),
                'betweenness': nx.betweenness_centrality(self.G),
                'eigenvector': nx.eigenvector_centrality_numpy(self.G),
                'pagerank': nx.pagerank(self.G)
            },
            'community_metrics': {
                'communities': community.best_partition(self.G),
                'modularity': community.modularity(community.best_partition(self.G), self.G)
            }
        }
        
        try:
            hits_scores = nx.hits(self.G)
            metrics['influence_metrics'] = {
                'hub_scores': hits_scores[0],
                'authority_scores': hits_scores[1]
            }
        except:
            metrics['influence_metrics'] = {
                'katz_centrality': nx.katz_centrality_numpy(self.G),
                'closeness_centrality': nx.closeness_centrality(self.G)
            }
        
        return metrics, self.G
    
    def analyze_geographical_distribution(self, df):
        """Analyze geographical distribution of posts"""
        def extract_locations(text):
            if pd.isna(text):
                return []
            common_locations = set([country.name for country in pycountry.countries])
            words = set(str(text).split())
            return list(words.intersection(common_locations))
        
        df['locations'] = df['text'].apply(extract_locations)
        
        location_counts = defaultdict(int)
        for locations in df['locations']:
            for loc in locations:
                location_counts[loc] += 1
                
        geo_data = []
        for location, count in location_counts.items():
            try:
                location_data = self.geolocator.geocode(location)
                if location_data:
                    geo_data.append({
                        'location': location,
                        'lat': location_data.latitude,
                        'lon': location_data.longitude,
                        'count': count
                    })
            except Exception as e:
                print(f"Error geocoding {location}: {str(e)}")
                
        return pd.DataFrame(geo_data)
    
    def get_influential_nodes(self, top_n=10):
        """Get most influential nodes based on multiple metrics"""
        pagerank = nx.pagerank(self.G)
        betweenness = nx.betweenness_centrality(self.G)
        degree = nx.degree_centrality(self.G)
        
        influence_scores = {}
        for node in self.G.nodes():
            influence_scores[node] = (
                0.4 * pagerank[node] +
                0.4 * betweenness[node] +
                0.2 * degree[node]
            )
        
        return dict(sorted(influence_scores.items(), 
                         key=lambda x: x[1], 
                         reverse=True)[:top_n])
    
    def detect_anomalies(self):
        """Detect potential anomalies in the network"""
        degrees = dict(self.G.degree())
        avg_degree = np.mean(list(degrees.values()))
        std_degree = np.std(list(degrees.values()))
        
        anomalies = {
            'high_degree_nodes': [],
            'isolated_nodes': [],
            'dense_clusters': []
        }
        
        for node, degree in degrees.items():
            if degree > avg_degree + 2 * std_degree:
                anomalies['high_degree_nodes'].append((node, degree))
            elif degree == 0:
                anomalies['isolated_nodes'].append(node)
        
        communities = community.best_partition(self.G)
        for community_id in set(communities.values()):
            nodes = [n for n, c in communities.items() if c == community_id]
            subgraph = self.G.subgraph(nodes)
            density = nx.density(subgraph)
            if density > 0.7: 
                anomalies['dense_clusters'].append((community_id, density))
        
        return anomalies