import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter

class SmithsonianAIAnalyzer:
    """Class for AI-powered analytics on Smithsonian data"""
    
    def __init__(self, df=None):
        self.df = df
        self.vectorizer = None
        self.clusters = None
        self.cluster_model = None
        self.pca_model = None
        self.feature_matrix = None
    
    def set_data(self, df):
        """Set the DataFrame to analyze"""
        self.df = df
    
    def preprocess_text(self, text_column='title'):
        """Preprocess text data for analysis"""
        if self.df is None:
            raise ValueError("No data available. Call set_data() first.")
        
        # Fill NaN values
        self.df[text_column] = self.df[text_column].fillna('')
        
        # Create text features using TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        text_features = self.vectorizer.fit_transform(self.df[text_column])
        
        return text_features
    
    def cluster_items(self, n_clusters=5, text_column='title', algorithm='kmeans'):
        """Cluster items based on text features"""
        if self.df is None:
            raise ValueError("No data available. Call set_data() first.")
        
        # Get text features
        text_features = self.preprocess_text(text_column)
        
        # Create clusters
        if algorithm.lower() == 'kmeans':
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
            self.clusters = self.cluster_model.fit_predict(text_features)
        elif algorithm.lower() == 'dbscan':
            self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
            self.clusters = self.cluster_model.fit_predict(text_features)
        else:
            raise ValueError("Unsupported algorithm. Use 'kmeans' or 'dbscan'.")
        
        # Add cluster labels to the DataFrame
        self.df['cluster'] = self.clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_items = self.df[self.df['cluster'] == cluster_id]
            top_terms = self._get_top_terms_for_cluster(cluster_id, text_features)
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_items),
                'top_terms': top_terms,
                'sample_items': cluster_items['title'].head(3).tolist()
            }
        
        return cluster_analysis
    
    def _get_top_terms_for_cluster(self, cluster_id, text_features, top_n=5):
        """Get the top terms for a specific cluster"""
        if self.cluster_model is None:
            raise ValueError("No clustering model available. Call cluster_items() first.")
        
        if hasattr(self.cluster_model, 'cluster_centers_'):
            # For KMeans
            centroid = self.cluster_model.cluster_centers_[cluster_id]
            indices = centroid.argsort()[-top_n:][::-1]
            top_terms = [self.vectorizer.get_feature_names_out()[i] for i in indices]
            return top_terms
        else:
            # For DBSCAN or other algorithms without centroids
            cluster_docs = text_features[self.clusters == cluster_id]
            if cluster_docs.shape[0] == 0:
                return []
                
            cluster_tf_idf = np.asarray(cluster_docs.mean(axis=0)).flatten()
            indices = cluster_tf_idf.argsort()[-top_n:][::-1]
            top_terms = [self.vectorizer.get_feature_names_out()[i] for i in indices]
            return top_terms
    
    def visualize_clusters(self, dim_reduction='pca'):
        """Visualize the clusters using dimensionality reduction"""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("No cluster data available. Call cluster_items() first.")
        
        # Get text features
        text_features = self.preprocess_text()
        
        # Apply dimensionality reduction
        if dim_reduction.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif dim_reduction.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            raise ValueError("Unsupported dimensionality reduction. Use 'pca' or 'tsne'.")
        
        # Convert sparse matrix to dense if needed
        if hasattr(text_features, 'toarray'):
            features_dense = text_features.toarray()
        else:
            features_dense = text_features
            
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_dense)
        
        # Reduce dimensions
        reduced_features = reducer.fit_transform(features_scaled)
        
        # Plot the clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                              c=self.clusters, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Clusters Visualization using {dim_reduction.upper()}')
        plt.xlabel(f'{dim_reduction.upper()} Component 1')
        plt.ylabel(f'{dim_reduction.upper()} Component 2')
        plt.tight_layout()
        
        return plt
    
    def predict_related_items(self, item_id, n_recommendations=5):
        """Predict related items based on clustering and feature similarity"""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("No cluster data available. Call cluster_items() first.")
        
        # Find the item
        if item_id not in self.df['id'].values:
            raise ValueError(f"Item with ID {item_id} not found in the dataset.")
        
        item = self.df[self.df['id'] == item_id].iloc[0]
        item_cluster = item['cluster']
        
        # Get items from the same cluster
        cluster_items = self.df[(self.df['cluster'] == item_cluster) & (self.df['id'] != item_id)]
        
        # If we have enough items in the cluster, return them
        if len(cluster_items) >= n_recommendations:
            return cluster_items.head(n_recommendations)
        
        # Otherwise, find items from other clusters that might be related
        remaining_items = n_recommendations - len(cluster_items)
        other_items = self.df[(self.df['cluster'] != item_cluster) & (self.df['id'] != item_id)]
        
        # Use text similarity to find related items
        text_features = self.preprocess_text()
        item_idx = self.df[self.df['id'] == item_id].index[0]
        item_vector = text_features[item_idx]
        
        # Calculate similarity scores
        if hasattr(text_features, 'toarray'):
            # For sparse matrices
            similarities = [(i, np.dot(item_vector.toarray().flatten(), text_features[i].toarray().flatten())) 
                           for i in other_items.index]
        else:
            # For dense matrices
            similarities = [(i, np.dot(item_vector.flatten(), text_features[i].flatten())) 
                           for i in other_items.index]
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top similar items
        top_indices = [idx for idx, _ in similarities[:remaining_items]]
        additional_items = self.df.loc[top_indices]
        
        # Combine results
        recommendations = pd.concat([cluster_items, additional_items])
        return recommendations.head(n_recommendations)
    
    def analyze_temporal_patterns(self, date_column='date'):
        """Analyze temporal patterns in the data"""
        if self.df is None:
            raise ValueError("No data available. Call set_data() first.")
        
        # Extract years from dates
        years = []
        for date_str in self.df[date_column]:
            try:
                # Try to extract a year from the date string
                # This handles various date formats
                if not date_str or pd.isna(date_str):
                    continue
                    
                date_str = str(date_str)
                
                # Look for 4-digit year pattern
                import re
                year_match = re.search(r'\b(1\d{3}|20\d{2})\b', date_str)
                if year_match:
                    years.append(int(year_match.group(1)))
            except:
                continue
        
        # Count items by year
        year_counts = Counter(years)
        
        # Sort by year
        sorted_years = sorted(year_counts.items())
        
        # Plot the distribution
        if sorted_years:
            plt.figure(figsize=(12, 6))
            x, y = zip(*sorted_years)
            plt.bar(x, y)
            plt.title('Temporal Distribution of Items')
            plt.xlabel('Year')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            return plt
        else:
            print("No valid year data found for analysis")
            return None
    
    def get_topic_predictions(self, sample_size=100):
        """Get predicted topics for items based on text clustering"""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("No cluster data available. Call cluster_items() first.")
        
        # Sample the data if it's large
        if len(self.df) > sample_size:
            df_sample = self.df.sample(sample_size, random_state=42)
        else:
            df_sample = self.df
        
        # Get cluster info
        cluster_info = {}
        for cluster_id in df_sample['cluster'].unique():
            cluster_items = df_sample[df_sample['cluster'] == cluster_id]
            text_features = self.preprocess_text()
            top_terms = self._get_top_terms_for_cluster(cluster_id, text_features)
            
            cluster_info[cluster_id] = {
                'name': f"Topic {cluster_id}: {' '.join(top_terms[:3])}",
                'terms': top_terms,
                'items': len(cluster_items)
            }
        
        # Create a summary
        summary = pd.DataFrame({
            'cluster_id': list(cluster_info.keys()),
            'topic_name': [info['name'] for info in cluster_info.values()],
            'key_terms': [', '.join(info['terms']) for info in cluster_info.values()],
            'item_count': [info['items'] for info in cluster_info.values()]
        })
        
        return summary