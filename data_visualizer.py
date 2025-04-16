import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import Counter
import seaborn as sns
import numpy as np
from IPython.display import display, HTML

class SmithsonianDataVisualizer:
    def __init__(self, df=None):
        self.df = df
        
    def set_data(self, df):
        """Set the DataFrame to visualize"""
        self.df = df
    
    def plot_data_sources(self):
        """Plot the distribution of data sources"""
        if self.df is None:
            raise ValueError("No data available. Call set_data() first.")
        
        source_counts = self.df['data_source'].value_counts()
        
        plt.figure(figsize=(10, 6))
        source_counts.plot(kind='bar')
        plt.title('Distribution of Data Sources')
        plt.xlabel('Data Source')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return plt
    
    def plot_object_types(self, top_n=10):
        """Plot the distribution of object types"""
        if self.df is None:
            raise ValueError("No data available. Call set_data() first.")
        
        # Parse the string representation of lists
        all_types = []
        for types_str in self.df['object_type']:
            try:
                types = eval(types_str)
                if isinstance(types, list):
                    all_types.extend(types)
                else:
                    all_types.append(types_str)
            except:
                all_types.append(types_str)
        
        type_counts = Counter(all_types)
        top_types = dict(type_counts.most_common(top_n))
        
        plt.figure(figsize=(12, 6))
        plt.bar(top_types.keys(), top_types.values())
        plt.title(f'Top {top_n} Object Types')
        plt.xlabel('Object Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return plt
    
    def create_network_graph(self, source_field='creators', target_field='topic', sample_size=50):
        """Create a network graph showing relationships between two fields"""
        if self.df is None:
            raise ValueError("No data available. Call set_data() first.")
        
        # Use a sample if the DataFrame is large
        if len(self.df) > sample_size:
            df_sample = self.df.sample(sample_size)
        else:
            df_sample = self.df
        
        # Create a graph
        G = nx.Graph()
        
        # Add edges between source and target fields
        for _, row in df_sample.iterrows():
            source_val = row[source_field]
            target_val = row[target_field]
            
            # Parse string representation of lists
            try:
                source_items = eval(source_val) if isinstance(source_val, str) else [source_val]
                target_items = eval(target_val) if isinstance(target_val, str) else [target_val]
                
                if isinstance(source_items, list) and isinstance(target_items, list):
                    for s in source_items:
                        for t in target_items:
                            if s and t:  # Only add if both values exist
                                G.add_edge(str(s), str(t))
            except:
                # If parsing fails, use the original strings
                if source_val and target_val:
                    G.add_edge(str(source_val), str(target_val))
        
        # Draw the graph
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with different colors for source and target
        source_nodes = set([n for n, d in G.nodes(data=True) if n in df_sample[source_field].astype(str).values])
        target_nodes = set(G.nodes()) - source_nodes
        
        nx.draw_networkx_nodes(G, pos, nodelist=list(source_nodes), node_color='skyblue', node_size=100, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=list(target_nodes), node_color='lightgreen', node_size=100, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
        
        # Limit the number of labels to avoid overcrowding
        if len(G.nodes()) > 30:
            # Only label nodes with higher degree centrality
            centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in sorted_nodes[:20]]
            labels = {node: node for node in top_nodes}
        else:
            labels = {node: node for node in G.nodes()}
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title(f'Network Graph: {source_field} to {target_field} Relationships')
        plt.axis('off')
        plt.tight_layout()
        return plt
    
    def create_interactive_dashboard(self):
        """Create a simple HTML dashboard for the data"""
        if self.df is None:
            raise ValueError("No data available. Call set_data() first.")
        
        # Sample data for display
        sample_df = self.df.head(10)
        
        # Count data sources
        data_sources = self.df['data_source'].value_counts().to_dict()
        
        # Get statistics on topics
        topics = []
        for topics_str in self.df['topic']:
            try:
                topics_list = eval(topics_str)
                if isinstance(topics_list, list):
                    topics.extend(topics_list)
                else:
                    topics.append(topics_str)
            except:
                topics.append(topics_str)
        
        topic_counts = Counter(topics)
        top_topics = dict(topic_counts.most_common(5))
        
        # Create HTML for dashboard
        html = f"""
        <div style="font-family: Arial; max-width: 1200px; margin: 0 auto;">
            <h1>Smithsonian Linked Data Explorer</h1>
            <p>Exploring connections in Smithsonian Open Access data</p>
            
            <h2>Dataset Summary</h2>
            <p>Total Items: {len(self.df)}</p>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <div style="width: 48%; background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h3>Data Sources</h3>
                    <ul>
        """
        
        for source, count in data_sources.items():
            html += f"<li>{source}: {count}</li>"
        
        html += """
                    </ul>
                </div>
                <div style="width: 48%; background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h3>Top Topics</h3>
                    <ul>
        """
        
        for topic, count in top_topics.items():
            html += f"<li>{topic}: {count}</li>"
        
        html += """
                    </ul>
                </div>
            </div>
            
            <h2>Sample Items</h2>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f2f2f2;">
                        <th style="padding: 12px; text-align: left;">Title</th>
                        <th style="padding: 12px; text-align: left;">Type</th>
                        <th style="padding: 12px; text-align: left;">Date</th>
                        <th style="padding: 12px; text-align: left;">Source</th>
                    </tr>
        """
        
        for _, row in sample_df.iterrows():
            html += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 12px;">{row['title']}</td>
                        <td style="padding: 12px;">{row['object_type']}</td>
                        <td style="padding: 12px;">{row['date']}</td>
                        <td style="padding: 12px;">{row['data_source']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <h2>Linked Data Structure</h2>
            <p>This explorer transforms Smithsonian data into JSON-LD format, creating semantic relationships between:</p>
            <ul>
                <li>Creators and their works</li>
                <li>Objects and their types</li>
                <li>Topics and related items</li>
                <li>Time periods and collections</li>
            </ul>
            
            <div style="margin-top: 30px; text-align: center; color: #666;">
                <p>Smithsonian Linked Data Explorer - Created with Python and the Smithsonian Open Access API</p>
            </div>
        </div>
        """
        
        return HTML(html)