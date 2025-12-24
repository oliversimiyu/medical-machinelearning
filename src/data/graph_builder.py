"""
Graph construction from relational patterns in medical claims data
"""

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def construct_claim_graph(X, y, k_neighbors=5):
    """
    Construct a graph from insurance claims data.
    
    Graph structure:
    - Nodes: Each claim is a node
    - Edges: Connect claims that share:
      1. Same provider
      2. Similar patient demographics
      3. Similar claim characteristics
    - Node features: Claim attributes
    - Node labels: Fraud labels
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    k_neighbors : int
        Number of neighbors for KNN-based edges
    
    Returns:
    --------
    G : networkx.Graph
        Constructed graph
    """
    print("Constructing graph from relational patterns...")
    
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes (each claim is a node)
    for idx in range(len(X)):
        G.add_node(idx, **X.iloc[idx].to_dict(), label=int(y.iloc[idx]))
    
    # Strategy 1: Connect claims from same provider
    provider_groups = X.groupby('provider_claim_count').groups
    edge_count = 0
    
    for provider, indices in provider_groups.items():
        indices_list = list(indices)
        if len(indices_list) > 1 and len(indices_list) < 50:
            for i in range(len(indices_list)):
                for j in range(i+1, min(i+5, len(indices_list))):
                    G.add_edge(indices_list[i], indices_list[j], edge_type='same_provider')
                    edge_count += 1
    
    # Strategy 2: Connect similar claims (KNN on features)
    feature_subset = ['claim_amount', 'patient_age', 'num_procedures', 
                     'hospital_stay_days', 'diagnosis_complexity']
    
    X_subset = X[feature_subset].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree')
    nbrs.fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    
    for i in range(len(X)):
        for j in range(1, k_neighbors+1):
            neighbor_idx = indices[i][j]
            if not G.has_edge(i, neighbor_idx):
                G.add_edge(i, neighbor_idx, 
                          edge_type='similar_claim',
                          weight=1.0 / (distances[i][j] + 1e-6))
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
