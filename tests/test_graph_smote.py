"""
Tests for Graph SMOTE implementation
"""

import pytest
import networkx as nx
import numpy as np
from src.models.graph_smote import GraphSMOTE
from src.data.data_generator import generate_synthetic_data
from src.data.graph_builder import construct_claim_graph


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing"""
    X, y = generate_synthetic_data(n_samples=500, fraud_ratio=0.1)
    G = construct_claim_graph(X, y, k_neighbors=3)
    return G, X.columns.tolist()


def test_graph_smote_initialization():
    """Test GraphSMOTE initialization"""
    smote = GraphSMOTE(k_neighbors=5, sampling_strategy=0.5, random_state=42)
    
    assert smote.k_neighbors == 5
    assert smote.sampling_strategy == 0.5
    assert smote.random_state == 42


def test_graph_smote_increases_minority_class(sample_graph):
    """Test that Graph SMOTE increases minority class"""
    G, feature_cols = sample_graph
    
    # Count original minority nodes
    original_minority = sum([1 for node in G.nodes() if G.nodes[node]['label'] == 1])
    
    # Apply Graph SMOTE
    smote = GraphSMOTE(k_neighbors=5, sampling_strategy=0.8, random_state=42)
    G_resampled = smote.fit_resample(G, feature_cols)
    
    # Count new minority nodes
    new_minority = sum([1 for node in G_resampled.nodes() if G_resampled.nodes[node]['label'] == 1])
    
    assert new_minority > original_minority
    assert G_resampled.number_of_nodes() > G.number_of_nodes()


def test_graph_smote_preserves_graph_structure(sample_graph):
    """Test that Graph SMOTE maintains graph structure"""
    G, feature_cols = sample_graph
    
    smote = GraphSMOTE(k_neighbors=5, sampling_strategy=0.5, random_state=42)
    G_resampled = smote.fit_resample(G, feature_cols)
    
    # Check that graph is still connected (or has similar number of components)
    original_components = nx.number_connected_components(G)
    new_components = nx.number_connected_components(G_resampled)
    
    assert new_components >= original_components
    assert G_resampled.number_of_edges() > G.number_of_edges()


def test_synthetic_nodes_have_correct_attributes(sample_graph):
    """Test that synthetic nodes have all required attributes"""
    G, feature_cols = sample_graph
    
    smote = GraphSMOTE(k_neighbors=5, sampling_strategy=0.5, random_state=42)
    G_resampled = smote.fit_resample(G, feature_cols)
    
    # Find synthetic nodes
    synthetic_nodes = [node for node in G_resampled.nodes() 
                      if G_resampled.nodes[node].get('is_synthetic', False)]
    
    if synthetic_nodes:  # If any synthetic nodes were created
        for node in synthetic_nodes:
            # Check label
            assert G_resampled.nodes[node]['label'] == 1
            
            # Check all features present
            for col in feature_cols:
                assert col in G_resampled.nodes[node]
