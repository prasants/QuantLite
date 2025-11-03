"""Tests for quantlite.network module."""

import numpy as np
import pandas as pd
import pytest

from quantlite.network import (
    cascade_simulation,
    community_detection,
    correlation_network,
    eigenvector_centrality,
    network_summary,
)


@pytest.fixture()
def returns_df():
    """Generate correlated returns DataFrame."""
    rng = np.random.RandomState(42)
    n = 500
    common = rng.normal(0, 0.01, n)
    data = {}
    for name in ["A", "B", "C", "D"]:
        data[name] = common + rng.normal(0, 0.005, n)
    return pd.DataFrame(data)


@pytest.fixture()
def star_graph():
    """Star graph: node 0 connected to all others."""
    n = 5
    adj = np.zeros((n, n))
    for i in range(1, n):
        adj[0, i] = 1.0
        adj[i, 0] = 1.0
    return adj


@pytest.fixture()
def complete_graph():
    """Complete graph with 4 nodes."""
    n = 4
    adj = np.ones((n, n)) - np.eye(n)
    return adj


class TestCorrelationNetwork:
    """Tests for correlation network construction."""

    def test_returns_structure(self, returns_df):
        result = correlation_network(returns_df)
        assert "adjacency_matrix" in result
        assert "edges" in result
        assert "nodes" in result

    def test_nodes_match_columns(self, returns_df):
        result = correlation_network(returns_df)
        assert result["nodes"] == list(returns_df.columns)

    def test_threshold_filters_edges(self, returns_df):
        high = correlation_network(returns_df, threshold=0.9)
        low = correlation_network(returns_df, threshold=0.1)
        assert len(low["edges"]) >= len(high["edges"])

    def test_symmetric_adjacency(self, returns_df):
        result = correlation_network(returns_df)
        adj = result["adjacency_matrix"]
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_no_self_edges(self, returns_df):
        result = correlation_network(returns_df, threshold=0.0)
        adj = result["adjacency_matrix"]
        for i in range(adj.shape[0]):
            assert adj[i, i] == 0.0


class TestEigenvectorCentrality:
    """Tests for eigenvector centrality."""

    def test_star_graph_centre_highest(self, star_graph):
        cent = eigenvector_centrality(star_graph)
        assert cent[0] == pytest.approx(max(cent))

    def test_complete_graph_uniform(self, complete_graph):
        cent = eigenvector_centrality(complete_graph)
        # All nodes should have equal centrality
        np.testing.assert_array_almost_equal(cent, np.ones(4) / 4, decimal=3)

    def test_sums_to_one(self, star_graph):
        cent = eigenvector_centrality(star_graph)
        assert sum(cent) == pytest.approx(1.0, abs=1e-6)

    def test_empty_graph(self):
        cent = eigenvector_centrality(np.array([]).reshape(0, 0))
        assert len(cent) == 0


class TestCascadeSimulation:
    """Tests for cascade simulation."""

    def test_initial_shock(self, complete_graph):
        result = cascade_simulation(complete_graph, shock_node=0)
        assert result["per_round"][0][0] == -0.5

    def test_propagation(self, complete_graph):
        result = cascade_simulation(complete_graph, shock_node=0)
        # After round 1, other nodes should be affected
        assert result["final_state"][1] != 0.0

    def test_shock_magnitude(self, star_graph):
        result = cascade_simulation(star_graph, shock_node=0, shock_magnitude=-1.0)
        assert result["final_state"][0] <= -1.0

    def test_isolated_node_no_spread(self):
        """Shock to isolated node should not propagate."""
        adj = np.zeros((3, 3))
        result = cascade_simulation(adj, shock_node=0)
        assert result["final_state"][1] == 0.0
        assert result["final_state"][2] == 0.0


class TestCommunityDetection:
    """Tests for community detection."""

    def test_block_diagonal_two_communities(self):
        """Two clearly separated communities."""
        adj = np.zeros((6, 6))
        # Community 1: nodes 0, 1, 2
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[i, j] = 1.0
        # Community 2: nodes 3, 4, 5
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    adj[i, j] = 1.0
        labels = community_detection(adj, n_communities=2)
        assert len(labels) == 6
        # Nodes in same community should have same label
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_single_node(self):
        labels = community_detection(np.array([[0.0]]))
        assert len(labels) == 1

    def test_auto_detect_communities(self):
        """Eigengap heuristic should find the right number."""
        adj = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[i, j] = 1.0
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    adj[i, j] = 1.0
        labels = community_detection(adj)
        unique = len(set(labels))
        assert unique >= 2


class TestNetworkSummary:
    """Tests for network summary."""

    def test_returns_all_keys(self, returns_df):
        result = network_summary(returns_df)
        assert "network" in result
        assert "centrality" in result
        assert "communities" in result
        assert "nodes" in result

    def test_centrality_length(self, returns_df):
        result = network_summary(returns_df)
        assert len(result["centrality"]) == len(returns_df.columns)
