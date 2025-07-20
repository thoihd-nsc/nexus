import networkx as nx
import numpy as np
import pandas as pd
import pickle

# Parameters
num_vasps = 3000
m_hyperparameter = 5          # Edges to attach from a new node (controls hub formation)
target_edges = 75000
risk_hub_prob = 0.3           # Probability to force high-degree nodes as high-risk
cluster_edge_fraction = 0.1   # Fraction of edges explicitly added to form high-risk clusters

print("Step 1: Generating scale-free graph...")
G = nx.barabasi_albert_graph(num_vasps, m_hyperparameter)

print("Step 2: Assigning node features with structural risk...")
degrees = np.array([G.degree(n) for n in G.nodes()])

# Assign base risk levels with the desired imbalanced distribution
risk_levels = np.random.choice([0, 1, 2], size=num_vasps, p=[0.7, 0.2, 0.1]) # 0:Low, 1:Medium, 2:High

# Make hubs (high-degree nodes) more likely to be high-risk
high_degree_indices = np.where(degrees > np.percentile(degrees, 90))[0]
for node_idx in high_degree_indices:
    if np.random.rand() < risk_hub_prob:
        risk_levels[node_idx] = 2

node_features = {
    'risk_level': risk_levels,
    'transaction_volume': np.random.exponential(1000, size=num_vasps)
}
for key, data in node_features.items():
    nx.set_node_attributes(G, dict(enumerate(data)), key)

print("Step 3: Adding preferential edges to form risky clusters...")
high_risk_nodes = [n for n, attr in G.nodes(data=True) if attr['risk_level'] == 2]
if len(high_risk_nodes) > 1:
    cluster_edges_to_add = int(target_edges * cluster_edge_fraction)
    added_cluster_edges = 0
    attempts = 0
    max_attempts = cluster_edges_to_add * 10  # Defensive counter to prevent infinite loops

    while added_cluster_edges < cluster_edges_to_add and attempts < max_attempts:
        u, v = np.random.choice(high_risk_nodes, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v, fraud_ring_edge=1)
            added_cluster_edges += 1
        attempts += 1

while G.number_of_edges() < target_edges:
    u, v = np.random.randint(0, num_vasps, 2)
    if u != v and not G.has_edge(u, v):
        G.add_edge(u, v, fraud_ring_edge=0)

print("Step 4: Adding edge features and exporting files...")
for u, v in G.edges():
    if 'fraud_ring_edge' not in G.edges[u, v]:
        G.edges[u, v]['fraud_ring_edge'] = 0
    G.edges[u, v]['amount'] = np.random.uniform(100, 10000)
    G.edges[u, v]['timestamp'] = np.random.randint(0, 1000)

with open("vasp_graph.gpickle", "wb") as f:
    pickle.dump(G, f)
node_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
node_df.to_csv("vasp_nodes.csv", index=False)
edge_df = nx.to_pandas_edgelist(G)
edge_df.to_csv("vasp_edges.csv", index=False)

print("\n--- Dataset generation complete ---")
print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
