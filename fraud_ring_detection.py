import pandas as pd
import networkx as nx
from networkx.algorithms import community

data = pd.read_csv("data/raw/PS_20174392719_1491204439457_log.csv")

print("Dataset Loaded")

fraud_data = data[data['isFraud'] == 1]

G = nx.from_pandas_edgelist(
    fraud_data,
    source="nameOrig",
    target="nameDest",
    edge_attr="amount"
)

communities = community.greedy_modularity_communities(G)

print("Number of suspicious communities:", len(communities))

for i, community_group in enumerate(communities):
    print(f"\nFraud Ring {i+1}")
    print(list(community_group)[:10])  # show first 10 accounts
    
    