import pandas as pd
import networkx as nx
from pyvis.network import Network

data = pd.read_csv("data/raw/PS_20174392719_1491204439457_log.csv")

print("Dataset Loaded")
print(data.shape)

# filter fraud transactions
data = data[data['isFraud'] == 1]

# create graph
G = nx.from_pandas_edgelist(
    data,
    source="nameOrig",
    target="nameDest",
    edge_attr="amount"
)

print("Total Nodes:", G.number_of_nodes())
print("Total Edges:", G.number_of_edges())

# visualization
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

for node in G.nodes():
    net.add_node(node)

for edge in G.edges(data=True):
    net.add_edge(edge[0], edge[1], value=edge[2]['amount'])

# save HTML file
net.write_html("fraud_network.html")

print("Graph saved as fraud_network.html")