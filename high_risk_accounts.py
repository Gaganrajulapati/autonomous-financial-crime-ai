import pandas as pd
import networkx as nx

data = pd.read_csv("data/raw/PS_20174392719_1491204439457_log.csv")

print("Dataset Loaded")

fraud_data = data[data['isFraud'] == 1]

G = nx.from_pandas_edgelist(
    fraud_data,
    source="nameOrig",
    target="nameDest",
    edge_attr="amount"
)

degree_centrality = nx.degree_centrality(G)

degree_df = pd.DataFrame(
    degree_centrality.items(),
    columns=["account","degree_score"]
)

betweenness = nx.betweenness_centrality(G)

betweenness_df = pd.DataFrame(
    betweenness.items(),
    columns=["account","betweenness_score"]
)

pagerank = nx.pagerank(G)

pagerank_df = pd.DataFrame(
    pagerank.items(),
    columns=["account","pagerank_score"]
)

risk_df = degree_df.merge(betweenness_df,on="account")
risk_df = risk_df.merge(pagerank_df,on="account")

top_risk_accounts = risk_df.sort_values(
    by="pagerank_score",
    ascending=False
).head(10)

print("\nTop High-Risk Accounts\n")
print(top_risk_accounts)