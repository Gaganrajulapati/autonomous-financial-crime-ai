import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
from sklearn.ensemble import IsolationForest

st.set_page_config(layout="wide")

st.title("🚨 Financial Crime Investigation Dashboard")

# ---------------------------
# Load dataset
# ---------------------------

data = pd.read_csv(
    "data/raw/PS_20174392719_1491204439457_log.csv",
    low_memory=True
)

fraud = data[data["isFraud"] == 1]

# ---------------------------
# Sidebar Filters
# ---------------------------

st.sidebar.header("Filters")

transaction_types = st.sidebar.multiselect(
    "Transaction Type",
    options=data["type"].unique(),
    default=data["type"].unique()
)

amount_range = st.sidebar.slider(
    "Transaction Amount",
    int(data["amount"].min()),
    int(data["amount"].max()),
    (0, int(data["amount"].max()))
)

filtered_data = data[
    (data["type"].isin(transaction_types)) &
    (data["amount"].between(amount_range[0], amount_range[1]))
]

fraud_filtered = filtered_data[filtered_data["isFraud"] == 1]

# ---------------------------
# Top Metrics
# ---------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", len(filtered_data))
col2.metric("Fraud Transactions", len(fraud_filtered))
col3.metric("Fraud Rate", f"{(len(fraud_filtered)/len(filtered_data))*100:.4f}%")

st.divider()

# ---------------------------
# Fraud Transaction Types
# ---------------------------

st.subheader("Fraud Transaction Types")

type_counts = fraud_filtered["type"].value_counts().reset_index()
type_counts.columns = ["Transaction Type", "Fraud Count"]

fig1 = px.bar(
    type_counts,
    x="Transaction Type",
    y="Fraud Count",
    color="Transaction Type"
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------------------
# Fraud Amount Distribution
# ---------------------------

st.subheader("Fraud Amount Distribution")

fig2 = px.histogram(
    fraud_filtered,
    x="amount",
    nbins=50,
    color_discrete_sequence=["#FF6B6B"]
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Suspicious Accounts
# ---------------------------

st.subheader("Top Suspicious Accounts")

top_accounts = fraud_filtered["nameOrig"].value_counts().head(10)

fig3 = px.bar(
    x=top_accounts.index,
    y=top_accounts.values,
    labels={"x": "Account", "y": "Fraud Transactions"},
    color=top_accounts.values
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# Fraud Trend Over Time
# ---------------------------

st.subheader("Fraud Trend Over Time")

fraud_trend = fraud_filtered.groupby("step").size().reset_index(name="Fraud Count")

fig4 = px.line(
    fraud_trend,
    x="step",
    y="Fraud Count",
    markers=True
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# Fraud Money Flow Sankey
# ---------------------------

st.subheader("Fraud Money Flow")

sample = fraud_filtered.sample(min(200, len(fraud_filtered)))

nodes = list(set(sample["nameOrig"]).union(set(sample["nameDest"])))
node_index = {node: i for i, node in enumerate(nodes)}

source = sample["nameOrig"].map(node_index)
target = sample["nameDest"].map(node_index)
value = sample["amount"]

fig5 = go.Figure(go.Sankey(

    node=dict(
        pad=20,
        thickness=20,
        label=nodes,
        color="#A0C4FF"
    ),

    link=dict(
        source=source,
        target=target,
        value=value,
        color="rgba(255,107,107,0.4)"
    )

))

st.plotly_chart(fig5, use_container_width=True)

# ---------------------------
# AI Fraud Risk Detection
# ---------------------------

st.subheader("AI Fraud Risk Detection")

sample_ml = data.sample(50000, random_state=42)

features = sample_ml[[
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]].fillna(0)

model = IsolationForest(contamination=0.002, random_state=42)

sample_ml["anomaly"] = model.fit_predict(features)

suspicious = sample_ml[sample_ml["anomaly"] == -1]

st.write("⚠️ Suspicious Transactions Detected:", len(suspicious))

st.dataframe(suspicious.head(10))

# ---------------------------
# Fraud Alerts
# ---------------------------

st.subheader("🚨 Fraud Alerts")

alerts = fraud_filtered.sort_values("amount", ascending=False).head(5)

for _, row in alerts.iterrows():

    st.warning(
        f"Large Fraud Transaction\n\n"
        f"{row['nameOrig']} → {row['nameDest']}\n\n"
        f"Amount: ${row['amount']:,.2f}"
    )

# ---------------------------
# Fraud Network Investigation
# ---------------------------

st.subheader("Fraud Network Investigation")

sample_graph = fraud_filtered.sample(min(100, len(fraud_filtered)))

G = nx.from_pandas_edgelist(
    sample_graph,
    source="nameOrig",
    target="nameDest",
    edge_attr="amount"
)

net = Network(height="500px", width="100%", bgcolor="#111111", font_color="white")

for node in G.nodes():
    net.add_node(node, label=node)

for edge in G.edges(data=True):
    net.add_edge(edge[0], edge[1], value=edge[2]["amount"])

temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
net.save_graph(temp_file.name)

HtmlFile = open(temp_file.name, 'r', encoding='utf-8')
components.html(HtmlFile.read(), height=500)

# ---------------------------
# Account Investigation Tool
# ---------------------------

st.subheader("Account Investigation")

account = st.text_input("Enter Account ID")

if account:

    account_data = data[
        (data["nameOrig"] == account) |
        (data["nameDest"] == account)
    ]

    if len(account_data) > 0:

        fraud_count = account_data["isFraud"].sum()

        risk = "HIGH" if fraud_count > 0 else "LOW"

        st.success(f"Transactions Found: {len(account_data)}")
        st.info(f"Fraud Transactions: {fraud_count}")
        st.warning(f"Risk Level: {risk}")

        st.dataframe(account_data.head(50))

    else:

        st.error("Account not found in dataset")