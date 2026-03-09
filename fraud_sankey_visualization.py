import pandas as pd
import plotly.graph_objects as go

# Load dataset
data = pd.read_csv("data/raw/PS_20174392719_1491204439457_log.csv")

print("Dataset Loaded")

# Filter fraud transactions
fraud = data[data["isFraud"] == 1].sample(400, random_state=42)

# Create flow stages
stage1 = "Source Accounts"
stage4 = "Destination Accounts"

stage2 = fraud["type"]
stage3 = fraud["nameDest"]

# Create node list
nodes = [stage1] + list(stage2.unique()) + list(stage3.unique()) + [stage4]

node_index = {node: i for i, node in enumerate(nodes)}

source = []
target = []
value = []

for _, row in fraud.iterrows():

    # Stage 1 → Stage 2
    source.append(node_index[stage1])
    target.append(node_index[row["type"]])
    value.append(row["amount"])

    # Stage 2 → Stage 3
    source.append(node_index[row["type"]])
    target.append(node_index[row["nameDest"]])
    value.append(row["amount"])

    # Stage 3 → Stage 4
    source.append(node_index[row["nameDest"]])
    target.append(node_index[stage4])
    value.append(row["amount"])

# Node colors
node_colors = []

for node in nodes:
    if node == stage1:
        node_colors.append("#FF6B6B")
    elif node == stage4:
        node_colors.append("#4D96FF")
    elif node in stage2.unique():
        node_colors.append("#FFD166")
    else:
        node_colors.append("#A0C4FF")

# Create Sankey
fig = go.Figure(go.Sankey(
    node=dict(
        pad=25,
        thickness=30,
        line=dict(color="black", width=0.5),
        label=nodes,
        color=node_colors
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color="rgba(255,107,107,0.35)"
    )
))

fig.update_layout(
    title="Financial Crime Investigation Flow",
    font_size=13,
    height=700
)

fig.write_html("fraud_money_flow_premium.html")

print("Premium Sankey visualization saved")