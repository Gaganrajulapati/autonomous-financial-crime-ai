import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

data = pd.read_csv("data/raw/PS_20174392719_1491204439457_log.csv")

print("Dataset Loaded")
print(data.shape)

features = data[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
features = features.fillna(0)

model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

data['anomaly'] = model.fit_predict(features)

suspicious_transactions = data[data['anomaly'] == -1]

print("Total Suspicious Transactions:", len(suspicious_transactions))