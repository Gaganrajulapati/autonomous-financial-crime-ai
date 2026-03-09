import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\gagan\OneDrive\Desktop\financial_crime_ai\data\raw\PS_20174392719_1491204439457_log.csv")

print("Dataset Loaded Successfully")
print(data.head())
print(data.shape)

print("\nDataset Info\n")
print(data.info())

print("\nMissing Values\n")
print(data.isnull().sum())

print("\nFraud Distribution\n")
print(data['isFraud'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='isFraud', data=data)
plt.title("Fraud vs Normal Transactions")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x='type', data=data)
plt.title("Transaction Types Distribution")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(data['amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

fraud_by_type = data.groupby('type')['isFraud'].sum()

print(fraud_by_type)

fraud_by_type.plot(kind='bar', title="Fraud by Transaction Type")
plt.show()

top_senders = data['nameOrig'].value_counts().head(10)

print("\nTop 10 Sender Accounts")
print(top_senders)

top_receivers = data['nameDest'].value_counts().head(10)

print("\nTop 10 Receiver Accounts")
print(top_receivers)