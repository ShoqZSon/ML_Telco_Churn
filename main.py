import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("blastchar/telco-customer-churn", output_dir='Datasets')

# CSV-Datei im Ordner finden
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

print(df['Churn'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print(df.isnull().sum())