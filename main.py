import kagglehub
import numpy as np
import pandas as pd
import os

os.makedirs('output/plots/descriptive', exist_ok=True)
os.makedirs('output/plots/explorative', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

path = kagglehub.dataset_download("blastchar/telco-customer-churn", output_dir='datasets')

# CSV-Datei im Ordner finden
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# Checking for invalid entries
print("===== Validierungsreport =====\n")

for col in df.columns[1:]:
    problems = []
    infos = []

    # NaN Werte
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        problems.append(f"NaN Werte: {nan_count}")

    # Leestrings
    if df[col].dtype == 'object':
        empty_count = (df[col] == ' ').sum()
        if empty_count > 0:
            problems.append(f"Leestrings: {empty_count}")

        # Kategorien als reine Info
        if df[col].nunique() < 10:
            infos.append(f"Kategorien: {df[col].unique().tolist()}")

    # Numerisch
    if df[col].dtype in ['int64', 'float64']:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            problems.append(f"Negative Werte: {neg_count}")

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        if outliers > 0:
            problems.append(f"Ausreißer (IQR): {outliers}")

    if problems:
        print(f"[!] {col}:")
        for p in problems:
            print(f"    -> {p}")
    else:
        print(f"[OK] {col}: keine Probleme")

    if infos:
        for i in infos:
            print(f"    [i] {i}")

    print()

print("\n===== Ende Report =====")

# Data Cleansing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Descriptive analysis - raw data
rows = []
cols = df.columns

for col in cols[1:]:
    nan_values = df[col].isna().sum()
    col_type = str(df[col].dtype)
    categorical_int = ['SeniorCitizen']

    if (df[col].dtype == 'int64' or df[col].dtype == 'float64') and col not in categorical_int:
        median = np.median(df[col]).round(2)
        mean = np.mean(df[col]).round(2)
        rows.append({
            'Column': col,
            'Type': col_type,
            'NaN Values': nan_values,
            'Median': median,
            'Mean': mean,
            'Value Distribution': ''
        })
    elif df[col].dtype == 'object' or col in categorical_int:
        distribution = df[col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
        distribution_str = ' | '.join([f"{k}: {v}" for k, v in distribution.items()])
        rows.append({
            'Column': col,
            'Type': col_type,
            'NaN Values': nan_values,
            'Median': '',
            'Mean': '',
            'Value Distribution': distribution_str
        })

summary_df = pd.DataFrame(rows)
os.makedirs('output', exist_ok=True)
summary_df.to_csv('output/descriptive_analysis.csv', index=False)
print(summary_df)

