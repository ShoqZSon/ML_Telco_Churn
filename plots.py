from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs('output/plots/descriptive', exist_ok=True)
os.makedirs('output/plots/explorative', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

# CSV-Datei im Ordner finden
path = Path(__file__).parent / 'datasets' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(path)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

summary_df = pd.read_csv('output/descriptive_analysis.csv')
numeric_df = summary_df[summary_df['Median'].notna()].copy()
categorical_df = summary_df[summary_df['Value Distribution'].notna()].copy()

# Descriptive Analysis - Plots
churned = df[df['Churn'] == 'Yes']
plt.hist(churned['tenure'], bins=150)
plt.xlabel('Months as a Customer')
plt.xticks(range(0, df['tenure'].max(), 6))
plt.ylabel('Churn Counts')
plt.title('When do customers churn?')
plt.savefig('output/plots/descriptive/churn_tenure_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

bins = [0, 6, 12, 24, 48, 72]
labels = ['0-6 Months', '6-12 Months', '12-24 Months', '24-48 Months', '48+ Months']

churned['tenure_group'] = pd.cut(churned['tenure'], bins=bins, labels=labels)
print(churned['tenure_group'].value_counts(normalize=True).mul(100).round(2))

df.groupby('tenure')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).plot()
plt.xlabel('Tenure (Months)')
plt.xticks(range(0, df['tenure'].max(), 6))
plt.ylabel('Churn Rate %')
plt.title('Churn Rate per Customer Tenure')
plt.savefig('output/plots/descriptive/churn_tenure_line_chart.png', dpi=300, bbox_inches='tight')
plt.show()

summary_df = pd.read_csv('output/descriptive_analysis.csv')
df = pd.read_csv(os.path.join(path, csv_file))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Split into numeric and categorical
numeric_df = summary_df[summary_df['Median'].notna()].copy()
categorical_df = summary_df[summary_df['Value Distribution'].notna()].copy()

# ── 1. Median & Mean numerischer Features ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(numeric_df))
width = 0.35
ax.bar([i - width/2 for i in x], numeric_df['Median'].astype(float), width, label='Median')
ax.bar([i + width/2 for i in x], numeric_df['Mean'].astype(float), width, label='Mean')
ax.set_xticks(list(x))
ax.set_xticklabels(numeric_df['Column'], rotation=15)
ax.set_title('Median & Mean – Numerische Features')
ax.legend()
plt.tight_layout()
plt.savefig('output/plots/numeric_median_mean.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 2. Verteilung kategorischer Features ───────────────────────────────────
for _, row in categorical_df.iterrows():
    col_name = row['Column']
    distribution = row['Value Distribution']

    items = distribution.split(' | ')
    labels, values = [], []
    for item in items:
        k, v = item.split(': ')
        labels.append(k.strip())
        values.append(float(v.replace('%', '')))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=labels, y=values, ax=ax, palette='Blues_d')
    ax.set_title(f'Verteilung – {col_name}')
    ax.set_ylabel('Prozent %')
    ax.set_xlabel(col_name)
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, f'{v}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'output/plots/dist_{col_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ── 3. Churn-Anteil pro Feature ────────────────────────────────────────────
churn_rows = []
for col in df.columns[1:]:
    if col == 'Churn':
        continue
    if df[col].dtype == 'object' or col == 'SeniorCitizen':
        churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        churn_rate.columns = ['Category', 'Churn Rate']
        churn_rate['Feature'] = col
        churn_rows.append(churn_rate)

churn_df = pd.concat(churn_rows, ignore_index=True)

for feature, group in churn_df.groupby('Feature'):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=group, x='Category', y='Churn Rate', ax=ax, palette='Reds_d')
    ax.set_title(f'Churn Rate – {feature}')
    ax.set_ylabel('Churn Rate %')
    ax.set_xlabel(feature)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    for i, v in enumerate(group['Churn Rate']):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'output/plots/churn_{feature}.png', dpi=300, bbox_inches='tight')
    plt.show()