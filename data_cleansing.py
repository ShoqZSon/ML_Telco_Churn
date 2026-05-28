import pandas as pd
import os

def clean_data(df, path, csv_file):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    for col in binary_cols:
        unique_vals = df[col].unique()
        df[col] = df[col].map({unique_vals[0]: 0, unique_vals[1]: 1})
    
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)
    
    df.to_csv(os.path.join(path, csv_file), index=False)
    
    return df