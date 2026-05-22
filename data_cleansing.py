

df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
df.to_csv(os.path.join(path, csv_file), index=False)