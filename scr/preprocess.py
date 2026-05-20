import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/raw/telco.csv')

df = df.dropna()

encoder = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = encoder.fit_transform(df[col])

df.to_csv('data/processed/processed.csv', index=False)