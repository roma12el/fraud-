import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data(path='data/dataset.csv'):
    df = pd.read_csv(path)
    return df

def basic_preprocess(df, drop_cols=None):
    df = df.copy()
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    # simple encoding for categorical vars
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # leave contract_id out
    cat_cols = [c for c in cat_cols if c!='contract_id']
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df
