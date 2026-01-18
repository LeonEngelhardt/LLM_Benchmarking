import pandas as pd

def load_csv(path: str):
    return pd.read_csv(path)

def save_csv(df, path: str):
    df.to_csv(path, index=False)
