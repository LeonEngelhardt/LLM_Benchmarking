import pandas as pd

def load_csv(path: str):
    return pd.read_csv(path)

def save_csv(df, path: str):
    df.to_csv(path, index=False)

def normalize_image_path(path: str) -> str:
    if path.startswith("https://github.com/") and "/blob/" in path:
        return path.replace(
            "https://github.com/",
            "https://raw.githubusercontent.com/"
        ).replace("/blob/", "/")
    return path