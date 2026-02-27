import pandas as pd
from PIL import Image
import requests
from io import BytesIO

def load_csv(path):
    try:
        # sep=None forces Pandas to automatically detect if the file uses ',' or ';'
        return pd.read_csv(
            path, 
            sep=None, 
            engine='python'
        )
    except UnicodeDecodeError:
        # If it's saved in Excel's weird Windows format, catch it and load it anyway
        print(f"  -> [WARNING] Encoding issue detected in {path}. Using Windows-1252 fallback.")
        return pd.read_csv(
            path, 
            sep=None, 
            engine='python',
            encoding='windows-1252'
        )

def save_csv(df, path: str):
    df.to_csv(path, index=False)

def normalize_image_path(path: str) -> str:
    if path.startswith("https://github.com/") and "/blob/" in path:
        return path.replace(
            "https://github.com/",
            "https://raw.githubusercontent.com/"
        ).replace("/blob/", "/")
    return path

def load_image(image_path: str) -> Image.Image:
    if image_path.startswith("http"):
        resp = requests.get(image_path)
        return Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(image_path).convert("RGB")