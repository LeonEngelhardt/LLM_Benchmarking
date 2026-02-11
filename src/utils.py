import pandas as pd
from PIL import Image
import requests
from io import BytesIO

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

def load_image(image_path: str) -> Image.Image:
    if image_path.startswith("http"):
        resp = requests.get(image_path)
        return Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(image_path).convert("RGB")
