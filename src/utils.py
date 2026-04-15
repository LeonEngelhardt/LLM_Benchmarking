import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
from typing import Optional

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
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/LeonEngelhardt/LLM_Benchmarking/main/"

    local_path = _resolve_local_image_path(path)
    if local_path is not None:
        return local_path

    if path.startswith("https://github.com/") and "/blob/" in path:
        return path.replace(
            "https://github.com/",
            "https://raw.githubusercontent.com/"
        ).replace("/blob/", "/")

    if path.startswith("data/") or path.startswith("/data/"):
        return GITHUB_RAW_BASE + path.lstrip("/")

    return path


def _resolve_local_image_path(path: str) -> Optional[str]:
    if not isinstance(path, str) or not path.strip():
        return None

    raw_path = path.strip()
    candidate = Path(raw_path)
    if candidate.exists():
        return str(candidate)

    normalized = raw_path.lstrip("/")
    candidate = Path(normalized)
    if candidate.exists():
        return str(candidate)

    # Some datasets use `data/image/...` while the repository stores files in
    # `data/images/...`.
    alias_candidates = []
    if normalized.startswith("data/image/"):
        alias_candidates.append(normalized.replace("data/image/", "data/images/", 1))
    if normalized.startswith("data/images/"):
        alias_candidates.append(normalized.replace("data/images/", "data/image/", 1))

    for alias in alias_candidates:
        alias_path = Path(alias)
        if alias_path.exists():
            return str(alias_path)

        # If the filename exists with a different extension, use that file.
        parent = alias_path.parent
        stem = alias_path.stem
        if parent.exists():
            matches = sorted(parent.glob(f"{stem}.*"))
            if matches:
                return str(matches[0])

    return None

def load_image(image_path: str) -> Image.Image:
    resolved_local_path = _resolve_local_image_path(image_path)
    if resolved_local_path is not None:
        return Image.open(resolved_local_path).convert("RGB")

    if image_path.startswith("http"):
        resp = requests.get(image_path)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(image_path).convert("RGB")
