import re
from pathlib import Path
import shutil

VIDEO_DIR = Path("test")
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi"]

def clean_filename(name: str):

    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"[()\[\]]", "", name)  
    return name.strip()

def extract_metadata(filename: str):
    cleaned = clean_filename(filename)

    patterns = [
        r"(?P<anime>.+?)\s*[Ss](?P<season>\d+)[Ee](?P<episode>\d+)",
        r"(?P<anime>.+?)\s*(?P<season>\d+)x(?P<episode>\d+)",
        r"(?P<anime>.+?)\s+(?P<season>\d+)[\.\s](?P<episode>\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            data = match.groupdict()
            anime = data["anime"].strip()
            anime = re.sub(r"[-–—]?\s*\d+$", "", anime)  # Remove final 001, 002...
            anime = anime.replace("_", " ").replace(".", " ").strip()
            season = int(data["season"])
            episode = int(data["episode"])
            normalized = f"{anime} - S{season:02}E{episode:02}"
            return normalized
    return None

def normalize_files():
    for file in VIDEO_DIR.rglob("*"):
        if file.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        base = file.stem
        normalized = extract_metadata(base)

        if normalized:
            new_name = f"{normalized}{file.suffix}"
            new_path = file.with_name(new_name)
            if new_path != file:
                print(f"✅ Renaming: {file.name} → {new_name}")
                shutil.move(str(file), str(new_path))
        else:
            print(f"⚠️  Could not normalize: {file.name}")

if __name__ == "__main__":
    normalize_files()
