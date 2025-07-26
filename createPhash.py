# === indexador.py ===
import subprocess
import imagehash
from PIL import Image
import numpy as np
import time
import av
import pickle
import faiss
import re
from pathlib import Path

# === CONFIG ===
VIDEO_DIR = Path("test")
INDEX_PATH = Path("indexes/global_index.faiss")
METADATA_PATH = Path("indexes/metadata.pkl")
CHECKPOINT_DIR = Path("checkpoints")
WIDTH = 512
FPS = 1
PIX_FMT = 'rgb24'

CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
INDEX_PATH.parent.mkdir(exist_ok=True, parents=True)

# === UTILS ===
def extract_metadata_from_filename(filename):
    patterns = [
        r"(?P<anime>.+?)\s*[Ss](?P<season>\d+)[Ee](?P<episode>\d+)",
        r"(?P<anime>.+?)\s*(?P<season>\d+)x(?P<episode>\d+)",
        r"(?P<anime>.+?)\s*[\[\(]?(?P<season>\d+)[\.\s](?P<episode>\d+)[\)\]]?"
    ]
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            data = match.groupdict()
            return {
                "anime": data["anime"].strip().replace("_", " "),
                "season": int(data["season"]),
                "episode": int(data["episode"])
            }
    return {"anime": filename, "season": 0, "episode": 0}

def seconds_to_timecode(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"

def get_duration(video_path):
    container = av.open(video_path)
    duration = int(container.duration / av.time_base)
    container.close()
    return duration

def phash_to_vector(phash_str):
    return np.array(imagehash.hex_to_hash(phash_str).hash.flatten(), dtype=np.float32)

def extract_phashes(filepath):
    duration = get_duration(filepath)
    cmd = [
        "ffmpeg", "-loglevel", "quiet", "-hwaccel", "cuda", "-i", str(filepath),
        "-vf", f"fps={FPS},scale={WIDTH}:-1", "-f", "rawvideo", "-pix_fmt", PIX_FMT, "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    estimated_height = int(WIDTH * 9 / 16)
    bpf = WIDTH * estimated_height * 3

    index = 0
    vectors, metadatas = [], []
    info = extract_metadata_from_filename(filepath.stem)

    while True:
        raw = process.stdout.read(bpf)
        if not raw or len(raw) < bpf:
            break

        img = Image.frombytes("RGB", (WIDTH, estimated_height), raw)
        if index == 0:
            estimated_height = img.height
            bpf = WIDTH * estimated_height * 3

        phash_str = str(imagehash.phash(img))
        vectors.append(phash_to_vector(phash_str))
        metadatas.append({
            **info,
            "source_file": filepath.name,
            "second": index,
            "timecode": seconds_to_timecode(index),
            "phash": phash_str
        })
        index += 1

    process.stdout.close()
    process.wait()
    return vectors, metadatas

def is_processed(relative_path):
    checkpoint_file = CHECKPOINT_DIR / (relative_path.as_posix().replace("/", "__") + ".done")
    return checkpoint_file.exists()

def mark_processed(relative_path):
    checkpoint_file = CHECKPOINT_DIR / (relative_path.as_posix().replace("/", "__") + ".done")
    checkpoint_file.touch()

def main():
    vectors_all, metadata_all = [], []
    video_files = [f for f in VIDEO_DIR.rglob("*") if f.suffix.lower() in [".mkv", ".mp4"]]

    for file in video_files:
        rel_path = file.relative_to(VIDEO_DIR)
        if is_processed(rel_path):
            print(f"[✔] Skipping {rel_path} (already processed)")
            continue

        print(f"[📥] Processing {rel_path}")
        vectors, metadatas = extract_phashes(file)
        vectors_all.extend(vectors)
        metadata_all.extend(metadatas)
        mark_processed(rel_path)

    if not vectors_all:
        print("[⚠] No new vectors to add.")
        return

    print(f"[➕] Adding {len(vectors_all)} vectors to FAISS...")
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatL2(64)

    index.add(np.array(vectors_all, dtype=np.float32))
    faiss.write_index(index, str(INDEX_PATH))

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "rb") as f:
            existing = pickle.load(f)
    else:
        existing = []

    existing.extend(metadata_all)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(existing, f)

    print(f"[✅] Database updated. Total entries: {len(existing)}")

if __name__ == "__main__":
    main()
