# === createPhash.py ===
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
VIDEO_DIR = Path("D:/animes/solo") 
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

def hashes_to_vector(ph, dh, ah):
    return np.concatenate([
        np.array(imagehash.hex_to_hash(ph).hash.flatten(), dtype=np.float32),
        np.array(imagehash.hex_to_hash(dh).hash.flatten(), dtype=np.float32),
        np.array(imagehash.hex_to_hash(ah).hash.flatten(), dtype=np.float32)
    ])

def extract_hash_vectors(filepath):
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
    start = time.time()

    while True:
        raw = process.stdout.read(bpf)
        if not raw or len(raw) < bpf:
            break

        img = Image.frombytes("RGB", (WIDTH, estimated_height), raw)
        if index == 0:
            estimated_height = img.height
            bpf = WIDTH * estimated_height * 3

        ph = str(imagehash.phash(img))
        dh = str(imagehash.dhash(img))
        ah = str(imagehash.average_hash(img))
        vectors.append(hashes_to_vector(ph, dh, ah))
        metadatas.append({
            **info,
            "source_file": filepath.name,
            "second": index,
            "timecode": seconds_to_timecode(index),
            "phash": ph,
            "dhash": dh,
            "ahash": ah
        })

        index += 1
        if index % 100 == 0:
            elapsed = time.time() - start
            print(f"[⏳] {index} frames | {elapsed:.2f}s elapsed")

    process.stdout.close()
    process.wait()
    return vectors, metadatas

def is_processed(name):
    return (CHECKPOINT_DIR / f"{name}.done").exists()

def mark_processed(name):
    (CHECKPOINT_DIR / f"{name}.done").touch()

def main():
    metadata_all = []
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "rb") as f:
            metadata_all = pickle.load(f)

    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatL2(192)

    for file in VIDEO_DIR.rglob("*"):
        if file.suffix.lower() not in [".mkv", ".mp4"]:
            continue
        if is_processed(file.name):
            print(f"[✔] Skipping {file.name} (already processed)")
            continue

        print(f"[📥] Processing {file.name}")
        try:
            start = time.time()
            vectors, metadatas = extract_hash_vectors(file)
            index.add(np.array(vectors, dtype=np.float32))
            metadata_all.extend(metadatas)

            faiss.write_index(index, str(INDEX_PATH))
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(metadata_all, f)

            mark_processed(file.name)
            print(f"[✅] Added {len(vectors)} vectors from {file.name} in {time.time() - start:.2f}s\n")
        except Exception as e:
            print(f"[❌] Error with {file.name}: {e}")

    print(f"[🎉] Done! Total indexed entries: {len(metadata_all)}")

if __name__ == "__main__":
    main()
