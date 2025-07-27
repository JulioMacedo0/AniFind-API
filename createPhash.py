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
VIDEO_DIR = Path("test")
INDEX_PATH = Path("indexes/global_index.faiss")
METADATA_PATH = Path("indexes/metadata.pkl")
CHECKPOINT_DIR = Path("checkpoints")
WIDTH = 512
FPS = 6
PIX_FMT = 'rgb24'
USE_SCALE = True
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
    total_seconds = int(round(seconds))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02}"

def get_video_info(filepath):
    container = av.open(filepath)
    video_stream = next(s for s in container.streams if s.type == "video")
    duration = int(container.duration / av.time_base)
    width = video_stream.codec_context.width
    height = video_stream.codec_context.height
    container.close()
    return duration, width, height

def hashes_to_vector(ph, dh, ah):
    return np.concatenate([
        np.array(imagehash.hex_to_hash(ph).hash.flatten(), dtype=np.float32),
        np.array(imagehash.hex_to_hash(dh).hash.flatten(), dtype=np.float32),
        np.array(imagehash.hex_to_hash(ah).hash.flatten(), dtype=np.float32)
    ])

def build_ffmpeg_command(filepath, use_cuda):
    base_cmd = ["ffmpeg", "-loglevel", "quiet"]
    if use_cuda:
        base_cmd += ["-hwaccel", "cuda"]
    vf_filters = [f"fps={FPS}"]
    if USE_SCALE:
        vf_filters.append(f"scale={WIDTH}:-1")
    base_cmd += [
        "-i", str(filepath),
        "-vf", ",".join(vf_filters),
        "-f", "rawvideo",
        "-pix_fmt", PIX_FMT,
        "-"
    ]
    return base_cmd

def extract_hash_vectors(filepath):
    duration, orig_width, orig_height = get_video_info(filepath)
    print(f"[📽️] {filepath.name} | Duration: {duration}s | Resolution: {orig_width}x{orig_height}")

    info = extract_metadata_from_filename(filepath.stem)
    vectors, metadatas = [], []

    for try_cuda in [True, False]:
        cmd = build_ffmpeg_command(filepath, use_cuda=try_cuda)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        estimated_width = WIDTH if USE_SCALE else orig_width
        estimated_height = int(estimated_width * orig_height / orig_width)
        bpf = estimated_width * estimated_height * 3

        index = 0
        success = False
        start = time.time()
        while True:
            raw = process.stdout.read(bpf)
            if not raw or len(raw) < bpf:
                break
            try:
                img = Image.frombytes("RGB", (estimated_width, estimated_height), raw)
                if index == 0:
                    estimated_height = img.height
                    estimated_width = img.width
                    bpf = estimated_width * estimated_height * 3

                ph = str(imagehash.phash(img))
                dh = str(imagehash.dhash(img))
                ah = str(imagehash.average_hash(img))
                vectors.append(hashes_to_vector(ph, dh, ah))
                real_seconds = index / FPS
                metadatas.append({
                    **info,
                    "source_file": filepath.name,
                    "preview_source_path": str(filepath.resolve()),
                    "second": real_seconds,
                    "timecode": seconds_to_timecode(real_seconds),
                    "phash": ph,
                    "dhash": dh,
                    "ahash": ah
                })

                index += 1
                if index % 100 == 0:
                    print(f"[⏱️] {index} frames processed in {time.time() - start:.2f}s")
                success = True
            except Exception as e:
                print(f"[⚠️] Failed frame at {index}: {e}")
                break

        process.stdout.close()
        process.wait()

        if success:
            print(f"[✅] Total frames read: {index}")
            return vectors, metadatas

    raise RuntimeError("Failed to extract frames with or without CUDA")

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

        print(f"\n[📥] Processing {file.name}")
        try:
            start = time.time()
            vectors, metadatas = extract_hash_vectors(file)
            index.add(np.array(vectors, dtype=np.float32))
            metadata_all.extend(metadatas)

            faiss.write_index(index, str(INDEX_PATH))
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(metadata_all, f)

            mark_processed(file.name)
            elapsed = time.time() - start
            print(f"[🎯] {file.name} done | {len(vectors)} hashes | Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"[❌] Error processing {file.name}: {e}")

    print(f"\n[🏁] Finished. Total indexed entries: {len(metadata_all)}")

if __name__ == "__main__":
    main()
