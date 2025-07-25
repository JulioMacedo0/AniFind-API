# -*- coding: utf-8 -*-
import os
import re
import time
import glob
import pickle
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import psutil
from PIL import Image
import imagehash

# === CONFIGURAÇÕES ===
BASE_DIR = "test"
OUTPUT_DIR = "database"
FRAME_POSITIONS = ['start', 'middle', 'end']
EXTENSIONS = ['.mp4', '.mkv', '.avi']
RESIZE_BEFORE_HASH = False  # NÃO redimensionar frames
MAX_PARALLEL_VIDEOS = 4
LOG_INTERVAL_SECONDS = 60

COMPILED_PATTERNS = [
    re.compile(r'(.+?)[_\s-][Ss](\d+)[Ee](\d+)'),
    re.compile(r'(.+?)[_\s-](\d+)x(\d+)'),
    re.compile(r'(.+?)[_\s]\[?(\d+)[.\s](\d+)\]?'),
    re.compile(r'\[.*?\]\s*(.+?)\s*-\s*(\d+)'),
    re.compile(r'(.+)[Ss](\d+)[Ee](\d+)')
]

def log(message, flush=False):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=flush)

def format_timedelta(seconds):
    return str(timedelta(seconds=int(seconds))) if seconds else "N/A"

def parse_video_path(filepath):
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(name)
        if match:
            groups = match.groups()
            anime = groups[0].strip().replace('_', ' ')
            if len(groups) == 3:
                return anime, int(groups[1]), int(groups[2])
            elif len(groups) == 2:
                return anime, 1, int(groups[1])
    raise ValueError(f"Não foi possível extrair metadados de: {filepath}")

def compute_phash_vector(frame):
    if frame is None:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    phash_obj = imagehash.phash(img)
    return np.packbits(phash_obj.hash.flatten()).view(np.uint64)[0]

def process_video(video_path):
    pid = os.getpid()
    video_name = os.path.basename(video_path)
    start_time = time.time()
    log(f"PID {pid} | ▶️  Iniciando processamento: {video_name}", flush=True)

    try:
        anime, season, episode = parse_video_path(video_path)
    except ValueError as e:
        log(f"PID {pid} | ⚠️  {e}", flush=True)
        return video_path, [], {}

    safe_name = re.sub(r'\W+', '_', anime)
    filename_base = f"{safe_name}_S{season:02d}E{episode:02d}"
    individual_dir = os.path.join(OUTPUT_DIR, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    hash_file = os.path.join(individual_dir, f"phashes_{filename_base}.npy")
    meta_file = os.path.join(individual_dir, f"metadata_{filename_base}.pkl")
    if os.path.exists(hash_file) and os.path.exists(meta_file):
        log(f"PID {pid} | ⏩ Ignorado: {filename_base} já existe", flush=True)
        return video_path, [], {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"PID {pid} | ❌ ERRO ao abrir vídeo", flush=True)
        return video_path, [], {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps) if fps > 0 else 0

    vectors, metadata = [], {}
    for sec in range(duration):
        if sec > 0 and sec % LOG_INTERVAL_SECONDS == 0:
            log(f"PID {pid} | ⏳ Progresso: {sec}/{duration}s", flush=True)

        for pos in FRAME_POSITIONS:
            offset = 0.1 if pos == 'start' else 0.5 if pos == 'middle' else 0.9
            idx = int((sec + offset) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                hash_val = compute_phash_vector(frame)
                if hash_val is not None:
                    frame_id = len(vectors)
                    vectors.append(hash_val)
                    metadata[frame_id] = {
                        'anime': anime, 'season': season, 'episode': episode,
                        'second': sec, 'position': pos, 'filepath': video_path
                    }
    cap.release()

    if vectors:
        np.save(hash_file, np.array(vectors, dtype=np.uint64))
        with open(meta_file, 'wb') as f:
            pickle.dump(metadata, f)
        log(f"PID {pid} | ✅ Salvo: {filename_base} ({len(vectors)} frames)", flush=True)
    else:
        log(f"PID {pid} | ⚠️ Nenhum hash gerado", flush=True)
    return video_path, vectors, metadata

def main():
    main_start = time.time()
    log("🚀 INICIANDO PROCESSAMENTO")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_videos = [f for ext in EXTENSIONS for f in glob.glob(os.path.join(BASE_DIR, '**', f'*{ext}'), recursive=True)]
    if not all_videos:
        log("❌ Nenhum vídeo encontrado.")
        return

    total_videos = len(all_videos)
    log(f"📦 {total_videos} vídeos encontrados.")
    processed = 0
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_VIDEOS) as executor:
        futures = {executor.submit(process_video, path): path for path in all_videos}
        for future in as_completed(futures):
            video_path, vectors, meta = future.result()
            processed += 1
            pct = (processed / total_videos) * 100
            log(f"📊 {processed}/{total_videos} ({pct:.2f}%) | {os.path.basename(video_path)}")

    log("✅ FIM DO PROCESSAMENTO")
    log(f"⏱️ Tempo total: {format_timedelta(time.time() - main_start)}")

if __name__ == "__main__":
    main()
