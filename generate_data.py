import cv2
import numpy as np
from PIL import Image
import imagehash
import os
import pickle

# === CONFIGURAÇÕES ===
VIDEO_PATH = "OnePiece_S01E01.mp4"
OUTPUT_HASH_PATH = "phashes.npy"
OUTPUT_META_PATH = "metadata.pkl"
FRAME_POSITIONS = ['start', 'middle', 'end']

# === EXTRAI METADADOS DO NOME DO ARQUIVO ===
def parse_video_filename(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    anime = parts[0]
    season = int(parts[1][1:])  # S01 -> 1
    episode = int(parts[2][1:])  # E01 -> 1
    return anime, season, episode

# === CONVERTE FRAME EM VETOR USANDO pHash ===
def compute_phash_vector(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Converte para RGB
    phash = imagehash.phash(img)
    return np.array(phash.hash, dtype=np.float32).flatten()  # 64-dim (8x8)

# === PROCESSA O VÍDEO ===
def process_video(video_path, start_id=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Erro ao abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames // fps)

    anime, season, episode = parse_video_filename(video_path)

    vectors = []
    metadata = {}
    current_id = start_id

    print(f"Processando {anime} S{season}E{episode} ({duration}s)")

    for sec in range(duration):
        for pos in FRAME_POSITIONS:
            # Cálculo do índice do frame
            if pos == 'start':
                frame_index = int(sec * fps)
            elif pos == 'middle':
                frame_index = int((sec + 0.5) * fps)
            elif pos == 'end':
                frame_index = int((sec + 0.99) * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            # Usa o frame original (sem resize)
            vector = compute_phash_vector(frame)

            vectors.append(vector)
            metadata[current_id] = {
                "anime": anime,
                "season": season,
                "episode": episode,
                "second": sec,
                "position": pos
            }
            current_id += 1

    cap.release()
    return vectors, metadata

# === MAIN ===
def main():
    vectors, metadata = process_video(VIDEO_PATH)

    # Salva os vetores
    np.save(OUTPUT_HASH_PATH, np.array(vectors, dtype=np.float32))

    # Salva os metadados
    with open(OUTPUT_META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Salvos {len(vectors)} vetores e metadados em memória.")

if __name__ == "__main__":
    main()
