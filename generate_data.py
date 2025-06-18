import cv2
import numpy as np
from PIL import Image
import imagehash
import os
import pickle
import time
from datetime import timedelta

# === CONFIGURAÇÕES ===
VIDEO_PATH = "OnePiece_S01E01.mp4"
OUTPUT_HASH_PATH = "phashes.npy"
OUTPUT_META_PATH = "metadata.pkl"
FRAME_POSITIONS = ['start', 'middle', 'end']

# === EXTRAI METADADOS DO NOME DO ARQUIVO ===
def parse_video_filename(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    
    # Handle the case where the format is "AnimeName_S01E01"
    if "_S" in name and "E" in name:
        # Split at the first underscore
        anime_name = name.split('_S')[0]
        # Extract season and episode numbers
        season_episode = name.split('_S')[1]
        # Find the position where 'E' starts
        e_pos = season_episode.find('E')
        
        if e_pos != -1:
            season_str = season_episode[:e_pos]
            episode_str = season_episode[e_pos+1:]
            season = int(season_str)
            episode = int(episode_str)
            return anime_name, season, episode
    
    # Fallback to original method (AnimeNome_S01_E01)
    parts = name.split("_")
    if len(parts) >= 3:
        anime = parts[0]
        season = int(parts[1][1:])  # S01 -> 1
        episode = int(parts[2][1:])  # E01 -> 1
        return anime, season, episode
    
    raise ValueError(f"Could not parse filename format: {name}")

# === CONVERTE FRAME EM VETOR USANDO pHash ===
def compute_phash_vector(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Converte para RGB
    phash = imagehash.phash(img)
    return np.array(phash.hash, dtype=np.float32).flatten()  # 64-dim (8x8)

# === PROCESSA O VÍDEO ===
def process_video(video_path, start_id=0):
    start_time = time.time()
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
    print(f"Total de frames no vídeo: {total_frames} | FPS: {fps:.2f}")
    
    progress_steps = max(1, duration // 20)  # Mostra progresso a cada 5%
    frames_processed = 0
    total_expected_frames = duration * len(FRAME_POSITIONS)
    
    for sec in range(duration):
        if sec % progress_steps == 0 or sec == duration - 1:
            elapsed = time.time() - start_time
            percent_done = (sec / duration) * 100
            eta = (elapsed / max(1, sec)) * (duration - sec) if sec > 0 else 0
            print(f"Progresso: {percent_done:.1f}% | Tempo decorrido: {timedelta(seconds=int(elapsed))} | ETA: {timedelta(seconds=int(eta))}")
        
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
                print(f"Aviso: Não foi possível ler o frame em {sec}s, posição '{pos}'")
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
            frames_processed += 1

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
