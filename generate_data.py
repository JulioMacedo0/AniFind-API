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
# Diretórios
BASE_DIR = "animes"
OUTPUT_DIR = "database"

# Nomes dos arquivos de saída
OUTPUT_HASH_PATH = os.path.join(OUTPUT_DIR, "phashes.npy")
OUTPUT_META_PATH = os.path.join(OUTPUT_DIR, "metadata.pkl")

# Parâmetros de processamento
FRAME_POSITIONS = ['start', 'middle', 'end']
EXTENSIONS = ['.mp4', '.mkv', '.avi']

# Otimizações de Hardware e Processamento
RESIZE_BEFORE_HASH = True
RESIZE_DIM = (64, 64)
MAX_PARALLEL_VIDEOS = 4
LOG_INTERVAL_SECONDS = 60 # NOVO: Intervalo para reportar o progresso de um vídeo (em segundos)

# Pré-compila os padrões de Regex para mais eficiência
COMPILED_PATTERNS = [
    re.compile(r'(.+?)[_\s-][Ss](\d+)[Ee](\d+)'),
    re.compile(r'(.+?)[_\s-](\d+)x(\d+)'),
    re.compile(r'(.+?)[_\s]\[?(\d+)[.\s](\d+)\]?'),
    re.compile(r'\[.*?\]\s*(.+?)\s*-\s*(\d+)')
]

# === FUNÇÕES DE LOG E UTILITÁRIAS ===
def log(message, flush=False): # MODIFICADO: Adicionado parâmetro 'flush'
    """Imprime uma mensagem de log com timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=flush) # MODIFICADO: Usa o flush

def format_timedelta(seconds):
    """Formata segundos para uma string H:M:S."""
    if seconds is None:
        return "N/A"
    return str(timedelta(seconds=int(seconds)))

def get_file_size_str(filepath):
    """Retorna o tamanho de um arquivo em formato legível (KB, MB, GB)."""
    if not os.path.exists(filepath):
        return "N/A"
    size_bytes = os.path.getsize(filepath)
    if size_bytes < 1024:
        return f"{size_bytes} Bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"

# === LÓGICA DE PROCESSAMENTO (EXECUTADA NOS WORKERS) ===
def parse_video_path(filepath):
    """Extrai metadados (anime, temporada, episódio) do caminho do arquivo."""
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(name)
        if match:
            groups = match.groups()
            anime = groups[0].strip().replace('_', ' ')
            if len(groups) == 3:
                season = int(groups[1])
                episode = int(groups[2])
                return anime, season, episode
            elif len(groups) == 2:
                season = 1
                episode = int(groups[1])
                return anime, season, episode

    parts = os.path.normpath(filepath).split(os.sep)
    if len(parts) >= 3:
        anime = parts[-3]
        season_dir = parts[-2]
        season_match = re.search(r'[Ss](?:eason)?\s*(\d+)', season_dir)
        season = int(season_match.group(1)) if season_match else 1
        ep_match = re.search(r'[Ee](?:pisode|p)?[\s._-]*(\d+)', name)
        if ep_match:
            episode = int(ep_match.group(1))
            return anime, season, episode

    raise ValueError(f"Não foi possível extrair metadados de: {filepath}")

def compute_phash_vector(frame):
    """Calcula o pHash de um frame e retorna como um inteiro de 64 bits."""
    if frame is None:
        return None
        
    if RESIZE_BEFORE_HASH:
        frame = cv2.resize(frame, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    phash_obj = imagehash.phash(img)
    
    return np.packbits(phash_obj.hash.flatten()).view(np.uint64)[0]

def process_video(video_path):
    """
    Função executada por cada worker. Processa um único vídeo do início ao fim.
    """
    pid = os.getpid()
    video_name = os.path.basename(video_path)
    start_time = time.time()
    
    # Usamos flush=True aqui para garantir que a mensagem de início apareça imediatamente
    log(f"PID {pid} | ▶️  Iniciando processamento de: {video_name}", flush=True) 

    try:
        anime, season, episode = parse_video_path(video_path)
    except ValueError as e:
        log(f"PID {pid} | ⚠️  AVISO: {e}", flush=True)
        return video_path, [], {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"PID {pid} | ❌ ERRO: Não foi possível abrir o vídeo {video_name}", flush=True)
        return video_path, [], {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps) if fps > 0 else 0
    
    if duration == 0:
        log(f"PID {pid} | ⚠️  AVISO: Vídeo sem duração ou corrompido: {video_name}", flush=True)
        cap.release()
        return video_path, [], {}

    log(f"PID {pid} | 🎬  Metadata: {anime} S{season:02d}E{episode:02d} ({duration}s)", flush=True)

    vectors = []
    metadata = {}
    
    for sec in range(duration):
        # NOVO: Bloco para logar o progresso dentro do vídeo
        if sec > 0 and sec % LOG_INTERVAL_SECONDS == 0:
            progress_percent = (sec / duration) * 100
            log(f"PID {pid} | ⏳ Progresso em '{video_name}': {sec}/{duration}s ({progress_percent:.1f}%)", flush=True)

        for pos_name in FRAME_POSITIONS:
            if pos_name == 'start': offset = 0.1
            elif pos_name == 'middle': offset = 0.5
            else: offset = 0.9
            
            frame_pos_idx = int((sec + offset) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos_idx)
            ret, frame = cap.read()

            if ret:
                phash_vector = compute_phash_vector(frame)
                if phash_vector is not None:
                    frame_id = len(vectors) 
                    vectors.append(phash_vector)
                    metadata[frame_id] = {
                        'anime': anime, 'season': season, 'episode': episode,
                        'second': sec, 'position': pos_name, 'filepath': video_path
                    }
    
    cap.release()
    
    total_time = time.time() - start_time
    frames_processed = len(vectors)
    fps_processed = frames_processed / total_time if total_time > 0 else 0
    
    log(f"PID {pid} | ✅  Concluído: {video_name} | {frames_processed} hashes gerados em {format_timedelta(total_time)} ({fps_processed:.2f} frames/s)", flush=True)
    
    return video_path, vectors, metadata

# ... (O restante do script, a partir da função main(), permanece o mesmo) ...

# === FUNÇÃO PRINCIPAL (ORQUESTRADOR) ===
def main():
    """Função principal que orquestra todo o processo."""
    main_start_time = time.time()
    log("🚀 INICIANDO GERAÇÃO DA BASE DE DADOS DE HASHES")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Encontrar todos os vídeos
    log("📁 Buscando arquivos de vídeo...")
    all_videos = [f for ext in EXTENSIONS for f in glob.glob(os.path.join(BASE_DIR, '**', f'*{ext}'), recursive=True)]
    
    if not all_videos:
        log("❌ Nenhum vídeo encontrado. Verifique o diretório `BASE_DIR` e as extensões.")
        return

    total_videos = len(all_videos)
    log(f"🔍 {total_videos} vídeos encontrados.")

    # 2. Processar vídeos em paralelo
    all_vectors = []
    all_metadata = {}
    videos_processed_count = 0
    current_global_id = 0

    log(f"🛠️  Iniciando pool de processamento com {MAX_PARALLEL_VIDEOS} workers...")
    
    process_times = []

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_VIDEOS) as executor:
        futures = {executor.submit(process_video, path): path for path in all_videos}
        
        for future in as_completed(futures):
            start_proc_time = time.time()
            
            processed_path, vectors, metadata = future.result()
            
            videos_processed_count += 1
            
            if vectors:
                all_vectors.extend(vectors)
                for local_id, meta_item in metadata.items():
                    all_metadata[current_global_id] = meta_item
                    current_global_id += 1
            
            end_proc_time = time.time()
            # Usamos o tempo que o processo principal levou para processar o resultado, que é rápido
            # mas ajuda a suavizar o ETA.
            process_times.append(end_proc_time - start_proc_time)
            avg_time_per_video = sum(process_times) / len(process_times)
            videos_remaining = total_videos - videos_processed_count
            eta_seconds = avg_time_per_video * videos_remaining
            
            progress = (videos_processed_count / total_videos) * 100
            
            # Este log principal continua como antes, mostrando o progresso geral
            log(f"📊 Progresso Geral: {videos_processed_count}/{total_videos} ({progress:.2f}%) | "
                f"Vídeo Recém-Concluído: {os.path.basename(processed_path)} | "
                f"ETA: {format_timedelta(eta_seconds)}")

    log("✅ Pool de processamento concluído.")

    # 3. Salvar os resultados
    log("💾 Salvando arquivos finais...")
    if all_vectors:
        phashes_array = np.array(all_vectors, dtype=np.uint64)
        np.save(OUTPUT_HASH_PATH, phashes_array)
        log(f"  -> Hashes salvos em: {OUTPUT_HASH_PATH} ({get_file_size_str(OUTPUT_HASH_PATH)})")

        with open(OUTPUT_META_PATH, 'wb') as f:
            pickle.dump(all_metadata, f)
        log(f"  -> Metadados salvos em: {OUTPUT_META_PATH} ({get_file_size_str(OUTPUT_META_PATH)})")
    else:
        log("⚠️ Nenhum vetor foi gerado. Nada a salvar.")

    # 4. Resumo final
    main_total_time = time.time() - main_start_time
    log("\n" + "="*50)
    log("🏁 RESUMO FINAL 🏁")
    log(f"⏱️  Tempo Total de Execução: {format_timedelta(main_total_time)}")
    log(f"📹 Vídeos Processados: {videos_processed_count}/{total_videos}")
    log(f"🔢 Total de Hashes Gerados: {len(all_vectors)}")
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 / 1024
    log(f"🧠 Pico de Memória (Processo Principal): {mem_usage:.2f} MB")
    log("="*50)


if __name__ == "__main__":
    main()