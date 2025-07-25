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
    
    # Timing: Conversão BGR para RGB
    convert_start = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    convert_time = time.time() - convert_start
    
    # Timing: Criação da imagem PIL
    pil_start = time.time()
    img = Image.fromarray(rgb_frame)
    pil_time = time.time() - pil_start
    
    # Timing: Cálculo do pHash
    phash_start = time.time()
    phash_obj = imagehash.phash(img)
    phash_time = time.time() - phash_start
    
    # Timing: Conversão para uint64
    pack_start = time.time()
    result = np.packbits(phash_obj.hash.flatten()).view(np.uint64)[0]
    pack_time = time.time() - pack_start
    
    # Retorna resultado e tempos para logging
    return result, {
        'convert_time': convert_time,
        'pil_time': pil_time,
        'phash_time': phash_time,
        'pack_time': pack_time
    }

def process_video(video_path):
    pid = os.getpid()
    video_name = os.path.basename(video_path)
    start_time = time.time()
    log(f"PID {pid} | ▶️ Iniciando processamento sequencial: {video_name}", flush=True)

    try:
        anime, season, episode = parse_video_path(video_path)
    except ValueError as e:
        log(f"PID {pid} | ⚠️ {e}", flush=True)
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
    if fps == 0:
        log(f"PID {pid} | ⚠️ FPS inválido", flush=True)
        return video_path, [], {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)

    # Gerar mapa de frames-alvo com precisão
    targets = {}
    for sec in range(duration):
        for pos in FRAME_POSITIONS:
            offset = 0.1 if pos == 'start' else 0.5 if pos == 'middle' else 0.9
            target_time = round(sec + offset, 2)
            targets[target_time] = (sec, pos)

    vectors = []
    metadata = {}
    frame_id = 0

    hash_time_total = 0
    convert_time_total = 0
    pil_time_total = 0
    phash_calc_time_total = 0

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = round(frame_index / fps, 2)
        if current_time in targets:
            sec, pos = targets[current_time]
            hash_start = time.time()
            result = compute_phash_vector(frame)
            hash_time_total += time.time() - hash_start

            if result and isinstance(result, tuple):
                h, breakdown = result
                vectors.append(h)
                metadata[frame_id] = {
                    'anime': anime, 'season': season, 'episode': episode,
                    'second': sec, 'position': pos, 'filepath': video_path
                }
                convert_time_total += breakdown['convert_time']
                pil_time_total += breakdown['pil_time']
                phash_calc_time_total += breakdown['phash_time']
                frame_id += 1

        frame_index += 1

    cap.release()

    if vectors:
        np.save(hash_file, np.array(vectors, dtype=np.uint64))
        with open(meta_file, 'wb') as f:
            pickle.dump(metadata, f)
        log(f"PID {pid} | ✅ Salvo: {filename_base} ({len(vectors)} frames)", flush=True)

    total_time = time.time() - start_time
    log(f"PID {pid} | ⏱️ Tempo total: {total_time:.2f}s | Frames válidos: {len(vectors)} | Média: {len(vectors)/total_time:.2f} f/s", flush=True)
    log(f"PID {pid} | 🔍 Hash breakdown: Convert {convert_time_total:.2f}s | PIL {pil_time_total:.2f}s | pHash {phash_calc_time_total:.2f}s", flush=True)

    return video_path, vectors, metadata

    pid = os.getpid()
    video_name = os.path.basename(video_path)
    start_time = time.time()
    log(f"PID {pid} | ▶️  Iniciando processamento: {video_name}", flush=True)

    # Timing: Parse metadados
    parse_start = time.time()
    try:
        anime, season, episode = parse_video_path(video_path)
        parse_time = time.time() - parse_start
        log(f"PID {pid} | ⏱️  Parse metadados: {parse_time:.3f}s", flush=True)
    except ValueError as e:
        log(f"PID {pid} | ⚠️  {e}", flush=True)
        return video_path, [], {}

    # Timing: Preparação de arquivos
    file_prep_start = time.time()
    safe_name = re.sub(r'\W+', '_', anime)
    filename_base = f"{safe_name}_S{season:02d}E{episode:02d}"
    individual_dir = os.path.join(OUTPUT_DIR, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    hash_file = os.path.join(individual_dir, f"phashes_{filename_base}.npy")
    meta_file = os.path.join(individual_dir, f"metadata_{filename_base}.pkl")
    file_prep_time = time.time() - file_prep_start
    
    # Timing: Verificação de arquivos existentes
    check_start = time.time()
    if os.path.exists(hash_file) and os.path.exists(meta_file):
        check_time = time.time() - check_start
        log(f"PID {pid} | ⏩ Ignorado: {filename_base} já existe (check: {check_time:.3f}s)", flush=True)
        return video_path, [], {}
    check_time = time.time() - check_start
    
    # Timing: Informações do arquivo
    file_info_start = time.time()
    video_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    file_info_time = time.time() - file_info_start
    
    log(f"PID {pid} | 📁 Prep arquivos: {file_prep_time:.3f}s | Check exist: {check_time:.3f}s | Info: {file_info_time:.3f}s", flush=True)
    log(f"PID {pid} | 📊 Tamanho do vídeo: {video_size:.2f} MB", flush=True)

    # Timing: Abertura do vídeo
    video_open_start = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        video_open_time = time.time() - video_open_start
        log(f"PID {pid} | ❌ ERRO ao abrir vídeo (tentativa: {video_open_time:.3f}s)", flush=True)
        return video_path, [], {}
    video_open_time = time.time() - video_open_start

    # Timing: Leitura de propriedades
    props_start = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps) if fps > 0 else 0
    props_time = time.time() - props_start
    
    log(f"PID {pid} | 🎬 Abertura: {video_open_time:.3f}s | Props: {props_time:.3f}s", flush=True)
    log(f"PID {pid} | 📈 FPS: {fps:.2f} | Frames: {total_frames} | Duração: {duration}s", flush=True)

    # Inicialização para timing detalhado
    vectors, metadata = [], {}
    frame_processing_start = time.time()
    
    # Contadores para estatísticas detalhadas
    total_seeks = 0
    total_reads = 0 
    total_hashes = 0
    seek_time_total = 0
    read_time_total = 0
    hash_time_total = 0
    
    # Contadores detalhados para sub-operações de hash
    convert_time_total = 0
    pil_time_total = 0
    phash_calc_time_total = 0
    pack_time_total = 0
    metadata_time_total = 0
    
    for sec in range(duration):
        if sec > 0 and sec % LOG_INTERVAL_SECONDS == 0:
            elapsed = time.time() - frame_processing_start
            frames_per_sec = len(vectors) / elapsed if elapsed > 0 else 0
            mb_per_sec = video_size / elapsed if elapsed > 0 else 0
            
            # Logs detalhados de progresso
            avg_seek_so_far = seek_time_total / total_seeks if total_seeks > 0 else 0
            avg_read_so_far = read_time_total / total_reads if total_reads > 0 else 0
            avg_hash_so_far = hash_time_total / total_hashes if total_hashes > 0 else 0
            
            log(f"PID {pid} | ⏳ Progresso: {sec}/{duration}s | Frames: {len(vectors)} | Taxa: {frames_per_sec:.1f} f/s | {mb_per_sec:.2f} MB/s", flush=True)
            log(f"PID {pid} | 🔄 Médias atuais: Seek {avg_seek_so_far*1000:.1f}ms | Read {avg_read_so_far*1000:.1f}ms | Hash {avg_hash_so_far*1000:.1f}ms", flush=True)

        for pos in FRAME_POSITIONS:
            offset = 0.1 if pos == 'start' else 0.5 if pos == 'middle' else 0.9
            idx = int((sec + offset) * fps)
            
            # Timing: Seek operation
            seek_start = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            seek_time = time.time() - seek_start
            seek_time_total += seek_time
            total_seeks += 1
            
            # Timing: Frame read
            read_start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - read_start
            read_time_total += read_time
            total_reads += 1
            
            if ret:
                # Timing: Hash computation (agora com breakdown detalhado)
                hash_start = time.time()
                hash_result = compute_phash_vector(frame)
                hash_time = time.time() - hash_start
                hash_time_total += hash_time
                total_hashes += 1
                
                if hash_result is not None:
                    if isinstance(hash_result, tuple) and len(hash_result) == 2:
                        hash_val, hash_breakdown = hash_result
                        
                        # Acumula tempos das sub-operações
                        convert_time_total += hash_breakdown['convert_time']
                        pil_time_total += hash_breakdown['pil_time']
                        phash_calc_time_total += hash_breakdown['phash_time']
                        pack_time_total += hash_breakdown['pack_time']
                    else:
                        # Compatibilidade com versão antiga
                        hash_val = hash_result
                        convert_time_total += 0
                        pil_time_total += 0
                        phash_calc_time_total += 0
                        pack_time_total += 0
                    
                    if hash_val is not None:
                        # Timing: Criação de metadados
                        metadata_start = time.time()
                        frame_id = len(vectors)
                        vectors.append(hash_val)
                        metadata[frame_id] = {
                            'anime': anime, 'season': season, 'episode': episode,
                            'second': sec, 'position': pos, 'filepath': video_path
                        }
                        metadata_time_total += time.time() - metadata_start
    
    frame_processing_time = time.time() - frame_processing_start
    cap.release()

    # Timing: Salvamento detalhado
    save_start = time.time()
    if vectors:
        # Timing: Conversão para array numpy
        array_start = time.time()
        hash_array = np.array(vectors, dtype=np.uint64)
        array_time = time.time() - array_start
        
        # Timing: Salvamento do arquivo numpy
        numpy_save_start = time.time()
        np.save(hash_file, hash_array)
        numpy_save_time = time.time() - numpy_save_start
        
        # Timing: Salvamento do pickle
        pickle_start = time.time()
        with open(meta_file, 'wb') as f:
            pickle.dump(metadata, f)
        pickle_time = time.time() - pickle_start
        
        log(f"PID {pid} | ✅ Salvo: {filename_base} ({len(vectors)} frames)", flush=True)
        log(f"PID {pid} | 💾 Salvamento: Array {array_time:.3f}s | Numpy {numpy_save_time:.3f}s | Pickle {pickle_time:.3f}s", flush=True)
    else:
        log(f"PID {pid} | ⚠️ Nenhum hash gerado", flush=True)
    save_time = time.time() - save_start
    
    # Cálculo de métricas finais
    total_time = time.time() - start_time
    mb_per_sec = video_size / total_time if total_time > 0 else 0
    frames_per_sec = len(vectors) / total_time if total_time > 0 else 0
    
    # Estatísticas médias detalhadas
    avg_seek = seek_time_total / total_seeks if total_seeks > 0 else 0
    avg_read = read_time_total / total_reads if total_reads > 0 else 0
    avg_hash = hash_time_total / total_hashes if total_hashes > 0 else 0
    
    # Médias das sub-operações de hash
    avg_convert = convert_time_total / total_hashes if total_hashes > 0 else 0
    avg_pil = pil_time_total / total_hashes if total_hashes > 0 else 0
    avg_phash_calc = phash_calc_time_total / total_hashes if total_hashes > 0 else 0
    avg_pack = pack_time_total / total_hashes if total_hashes > 0 else 0
    avg_metadata = metadata_time_total / len(vectors) if len(vectors) > 0 else 0
    
    # Log de performance SUPER detalhado
    log(f"PID {pid} | 🏁 FINALIZADO: {filename_base}", flush=True)
    log(f"PID {pid} | ⏱️  Tempo total: {total_time:.3f}s | Taxa: {mb_per_sec:.2f} MB/s | {frames_per_sec:.1f} frames/s", flush=True)
    log(f"PID {pid} | 📊 BREAKDOWN DETALHADO DO TEMPO:", flush=True)
    log(f"PID {pid} |   🎬 Processamento frames: {frame_processing_time:.3f}s ({frame_processing_time/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} |     ├─ Seeks: {seek_time_total:.3f}s ({seek_time_total/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} |     ├─ Reads: {read_time_total:.3f}s ({read_time_total/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} |     ├─ Hashes: {hash_time_total:.3f}s ({hash_time_total/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} |     │   ├─ BGR→RGB: {convert_time_total:.3f}s ({convert_time_total/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} |     │   ├─ PIL Image: {pil_time_total:.3f}s ({pil_time_total/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} |     │   ├─ pHash calc: {phash_calc_time_total:.3f}s ({phash_calc_time_total/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} |     │   └─ Pack bits: {pack_time_total:.3f}s ({pack_time_total/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} |     └─ Metadata: {metadata_time_total:.3f}s ({metadata_time_total/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} |   💾 Salvamento: {save_time:.3f}s ({save_time/total_time*100:.1f}%)", flush=True)
    log(f"PID {pid} | 🔍 OPERAÇÕES MÉDIAS (ms):", flush=True)
    log(f"PID {pid} |   • Seek: {avg_seek*1000:.2f} | Read: {avg_read*1000:.2f} | Hash total: {avg_hash*1000:.2f}", flush=True)
    log(f"PID {pid} |   • Hash breakdown: Convert {avg_convert*1000:.2f} | PIL {avg_pil*1000:.2f} | Calc {avg_phash_calc*1000:.2f} | Pack {avg_pack*1000:.2f}", flush=True)
    log(f"PID {pid} |   • Metadata: {avg_metadata*1000:.2f}ms por frame", flush=True)
    log(f"PID {pid} | 🎯 Total ops: {total_seeks} seeks, {total_reads} reads, {total_hashes} hashes, {len(vectors)} frames válidos", flush=True)
    
    return video_path, vectors, metadata

def main():
    main_start = time.time()
    log("🚀 INICIANDO PROCESSAMENTO")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Timing: Busca de arquivos
    search_start = time.time()
    all_videos = [f for ext in EXTENSIONS for f in glob.glob(os.path.join(BASE_DIR, '**', f'*{ext}'), recursive=True)]
    search_time = time.time() - search_start
    
    if not all_videos:
        log("❌ Nenhum vídeo encontrado.")
        return

    total_videos = len(all_videos)
    total_size = sum(os.path.getsize(v) for v in all_videos) / (1024 * 1024)  # MB
    log(f"📦 {total_videos} vídeos encontrados em {search_time:.3f}s")
    log(f"💾 Tamanho total: {total_size:.2f} MB")
    
    processed = 0
    start_processing = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_VIDEOS) as executor:
        futures = {executor.submit(process_video, path): path for path in all_videos}
        for future in as_completed(futures):
            video_path, vectors, meta = future.result()
            processed += 1
            elapsed = time.time() - start_processing
            pct = (processed / total_videos) * 100
            eta = (elapsed / processed) * (total_videos - processed) if processed > 0 else 0
            
            log(f"📊 {processed}/{total_videos} ({pct:.1f}%) | {os.path.basename(video_path)} | ETA: {format_timedelta(eta)}")

    total_time = time.time() - main_start
    processing_time = time.time() - start_processing
    avg_time_per_video = processing_time / total_videos if total_videos > 0 else 0
    total_mb_per_sec = total_size / processing_time if processing_time > 0 else 0

    log("✅ FIM DO PROCESSAMENTO")
    log(f"⏱️  Tempo total: {format_timedelta(total_time)}")
    log(f"🚀 Tempo processamento: {format_timedelta(processing_time)}")
    log(f"📈 Média por vídeo: {avg_time_per_video:.2f}s")
    log(f"💨 Taxa geral: {total_mb_per_sec:.2f} MB/s")
    log(f"🧠 Uso de memória final: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
