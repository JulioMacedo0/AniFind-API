import cv2
import numpy as np
from PIL import Image
import imagehash
import os
import pickle
import time
import glob
import re
from datetime import timedelta
import psutil  # Para monitoramento de memória

# === CONFIGURAÇÕES ===
# Estrutura de diretórios recomendada:
# BASE_DIR/
# ├── AnimeNome1/
# │   ├── Season01/
# │   │   ├── AnimeNome1_S01E01.mp4
# │   │   ├── AnimeNome1_S01E02.mp4
# │   │   └── ...
# │   ├── Season02/
# │   │   └── ...
# ├── AnimeNome2/
# │   └── ...

BASE_DIR = "animes"  # Diretório base onde os animes estão organizados
OUTPUT_DIR = "database"  # Diretório onde serão salvos os arquivos de dados
OUTPUT_HASH_PATH = os.path.join(OUTPUT_DIR, "phashes.npy")
OUTPUT_META_PATH = os.path.join(OUTPUT_DIR, "metadata.pkl")
FRAME_POSITIONS = ['start', 'middle', 'end']
EXTENSIONS = ['.mp4', '.mkv', '.avi']  # Extensões de vídeo suportadas

# === EXTRAI METADADOS DO NOME DO ARQUIVO E DIRETÓRIO ===
def parse_video_path(filepath):
    """
    Extrai metadados (anime, temporada, episódio) do caminho do arquivo.
    Tenta extrair informações tanto do nome do arquivo quanto da estrutura de diretórios.
    
    Formatos suportados:
    - Por diretório: BASE_DIR/NomeAnime/SeasonXX/Arquivo.mp4
    - Por nome de arquivo: NomeAnime_SXXEXX.mp4, NomeAnime - SXX EXX.mp4
    """
    filepath = os.path.normpath(filepath)
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    
    # Primeiro, tenta extrair do nome do arquivo usando regex
    # Padrões comuns de nomes de episódios
    patterns = [
        r'(.+)[_\s-][Ss](\d+)[Ee](\d+)',  # AnimeNome_S01E01, AnimeNome - S01E01
        r'(.+)[_\s-](\d+)x(\d+)',         # AnimeNome_01x01, AnimeNome - 01x01
        r'(.+)[_\s]\[?(\d+)[\.\s](\d+)\]?' # AnimeNome 01.01, AnimeNome [01.01]
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            anime = match.group(1).strip()
            season = int(match.group(2))
            episode = int(match.group(3))
            return anime, season, episode
    
    # Se não conseguir extrair do nome, tenta pela estrutura de diretórios
    parts = filepath.split(os.sep)
    if len(parts) >= 3:
        # Tenta encontrar temporada do diretório (ex: Season01, S01, etc)
        season_dir = parts[-2]
        season_match = re.search(r'[Ss](?:eason)?(\d+)', season_dir)
        
        if season_match:
            season = int(season_match.group(1))
            anime = parts[-3]  # O diretório pai é o nome do anime
            
            # Tenta extrair o número do episódio do nome do arquivo
            ep_match = re.search(r'[Ee](?:pisode|p)?[\s\._-]*(\d+)', name)
            if ep_match:
                episode = int(ep_match.group(1))
                return anime, season, episode
    
    # Se chegou aqui, não conseguiu extrair informações suficientes
    raise ValueError(f"Não foi possível extrair metadados do caminho: {filepath}")

# === CONVERTE FRAME EM VETOR USANDO pHash ===
def compute_phash_vector(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Converte para RGB
    phash = imagehash.phash(img)
    return np.array(phash.hash, dtype=np.float32).flatten()  # 64-dim (8x8)

# === PROCESSA UM VÍDEO ===
def process_video(video_path, start_id=0):
    start_time = time.time()
    print(f"Memória antes de abrir o vídeo: {update_max_memory():.2f} MB")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERRO: Não foi possível abrir o vídeo: {video_path}")
        return [], {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"ERRO: Vídeo vazio ou inválido: {video_path}")
        return [], {}
        
    duration = int(total_frames // fps)

    print(f"Memória após abrir o vídeo: {update_max_memory():.2f} MB")

    try:
        anime, season, episode = parse_video_path(video_path)
    except ValueError as e:
        print(f"ERRO: {e}")
        return [], {}
        
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
            try:
                vector = compute_phash_vector(frame)
            except Exception as e:
                print(f"Erro ao processar frame em {sec}s, posição '{pos}': {e}")
                continue

            vectors.append(vector)
            metadata[current_id] = {
                "anime": anime,
                "season": season,
                "episode": episode,
                "second": sec,
                "position": pos,
                "filepath": video_path  # Adicionamos o caminho para referência
            }
            current_id += 1
            frames_processed += 1  
    cap.release()
    
    # Monitoramento de memória final
    print(f"\nProcessamento do vídeo concluído.")
    print(f"Frames processados: {frames_processed}/{total_expected_frames}")
    print(f"Vetores gerados: {len(vectors)}")
    print(f"Metadados gerados: {len(metadata)}")
    print(f"Memória após processar todos os frames: {update_max_memory():.2f} MB")
    
    return vectors, metadata

# === ENCONTRAR VÍDEOS RECURSIVAMENTE ===
def find_videos(base_dir):
    """Encontra todos os arquivos de vídeo em um diretório e suas subpastas"""
    videos = []
    for ext in EXTENSIONS:
        videos.extend(glob.glob(os.path.join(base_dir, "**", f"*{ext}"), recursive=True))
    return sorted(videos)

# === PROCESSA MÚLTIPLOS VÍDEOS ===
def process_anime_collection(base_dir):
    """Processa uma coleção de animes organizada em diretórios"""
    all_videos = find_videos(base_dir)
    total_videos = len(all_videos)
    
    if total_videos == 0:
        print(f"Nenhum vídeo encontrado em: {base_dir}")
        return [], {}
    
    print(f"Encontrados {total_videos} vídeos para processamento")
    
    all_vectors = []
    all_metadata = {}
    current_id = 0
    
    for i, video_path in enumerate(all_videos):
        print(f"===== Processando vídeo {i+1}/{total_videos} =====")
        print(f"Arquivo: {video_path}")
        
        vectors, metadata = process_video(video_path, current_id)
        
        if vectors and metadata:
            all_vectors.extend(vectors)
            all_metadata.update(metadata)
            current_id += len(vectors)
            
            # Salva checkpoints para evitar perda de dados em caso de falhas
            if (i + 1) % 5 == 0 or (i + 1) == total_videos:
                print(f"\nSalvando checkpoint após {i+1}/{total_videos} vídeos...")
                save_checkpoint(all_vectors, all_metadata, i+1)
    
    return all_vectors, all_metadata

# === SALVAR CHECKPOINT ===
def save_checkpoint(vectors, metadata, num_videos):
    """Salva os dados processados em um checkpoint temporário"""
    # Cria diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    checkpoint_hash = os.path.join(OUTPUT_DIR, f"checkpoint_{num_videos}_phashes.npy")
    checkpoint_meta = os.path.join(OUTPUT_DIR, f"checkpoint_{num_videos}_metadata.pkl")
    
    np.save(checkpoint_hash, np.array(vectors, dtype=np.float32))
    with open(checkpoint_meta, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"Checkpoint salvo: {num_videos} vídeos, {len(vectors)} frames processados")

# === FUNÇÕES PARA MONITORAMENTO DE MEMÓRIA ===
def get_memory_usage():
    """Retorna o uso de memória atual do processo em MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Converte bytes para MB

# Variável global para registrar uso máximo de memória
max_memory = 0
def update_max_memory():
    """Atualiza e retorna o uso máximo de memória"""
    global max_memory
    current = get_memory_usage()
    if current > max_memory:
        max_memory = current
    return current

# === MAIN ===
def main():
    global max_memory
    
    # Cria o diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("PROCESSADOR DE COLEÇÃO DE ANIMES PARA FRAMES PHASH")
    print("=" * 60)
    print(f"Diretório base: {BASE_DIR}")
    print(f"Diretório de saída: {OUTPUT_DIR}")
    print(f"Memória inicial: {update_max_memory():.2f} MB")
    
    # Verifica se o diretório base existe
    if not os.path.isdir(BASE_DIR):
        print(f"ERRO: Diretório base não encontrado: {BASE_DIR}")
        print("Criando diretório de exemplo...")
        os.makedirs(BASE_DIR, exist_ok=True)
        print(f"Por favor, coloque seus animes em '{BASE_DIR}' seguindo a estrutura recomendada:")
        print("BASE_DIR/")
        print("├── AnimeNome1/")
        print("│   ├── Season01/")
        print("│   │   ├── AnimeNome1_S01E01.mp4")
        print("│   │   ├── AnimeNome1_S01E02.mp4")
        print("│   │   └── ...")
        print("└── AnimeNome2/")
        print("    └── ...")
        return
    
    start_time = time.time()
    
    # Processa todos os vídeos na coleção
    vectors, metadata = process_anime_collection(BASE_DIR)
    
    if not vectors:
        print("Nenhum vetor foi gerado. Verifique os erros acima.")
        return
    
    print(f"\nMemória após processar coleção: {update_max_memory():.2f} MB")
    
    # Salva os vetores finais
    print(f"Salvando {len(vectors)} vetores no arquivo final...")
    np.save(OUTPUT_HASH_PATH, np.array(vectors, dtype=np.float32))
    print(f"Memória após salvar vetores: {update_max_memory():.2f} MB")
    
    # Salva os metadados finais
    print(f"Salvando metadados de {len(metadata)} frames no arquivo final...")
    with open(OUTPUT_META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    # Gera um resumo da coleção processada
    anime_stats = {}
    for meta in metadata.values():
        anime = meta['anime']
        season = meta['season']
        episode = meta['episode']
        
        if anime not in anime_stats:
            anime_stats[anime] = {}
        
        if season not in anime_stats[anime]:
            anime_stats[anime][season] = set()
            
        anime_stats[anime][season].add(episode)
    
    print("\n" + "=" * 60)
    print("RESUMO DA COLEÇÃO PROCESSADA")
    print("=" * 60)
    for anime, seasons in anime_stats.items():
        print(f"Anime: {anime}")
        for season, episodes in seasons.items():
            print(f"  - Temporada {season}: {len(episodes)} episódios (Eps: {min(episodes)}-{max(episodes)})")
    
    print("\n" + "=" * 60)
    print(f"Salvos {len(vectors)} vetores e {len(metadata)} metadados em disco.")
    print(f"Arquivos gerados:")
    print(f"  - Vetores: {OUTPUT_HASH_PATH}")
    print(f"  - Metadados: {OUTPUT_META_PATH}")
    print(f"Memória final: {update_max_memory():.2f} MB")
    print(f"Pico de uso de memória: {max_memory:.2f} MB")
    
    processo_time = time.time() - start_time
    print(f"Tempo total de execução: {timedelta(seconds=int(processo_time))}")
    print("=" * 60)

if __name__ == "__main__":
    main()
