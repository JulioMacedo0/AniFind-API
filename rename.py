import os
import re
import shutil

# === CONFIGURAÇÕES ===
INPUT_DIR = "v"  # Onde estão os arquivos .mkv originais
OUTPUT_DIR = "animes/OnePiece"  # Estrutura final normalizada

# === MAPEAMENTO DE EPISÓDIOS POR TEMPORADA (baseado nas sagas) ===
SAGA_INTERVALS = [
    (1, 61),     # Temporada 1
    (62, 130),   # Temporada 2
    (131, 195),  # Temporada 3
    (196, 325),  # Temporada 4
    (326, 384),  # Temporada 5
    (385, 516),  # Temporada 6
    (517, 574),  # Temporada 7
    (575, 629),  # Temporada 8
    (630, 746),  # Temporada 9
    (747, 782),  # Temporada 10
    (783, 877),  # Temporada 11
    (878, 1085), # Temporada 12
    (1086, 1200) # Temporada 13 (Egghead e além)
]

# === DETECTA EPISÓDIO PELO NOME ===
def extract_episode_number(filename):
    # Tenta extrair número do episódio com base em padrões comuns
    match = re.search(r'\b[Ee]?(\d{1,4})\b', filename)
    if match:
        return int(match.group(1))
    return None

# === IDENTIFICA TEMPORADA E EPISÓDIO LOCAL ===
def get_season_and_local_ep(global_ep):
    for i, (start, end) in enumerate(SAGA_INTERVALS, start=1):
        if start <= global_ep <= end:
            local_ep = global_ep - start + 1
            return i, local_ep
    return None, None

# === NORMALIZA TODOS OS ARQUIVOS ===
def normalize_files():
    if not os.path.exists(INPUT_DIR):
        print(f"Diretório de entrada não encontrado: {INPUT_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    arquivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.mkv', '.mp4', '.avi'))]
    if not arquivos:
        print("Nenhum arquivo de vídeo encontrado.")
        return
    
    arquivos.sort()
    renomeados = 0

    for arquivo in arquivos:
        caminho_origem = os.path.join(INPUT_DIR, arquivo)
        ep = extract_episode_number(arquivo)
        if not ep:
            print(f"[IGNORADO] Não consegui extrair episódio de: {arquivo}")
            continue
        
        season, local_ep = get_season_and_local_ep(ep)
        if not season:
            print(f"[IGNORADO] Episódio fora do intervalo mapeado: {ep}")
            continue

        nome_final = f"OnePiece_S{season:02d}E{local_ep:02d}.mkv"
        pasta_temporada = os.path.join(OUTPUT_DIR, f"Season{season:02d}")
        os.makedirs(pasta_temporada, exist_ok=True)
        destino = os.path.join(pasta_temporada, nome_final)

        shutil.copy2(caminho_origem, destino)
        print(f"[OK] Episódio {ep} → {nome_final}")
        renomeados += 1

    print(f"\n✅ Total de arquivos renomeados: {renomeados}")

# === EXECUÇÃO ===
if __name__ == "__main__":
    normalize_files()
