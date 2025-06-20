import numpy as np
import pickle
import faiss
import cv2
from PIL import Image
import imagehash
import os
import psutil  # Para monitoramento de memória

# Funções para monitoramento de memória
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

# Caminhos dos arquivos salvos
PHASHES_PATH = "database/phashes.npy"
METADATA_PATH = "database/metadata.pkl"

# Carrega vetores e metadados
print(f"Memória antes de carregar os dados: {update_max_memory():.2f} MB")
vectors_raw = np.load(PHASHES_PATH)
print(f"Vetores carregados com formato: {vectors_raw.shape}")

# Descompactar os hashes para o formato 2D que o FAISS espera
vectors_list = []
for hash_value in vectors_raw:
    # Converte cada hash de uint64 para 64 bits binários
    binary_hash = np.unpackbits(np.array([hash_value], dtype=np.uint64).view(np.uint8))[:64]
    # Converte para float32 para compatibilidade com FAISS
    vectors_list.append(binary_hash.astype(np.float32))

vectors = np.array(vectors_list)
print(f"Vetores reformatados para formato: {vectors.shape}")
print(f"Memória após processar vetores: {update_max_memory():.2f} MB")

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)
print(f"Memória após carregar metadados: {update_max_memory():.2f} MB")

# Cria índice FAISS
print("Criando índice FAISS...")
base_index = faiss.IndexFlatL2(64)  # Índice base para similaridade L2 (64 bits do hash)
index = faiss.IndexIDMap(base_index)  # Wrapper que permite IDs personalizados
ids = np.array(list(metadata.keys()), dtype=np.int64)  # Certifica que os IDs são int64
print(f"Memória antes de adicionar vetores ao índice: {update_max_memory():.2f} MB")
index.add_with_ids(vectors, ids)
print(f"Memória após criar índice FAISS: {update_max_memory():.2f} MB")

# === Converte frame (ou imagem) para vetor pHash ===
def compute_phash_vector(frame_bgr):
    # Redimensiona para o mesmo tamanho usado em generate_data.py
    if frame_bgr.shape[0] > 64 or frame_bgr.shape[1] > 64:
        frame_bgr = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    phash = imagehash.phash(img_pil)
    
    # Retorna o hash em formato compatível com o índice FAISS
    binary_hash = np.array(phash.hash, dtype=np.float32).flatten()
    return binary_hash

# === Busca por similaridade ===
def buscar_frame_similar(frame_bgr, k=1):
    import time
    from datetime import timedelta
    
    print("\n=== Iniciando busca por similaridade ===")
    print(f"Memória antes da busca: {update_max_memory():.2f} MB")
    
    # Tempo para processamento do frame
    t_inicio_proc = time.time()
    query_vector = compute_phash_vector(frame_bgr).reshape(1, -1)
    t_fim_proc = time.time()
    tempo_proc = t_fim_proc - t_inicio_proc
    print(f"Tempo para processar o frame: {tempo_proc:.4f}s")
    print(f"Memória após processar o frame: {update_max_memory():.2f} MB")
    
    # Tempo para busca no índice FAISS
    t_inicio_busca = time.time()
    D, I = index.search(query_vector, k)
    t_fim_busca = time.time()
    tempo_busca = t_fim_busca - t_inicio_busca
    print(f"Tempo para busca no índice FAISS: {tempo_busca:.4f}s")
    print(f"Memória após busca no FAISS: {update_max_memory():.2f} MB")
    
    # Tempo para montar resultados
    t_inicio_result = time.time()
    resultados = []
    for i in range(k):
        id_resultado = int(I[0][i])
        distancia = float(D[0][i])
        resultados.append({
            "id": id_resultado,
            "distancia": distancia,
            "metadata": metadata.get(id_resultado)
        })
    t_fim_result = time.time()
    tempo_result = t_fim_result - t_inicio_result
    print(f"Tempo para montar resultados: {tempo_result:.4f}s")
    
    # Tempo total
    tempo_total = tempo_proc + tempo_busca + tempo_result
    print(f"Tempo total da busca: {tempo_total:.4f}s")
    print("=======================================\n")
    
    return resultados

# === TESTE ===
if __name__ == "__main__":
    import time
    from datetime import timedelta
    
    print("=== Iniciando teste de busca ===")
    t_inicio_total = time.time()
    
    # Informações sobre os dados carregados
    print(f"Total de vetores no banco de dados: {len(vectors)}")
    print(f"Total de metadados no banco de dados: {len(metadata)}")
    print(f"Memória atual: {update_max_memory():.2f} MB")
    
    # Carrega um frame de teste de uma imagem ou vídeo
    print("\nCarregando imagem para teste...")
    t_inicio_carga = time.time()
    frame_teste = cv2.imread("image.png")  # Substitua com sua imagem de entrada
    t_fim_carga = time.time()
    
    if frame_teste is None:
        print("Erro: imagem não encontrada.")
    else:
        print(f"Imagem carregada. Dimensões: {frame_teste.shape}")
        print(f"Tempo para carregar imagem: {t_fim_carga - t_inicio_carga:.4f}s")
        
        # Busca por similaridade
        print("\nRealizando busca de similaridade...")
        resultados = buscar_frame_similar(frame_teste, k=1)
        
        # Exibe resultados
        print("\n=== Resultados encontrados ===")
        for r in resultados:
            print(f"ID: {r['id']}")
            print(f"Distância: {r['distancia']}")
            print(f"Metadados: {r['metadata']}")
            
            # Se tiver metadados, mostrar informações mais detalhadas
            if r['metadata']:
                anime = r['metadata'].get('anime', 'Desconhecido')
                season = r['metadata'].get('season', 0)
                episode = r['metadata'].get('episode', 0)
                second = r['metadata'].get('second', 0)
                position = r['metadata'].get('position', '')
                print(f"Encontrado em: {anime} - Temporada {season} Episódio {episode}")
                print(f"Tempo: {second//60}:{second%60:02d} ({position})")
      # Tempo total de execução
    t_fim_total = time.time()
    tempo_total = t_fim_total - t_inicio_total
    print(f"\nTempo total de execução: {tempo_total:.4f}s ({timedelta(seconds=int(tempo_total))})")
    print(f"Memória final: {update_max_memory():.2f} MB")
    print(f"Pico de uso de memória: {max_memory:.2f} MB")
    print("=== Teste concluído ===\n")
