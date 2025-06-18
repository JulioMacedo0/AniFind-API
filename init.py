import numpy as np
import pickle
import faiss
import cv2
from PIL import Image
import imagehash

# Caminhos dos arquivos salvos
PHASHES_PATH = "phashes.npy"
METADATA_PATH = "metadata.pkl"

# Carrega vetores e metadados
vectors = np.load(PHASHES_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# Cria índice FAISS
base_index = faiss.IndexFlatL2(64)  # Índice base para similaridade L2
index = faiss.IndexIDMap(base_index)  # Wrapper que permite IDs personalizados
ids = np.array(list(metadata.keys()), dtype=np.int64)  # Certifica que os IDs são int64
index.add_with_ids(vectors, ids)

# === Converte frame (ou imagem) para vetor pHash ===
def compute_phash_vector(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    phash = imagehash.phash(img_pil)
    return np.array(phash.hash, dtype=np.float32).flatten()

# === Busca por similaridade ===
def buscar_frame_similar(frame_bgr, k=1):
    import time
    from datetime import timedelta
    
    print("\n=== Iniciando busca por similaridade ===")
    
    # Tempo para processamento do frame
    t_inicio_proc = time.time()
    query_vector = compute_phash_vector(frame_bgr).reshape(1, -1)
    t_fim_proc = time.time()
    tempo_proc = t_fim_proc - t_inicio_proc
    print(f"Tempo para processar o frame: {tempo_proc:.4f}s")
    
    # Tempo para busca no índice FAISS
    t_inicio_busca = time.time()
    D, I = index.search(query_vector, k)
    t_fim_busca = time.time()
    tempo_busca = t_fim_busca - t_inicio_busca
    print(f"Tempo para busca no índice FAISS: {tempo_busca:.4f}s")
    
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
    print("=== Teste concluído ===\n")
