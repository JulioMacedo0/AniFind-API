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
index = faiss.IndexFlatL2(64)
ids = np.array(list(metadata.keys()))
index.add_with_ids(vectors, ids)

# === Converte frame (ou imagem) para vetor pHash ===
def compute_phash_vector(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    phash = imagehash.phash(img_pil)
    return np.array(phash.hash, dtype=np.float32).flatten()

# === Busca por similaridade ===
def buscar_frame_similar(frame_bgr, k=1):
    query_vector = compute_phash_vector(frame_bgr).reshape(1, -1)
    D, I = index.search(query_vector, k)
    resultados = []
    for i in range(k):
        id_resultado = int(I[0][i])
        distancia = float(D[0][i])
        resultados.append({
            "id": id_resultado,
            "distancia": distancia,
            "metadata": metadata.get(id_resultado)
        })
    return resultados

# === TESTE ===
if __name__ == "__main__":
    # Carrega um frame de teste de uma imagem ou vídeo
    frame_teste = cv2.imread("frame_teste.jpg")  # Substitua com sua imagem de entrada

    if frame_teste is None:
        print("Erro: imagem não encontrada.")
    else:
        resultados = buscar_frame_similar(frame_teste, k=1)
        for r in resultados:
            print(f"\nID: {r['id']}")
            print(f"Distância: {r['distancia']}")
            print(f"Metadados: {r['metadata']}")
