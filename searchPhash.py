import pickle
import numpy as np
from PIL import Image
import imagehash
import faiss

# === CONFIG ===
IMAGE_PATH = "image.png"  # caminho para imagem de consulta
INDEX_PATH = "indexes/global_index.faiss"
METADATA_PATH = "indexes/metadata.pkl"
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.70  # mínimo para exibir (em percentual)

# === Funções ===
def search(image_path):
    # Carregar FAISS + metadados
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    # Gera o pHash da imagem de consulta
    img = Image.open(image_path).convert("RGB")
    query_hash = imagehash.phash(img)
    query_vector = np.array(query_hash.hash.flatten(), dtype=np.float32).reshape(1, -1)

    # Buscar no FAISS
    D, I = index.search(query_vector, TOP_K)

    print(f"[🔍] Results for: {image_path}\n")
    for i, idx in enumerate(I[0]):
        if idx == -1:
            continue  # ignorar resultados inválidos (pode acontecer com base vazia)

        meta = metadata[idx]
        db_hash = imagehash.hex_to_hash(meta["phash"])
        hamming_dist = query_hash - db_hash
        confidence = 1 - (hamming_dist / 64)

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        print(
            f"#{i+1}: {meta['anime']} S{meta['season']:02}E{meta['episode']:02} @ {meta['timecode']} "
            f"({meta['source_file']}) | 🔗 Confidence: {confidence*100:.2f}%"
        )

if __name__ == "__main__":
    search(IMAGE_PATH)
