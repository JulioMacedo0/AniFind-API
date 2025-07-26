# === searchPhash.py ===
import faiss
import pickle
import numpy as np
from PIL import Image
import imagehash

INDEX_PATH = "indexes/global_index.faiss"
METADATA_PATH = "indexes/metadata.pkl"
TOP_K = 5

def hashes_to_vector(ph, dh, ah):
    return np.concatenate([
        np.array(imagehash.hex_to_hash(ph).hash.flatten(), dtype=np.float32),
        np.array(imagehash.hex_to_hash(dh).hash.flatten(), dtype=np.float32),
        np.array(imagehash.hex_to_hash(ah).hash.flatten(), dtype=np.float32)
    ]).reshape(1, -1)

def search(image_path):
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    img = Image.open(image_path).convert("RGB")
    ph = str(imagehash.phash(img))
    dh = str(imagehash.dhash(img))
    ah = str(imagehash.average_hash(img))
    query_vec = hashes_to_vector(ph, dh, ah)

    D, I = index.search(query_vec, TOP_K)

    print(f"[🔍] Results for: {image_path}\n")
    for rank, idx in enumerate(I[0]):
        meta = metadata[idx]
        dist = D[0][rank]
        similarity = (1 - (dist / 64)) * 100  # Simples aproximação
        print(f"#{rank+1}: {meta['anime']} S{meta['season']:02}E{meta['episode']:02} @ {meta['timecode']} "
              f"({meta['source_file']}) | 🔗 Confidence: {similarity:.2f}%")

if __name__ == "__main__":
    test_image_path = "image.png"  # Substitua pelo caminho da imagem que quer testar
    search(test_image_path)
