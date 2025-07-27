import faiss
import pickle
import numpy as np
from PIL import Image
import imagehash
from pathlib import Path
from pprint import pprint
from create_preview import create_preview  
from minio_client import upload_preview  

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

    results = []
    top_result = None
    preview_url = None

    for rank, idx in enumerate(I[0]):
        meta = metadata[idx]
        dist = D[0][rank]
        similarity = float((1 - (dist / 64)) * 100)

        result = {
            "rank": rank + 1,
            "anime": meta["anime"],
            "season": meta["season"],
            "episode": meta["episode"],
            "timecode": meta["timecode"],
            "second": meta["second"],
            "similarity": similarity,
            "source_file": meta["source_file"],
            "preview_source_path": meta["preview_source_path"]
        }

        if rank == 0:
            preview_path = create_preview(
                meta["preview_source_path"], meta["second"]
            )
            anime_folder = meta["anime"].replace(" ", "_")
            preview_url = upload_preview(preview_path, anime_folder, preview_path.name)
            result["preview_video"] = preview_url
            top_result = result

        results.append(result)

    return {
        "query": str(image_path),
        "top_result": top_result,
        "all_results": results,
        "preview_url": preview_url
    }

# Execução direta para testes
if __name__ == "__main__":
    test_image_path = "image.png"  # Substitua aqui com o caminho do frame de teste
    result = search(test_image_path)
    pprint(result)
