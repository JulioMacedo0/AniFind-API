import faiss
import pickle
import numpy as np
from PIL import Image
import imagehash
from pathlib import Path
from pprint import pprint
from create_preview import create_preview  
from minio_client import upload_preview  
from config import config

INDEX_PATH = config.FAISS_INDEX_PATH
METADATA_PATH = config.METADATA_PATH
TOP_K = config.SEARCH_TOP_K

# Global variables for persistent data loading
_cached_index = None
_cached_metadata = None
_is_loaded = False

def load_data():
    """Load FAISS index and metadata once and cache them globally."""
    global _cached_index, _cached_metadata, _is_loaded
    
    if _is_loaded:
        return _cached_index, _cached_metadata
    
    try:
        print("🔄 Loading FAISS index and metadata...")
        _cached_index = faiss.read_index(str(INDEX_PATH))
        with open(str(METADATA_PATH), "rb") as f:
            _cached_metadata = pickle.load(f)
        
        _is_loaded = True
        print(f"✅ Data loaded successfully:")
        print(f"📊 Index size: {_cached_index.ntotal} vectors")
        print(f"📋 Metadata entries: {len(_cached_metadata)}")
        
        return _cached_index, _cached_metadata
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def get_data_status():
    """Get the current status of loaded data."""
    return {
        "loaded": _is_loaded,
        "index_size": _cached_index.ntotal if _cached_index else 0,
        "metadata_entries": len(_cached_metadata) if _cached_metadata else 0
    }

def clean_anime_name(name):
    """Clean anime name by removing unwanted characters."""
    if not name:
        return name
    
    # Remove trailing dashes and spaces
    name = name.rstrip('- ')
    
    # Remove leading dashes and spaces
    name = name.lstrip('- ')
    
    # Replace multiple spaces with single space
    name = ' '.join(name.split())
    
    return name

def hashes_to_vector(ph, dh, ah):
    return np.concatenate([
        np.array(imagehash.hex_to_hash(ph).hash.flatten(), dtype=np.float32),
        np.array(imagehash.hex_to_hash(dh).hash.flatten(), dtype=np.float32),
        np.array(imagehash.hex_to_hash(ah).hash.flatten(), dtype=np.float32)
    ]).reshape(1, -1)

def search(image_path, use_cached=False):
    """
    Search for anime episode using an image.
    
    Args:
        image_path: Path to the search image
        use_cached: If True, use cached data (load once). If False, load fresh data each time.
    
    Returns:
        Dictionary with search results
    """
    if use_cached:
        index, metadata = load_data()
    else:
        # Original behavior: load fresh data each time
        index = faiss.read_index(str(INDEX_PATH))
        with open(str(METADATA_PATH), "rb") as f:
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

        # Construct full video path from relative path stored in metadata
        video_full_path = config.VIDEO_BASE_DIR / meta["preview_source_path"]
        
        result = {
            "rank": rank + 1,
            "anime": clean_anime_name(meta["anime"]),
            "season": meta["season"],
            "episode": meta["episode"],
            "timecode": meta["timecode"],
            "second": meta["second"],
            "similarity": similarity,
            "anime_id": meta["anime_id"], 
            "source_file": meta["source_file"],
            "preview_source_path": str(video_full_path)
        }

        if rank == 0:
            try:
                preview_path = create_preview(
                    str(video_full_path), meta["second"]
                )
                # Use cleaned anime name for folder
                anime_folder = clean_anime_name(meta["anime"]).replace(" ", "_")
                preview_url = upload_preview(preview_path, anime_folder, preview_path.name)
                result["preview_video"] = preview_url
            except Exception as preview_error:
                print(f"Warning: Could not generate preview: {preview_error}")
                result["preview_video"] = None
            
            top_result = result

        results.append(result)

    return {
        "top_result": top_result,
        "all_results": results,
        "preview_url": preview_url
    }

# Execução direta para testes
if __name__ == "__main__":
    test_image_path = "image.png"  # Substitua aqui com o caminho do frame de teste
    result = search(test_image_path, use_cached=False)  # Use False for testing original behavior
    pprint(result)
