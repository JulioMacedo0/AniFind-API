# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

INDIVIDUAL_DIR = "database/individual"
OUTPUT_HASHES_PATH = "database/phashes.npy"
OUTPUT_METADATA_PATH = "database/metadata.pkl"

combined_hashes = []
combined_metadata = {}
current_id = 0

print(f"🔍 Lendo arquivos individuais de: {INDIVIDUAL_DIR}")

for file in os.listdir(INDIVIDUAL_DIR):
    if file.startswith("phashes_") and file.endswith(".npy"):
        base = file.replace("phashes_", "").replace(".npy", "")
        hash_path = os.path.join(INDIVIDUAL_DIR, f"phashes_{base}.npy")
        meta_path = os.path.join(INDIVIDUAL_DIR, f"metadata_{base}.pkl")

        if not os.path.exists(meta_path):
            print(f"⚠️  Metadado ausente para {base}, pulando...")
            continue

        hashes = np.load(hash_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        for i, h in enumerate(hashes):
            combined_hashes.append(h)
            combined_metadata[current_id] = metadata.get(i, {})
            current_id += 1

print(f"✅ Total de hashes: {len(combined_hashes)}")
print(f"✅ Total de metadados: {len(combined_metadata)}")

os.makedirs("database", exist_ok=True)
np.save(OUTPUT_HASHES_PATH, np.array(combined_hashes, dtype=np.uint64))
with open(OUTPUT_METADATA_PATH, "wb") as f:
    pickle.dump(combined_metadata, f)

print(f"💾 Consolidado em:")
print(f"  - {OUTPUT_HASHES_PATH}")
print(f"  - {OUTPUT_METADATA_PATH}")
