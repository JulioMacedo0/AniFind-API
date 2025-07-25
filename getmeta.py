import os
import cv2
import pickle
import numpy as np

# Caminhos dos arquivos
metadata_path = "database/individual/metadata_naruto_S01E01.pkl"
output_image_path = "frame_extraido.png"

# Carrega os metadados
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# Pega o primeiro item (ou altere para um ID específico)
sample_id, sample_meta = next(iter(metadata.items()))

# Extrai as informações do metadado
video_path = sample_meta["filepath"]
second = sample_meta["second"]
position = sample_meta["position"]

# Determina o offset com base na posição (start, middle, end)
offsets = {"start": 0.1, "middle": 0.5, "end": 0.9}
offset = offsets.get(position, 0.5)
timestamp = second + offset

# Abre o vídeo
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Calcula o índice do frame
frame_index = int(timestamp * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

# Lê o frame
ret, frame = cap.read()
cap.release()

# Salva o frame extraído como imagem
if ret:
    cv2.imwrite(output_image_path, frame)
    print(f"✅ Frame extraído e salvo como '{output_image_path}'")
else:
    print("❌ Erro ao extrair frame do vídeo")
