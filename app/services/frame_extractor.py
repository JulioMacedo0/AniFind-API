# app/services/frame_extractor.py

import cv2
import os
import json
from typing import List, Dict
from datetime import datetime
import math
import imagehash
from PIL import Image

class FrameExtractorService:
    """
    Serviço para extrair frames de vídeos
    Similar a uma classe service no Node.js
    """
    
    def __init__(self, output_dir: str = "extracted_frames"):
        self.output_dir = output_dir
        self.json_file = "frames_data.json"
        
        # Criar diretório se não existir (como fs.mkdirSync no Node)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def extract_frames_from_video(self, video_path: str, anime_name: str, episode: int) -> Dict:
        """
        Extrai 3 frames por segundo: primeiro, meio e último
        
        Args:
            video_path: Caminho para o vídeo
            anime_name: Nome do anime
            episode: Número do episódio
            
        Returns:
            Dict com informações dos frames extraídos
        """
        try:
            # Abrir vídeo (como VideoCapture)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception(f"Erro ao abrir vídeo: {video_path}")
            
            # Informações do vídeo
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps
            
            print(f"📹 Vídeo: {fps} FPS, {total_frames} frames, {duration_seconds:.2f}s")
            
            extracted_frames = []
            current_second = 0
            
            # Processar segundo por segundo
            while current_second < duration_seconds:
                frames_this_second = self._extract_3_frames_from_second(
                    cap, current_second, fps, anime_name, episode
                )
                extracted_frames.extend(frames_this_second)
                current_second += 1
            
            cap.release()
            
            # Salvar informações no JSON
            result = {
                "anime": anime_name,
                "episode": episode,
                "video_info": {
                    "fps": fps,
                    "duration": duration_seconds,
                    "total_frames": total_frames
                },
                "extracted_frames": extracted_frames,
                "extraction_date": datetime.now().isoformat()
            }
            
            self._save_to_json(result)
            
            return {
                "success": True,
                "message": f"Extraídos {len(extracted_frames)} frames",
                "data": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_3_frames_from_second(self, cap, second: int, fps: float, anime: str, episode: int) -> List[Dict]:
        """
        Extrai 3 frames de um segundo específico: início, meio, fim
        """
        frames_data = []
        
        # Calcular posições dos frames (como seu algoritmo)
        start_frame = int(second * fps)  # Frame 1
        middle_frame = int(second * fps + fps / 2)  # Frame do meio
        end_frame = int((second + 1) * fps) - 1  # Último frame
        
        frame_positions = [
            ("start", start_frame),
            ("middle", middle_frame),
            ("end", end_frame)
        ]
        
        for position_name, frame_number in frame_positions:
            # Ir para o frame específico
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Gerar nome do arquivo
                timestamp = self._frame_to_timestamp(frame_number, fps)
               
              
      
                
                # Calcular hashes perceptuais
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                phash = str(imagehash.phash(pil_image))
                dhash = str(imagehash.dhash(pil_image))
                ahash = str(imagehash.average_hash(pil_image))
                
                # Adicionar aos dados
                frame_data = {
                    "frame_number": frame_number,
                    "second": second,
                    "position": position_name,
                    "timestamp": timestamp,
                    "anime": anime,
                    "episode": episode,
                    "hashes": {
                        "phash": phash,
                        "dhash": dhash,
                        "ahash": ahash
                    }
                }
                
                frames_data.append(frame_data)
                print(f"✅ Frame salvo: {position_name} ({timestamp}) - {anime} EP{episode} Frame {frame_number}")
        
        return frames_data
    
    def _frame_to_timestamp(self, frame_number: int, fps: float) -> str:
        """
        Converte número do frame para timestamp (HH:MM:SS)
        """
        total_seconds = frame_number / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _save_to_json(self, new_data: Dict):
        """
        Salva dados no JSON (como fs.writeFileSync no Node)
        """
        try:
            # Ler dados existentes
            if os.path.exists(self.json_file):
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {"extractions": []}
            
            # Adicionar nova extração
            existing_data["extractions"].append(new_data)
            
            # Salvar de volta
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Dados salvos em {self.json_file}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar JSON: {e}")
    
    def get_all_extractions(self) -> Dict:
        """
        Retorna todas as extrações salvas (como ler um db.json)
        """
        try:
            if os.path.exists(self.json_file):
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"extractions": []}
        except Exception as e:
            return {"error": str(e)}
    
    def clear_extractions(self):
        """
        Limpa todas as extrações (desenvolvimento)
        """
        try:
            # Limpar JSON
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump({"extractions": []}, f, indent=2)
            
            # Limpar pasta de frames
            if os.path.exists(self.output_dir):
                for filename in os.listdir(self.output_dir):
                    filepath = os.path.join(self.output_dir, filename)
                    if os.path.isfile(filepath):
                        os.remove(filepath)
            
            return {"message": "Todas as extrações foram limpas"}
        except Exception as e:
            return {"error": str(e)}