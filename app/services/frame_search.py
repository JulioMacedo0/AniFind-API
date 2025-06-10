# app/services/frame_search.py

import json
import os
from typing import List, Dict, Optional
from PIL import Image
import imagehash
import cv2
import numpy as np

class FrameSearchService:
    """
    Service for searching similar frames
    """
    
    def __init__(self, json_file: str = "frames_data.json"):
        self.json_file = json_file
    
    def search_similar_frame(self, image_path: str, threshold: int = 5, hash_type: str = "phash") -> Dict:
        """
        Search for similar frames based on an image
        
        Args:
            image_path: Path to the search image
            threshold: Distance threshold to consider frames similar (0-64)
            hash_type: Type of hash to use for comparison ("phash", "dhash" or "ahash")
            
        Returns:
            Search result with matches found
        """
        try:
            # Validate hash type
            if hash_type not in ["phash", "dhash", "ahash"]:
                return {
                    "success": False,
                    "error": f"Invalid hash type: {hash_type}. Use 'phash', 'dhash' or 'ahash'."
                }
                
            # Load image and calculate hashes
            search_image = Image.open(image_path)
            search_hashes = self._calculate_hashes(search_image)
            
            # Load database
            database = self._load_database()
            if not database or "extractions" not in database:
                return {
                    "success": False,
                    "message": "No indexed frames found"
                }
            
            # Search for the first match that meets the threshold
            for extraction in database["extractions"]:
                for frame in extraction["extracted_frames"]:
                    if "hashes" in frame:
                        # Calculate similarity using only the specified hash type
                        if hash_type in search_hashes and hash_type in frame["hashes"]:
                            distance = self._hamming_distance(
                                search_hashes[hash_type], 
                                frame["hashes"][hash_type]
                            )
                            
                            if distance <= threshold:
                                # Return immediately when finding the first match
                                match = {
                                    **frame,
                                    "distance": distance,
                                    "hash_type_used": hash_type,
                                    "similarity_score": 1 - (distance / 64)
                                }
                                return {
                                    "success": True,
                                    "query_hashes": search_hashes,
                                    "total_matches": 1,
                                    "matches": [match],
                                    "threshold_used": threshold,
                                    "hash_type_used": hash_type,
                                    "early_match": True
                                }
            
            # If no match was found
            return {
                "success": False,
                "query_hashes": search_hashes,
                "total_matches": 0,
                "matches": [],
                "threshold_used": threshold,
                "hash_type_used": hash_type,
                "message": "No similar frames found with the specified threshold"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_by_upload(self, image_data: bytes, threshold: int = 10, hash_type: str = "phash") -> Dict:
        """
        Search from image data in bytes (upload)
        
        Args:
            image_data: Binary image data
            threshold: Distance threshold to consider frames similar (0-64)
            hash_type: Type of hash to use for comparison ("phash", "dhash" or "ahash")
            
        Returns:
            Search result with matches found
        """
        try:
            # Validate hash type
            if hash_type not in ["phash", "dhash", "ahash"]:
                return {
                    "success": False,
                    "error": f"Invalid hash type: {hash_type}. Use 'phash', 'dhash' or 'ahash'."
                }
                
            # Convert bytes to PIL image
            image_array = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Calculate hashes
            search_hashes = self._calculate_hashes(pil_image)
            
            # Load database
            database = self._load_database()
            if not database or "extractions" not in database:
                return {
                    "success": False,
                    "message": "No indexed frames found"
                }
            
            # Search for the first match that meets the threshold
            for extraction in database["extractions"]:
                for frame in extraction["extracted_frames"]:
                    if "hashes" in frame:
                        # Calculate similarity using only the specified hash type
                        if hash_type in search_hashes and hash_type in frame["hashes"]:
                            distance = self._hamming_distance(
                                search_hashes[hash_type], 
                                frame["hashes"][hash_type]
                            )
                            
                            if distance <= threshold:
                                # Return immediately when finding the first match
                                match = {
                                    **frame,
                                    "distance": distance,
                                    "hash_type_used": hash_type,
                                    "similarity_score": 1 - (distance / 64)
                                }
                                return {
                                    "success": True,
                                    "query_hashes": search_hashes,
                                    "total_matches": 1,
                                    "matches": [match],
                                    "threshold_used": threshold,
                                    "hash_type_used": hash_type,
                                    "early_match": True
                                }
            
            # If no match was found
            return {
                "success": False,
                "query_hashes": search_hashes,
                "total_matches": 0,
                "matches": [],
                "threshold_used": threshold,
                "hash_type_used": hash_type,
                "message": "No similar frames found with the specified threshold"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_hashes(self, image: Image.Image) -> Dict[str, str]:
        """
        Calcula múltiplos hashes de uma imagem
        """
        return {
            "phash": str(imagehash.phash(image)),
            "dhash": str(imagehash.dhash(image)),
            "ahash": str(imagehash.average_hash(image))
        }
    
    def _calculate_similarity(self, hash1: Dict[str, str], hash2: Dict[str, str]) -> Dict:
        """
        Calcula similaridade entre dois conjuntos de hashes
        """
        distances = {}
        
        # Calcular distância para cada tipo de hash
        for hash_type in ["phash", "dhash", "ahash"]:
            if hash_type in hash1 and hash_type in hash2:
                dist = self._hamming_distance(hash1[hash_type], hash2[hash_type])
                distances[hash_type] = dist
        
        # Encontrar menor distância (mais similar)
        min_distance = min(distances.values()) if distances else 64
        best_hash_type = min(distances, key=distances.get) if distances else "none"
        
        return {
            "distances": distances,
            "min_distance": min_distance,
            "best_match_type": best_hash_type
        }
    
    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Calcula distância de Hamming entre dois hashes
        """
        if len(hash1) != len(hash2):
            return 64  # Máxima distância
        
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def _load_database(self) -> Optional[Dict]:
        """
        Carrega banco de dados do JSON
        """
        try:
            if os.path.exists(self.json_file):
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception:
            return None
    
    def get_stats(self) -> Dict:
        """
        Estatísticas do banco de frames
        """
        database = self._load_database()
        if not database:
            return {"total_extractions": 0, "total_frames": 0}
        
        total_frames = 0
        animes = set()
        episodes = set()
        
        for extraction in database.get("extractions", []):
            total_frames += len(extraction.get("extracted_frames", []))
            animes.add(extraction.get("anime", "unknown"))
            episodes.add(f"{extraction.get('anime', 'unknown')}_ep{extraction.get('episode', 0)}")
        
        return {
            "total_extractions": len(database.get("extractions", [])),
            "total_frames": total_frames,
            "unique_animes": len(animes),
            "unique_episodes": len(episodes),
            "animes": list(animes)
        }