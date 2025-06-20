"""
Service for searching similar images using perceptual hash
Based on the functionality from init.py
"""

import os
import numpy as np
import pickle
import faiss
import cv2
from PIL import Image
import imagehash
import time
from datetime import timedelta
import logging
import psutil

# Setup logging
logger = logging.getLogger("image_search_service")

class ImageSearchService:
    """
    Service for searching similar images using perceptual hash and FAISS
    
    This service implements the functionality from init.py as a reusable class
    that can be used in a web API or other applications.
    """
    
    def __init__(self):
        """Initialize the service (data is loaded on demand or with load_data)"""
        self.index = None
        self.metadata = None
        self.binary_vectors = None
        self.max_memory = 0
        self._data_loaded = False
        
        # Path constants
        self.PHASHES_PATH = "database/phashes.npy"
        self.METADATA_PATH = "database/metadata.pkl"
    
    def get_memory_usage(self):
        """Return the current memory usage in MB"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024  # Convert bytes to MB
    
    def update_max_memory(self):
        """Update and return the maximum memory usage"""
        current = self.get_memory_usage()
        if current > self.max_memory:
            self.max_memory = current
        return current
    
    def load_data(self):
        """Load perceptual hash data and create FAISS index"""
        if self._data_loaded:
            logger.info("Data already loaded, skipping")
            return
        
        logger.info(f"Memory before loading data: {self.update_max_memory():.2f} MB")
        
        try:
            # Check if data files exist
            if not os.path.exists(self.PHASHES_PATH):
                raise FileNotFoundError(f"Hash file not found at {self.PHASHES_PATH}")
            if not os.path.exists(self.METADATA_PATH):
                raise FileNotFoundError(f"Metadata file not found at {self.METADATA_PATH}")
            
            # Load hashes
            hashes_raw = np.load(self.PHASHES_PATH)
            logger.info(f"Loaded {len(hashes_raw)} hashes of 64 bits")
            logger.info(f"Memory after loading hashes: {self.update_max_memory():.2f} MB")

            # Load metadata
            with open(self.METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded metadata with {len(self.metadata)} entries")
            logger.info(f"Memory after loading metadata: {self.update_max_memory():.2f} MB")
            
            # Prepare binary vectors for FAISS
            logger.info("Preparing hashes for binary index...")
            self.binary_vectors = np.zeros((len(hashes_raw), 8), dtype=np.uint8)
            for i, hash_value in enumerate(hashes_raw):
                # Convert uint64 hash to 8 bytes (uint8)
                self.binary_vectors[i] = np.array([hash_value], dtype=np.uint64).view(np.uint8)
                
            logger.info(f"Binary vectors prepared: {self.binary_vectors.shape}")
            logger.info(f"Memory after preparing binary vectors: {self.update_max_memory():.2f} MB")
            
            # Create FAISS binary index with Hamming distance
            logger.info("Creating FAISS binary index with Hamming distance...")
            binary_dim = 64  # 64 bits for each hash
            base_index = faiss.IndexBinaryFlat(binary_dim)
            self.index = faiss.IndexBinaryIDMap(base_index)
            ids = np.array(list(self.metadata.keys()), dtype=np.int64)
            
            logger.info(f"Memory before adding vectors to index: {self.update_max_memory():.2f} MB")
            self.index.add_with_ids(self.binary_vectors, ids)
            logger.info(f"Memory after creating FAISS binary index: {self.update_max_memory():.2f} MB")
            
            self._data_loaded = True
            logger.info("Data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def compute_phash_vector(self, frame_bgr):
        """
        Compute perceptual hash vector for an image frame
        
        Args:
            frame_bgr: OpenCV image in BGR format
            
        Returns:
            Binary vector suitable for FAISS IndexBinaryFlat (shape [1, 8])
        """
        # Resize for consistency with generate_data.py
        if frame_bgr.shape[0] > 64 or frame_bgr.shape[1] > 64:
            frame_bgr = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Convert from BGR to RGB for PIL
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        phash_obj = imagehash.phash(img_pil)
        
        # Convert to uint64 format
        hash_uint64 = np.packbits(phash_obj.hash.flatten()).view(np.uint64)[0]
        
        # Convert to binary format for IndexBinaryFlat (array of 8 bytes)
        binary_hash = np.array([hash_uint64], dtype=np.uint64).view(np.uint8)
        
        return binary_hash.reshape(1, 8)  # Format: (1, 8) - one vector of 8 bytes (64 bits)
    
    def search_similar_frame(self, frame_bgr, k=5):
        """
        Search for similar frames in the FAISS index
        
        Args:
            frame_bgr: OpenCV image in BGR format
            k: Number of results to return
            
        Returns:
            List of results with metadata, similarity scores and timing information
        """
        if not self._data_loaded:
            self.load_data()
        
        start_time = time.time()
        logger.info("Starting similarity search")
        logger.info(f"Memory before search: {self.update_max_memory():.2f} MB")
        
        # Process query frame
        t_start_proc = time.time()
        query_vector = self.compute_phash_vector(frame_bgr)
        t_end_proc = time.time()
        proc_time = t_end_proc - t_start_proc
        logger.info(f"Frame processing time: {proc_time:.4f}s")
        logger.info(f"Memory after processing frame: {self.update_max_memory():.2f} MB")
        
        # Search in FAISS index
        t_start_search = time.time()
        D, I = self.index.search(query_vector, k)
        t_end_search = time.time()
        search_time = t_end_search - t_start_search
        logger.info(f"FAISS search time: {search_time:.4f}s")
        logger.info(f"Memory after FAISS search: {self.update_max_memory():.2f} MB")
        
        # Format results
        t_start_results = time.time()
        results = []
        for i in range(k):
            if i < len(I[0]):  # Verify we have enough results
                id_result = int(I[0][i])
                # Hamming distance is the number of different bits (0-64)
                hamming_dist = int(D[0][i])
                # Calculate similarity percentage (64 bits - 0 bits different = 100% similar)
                similarity_pct = 100 * (64 - hamming_dist) / 64
                
                result_metadata = self.metadata.get(id_result, {})
                
                results.append({
                    "id": id_result,
                    "distance": hamming_dist,
                    "similarity_percentage": similarity_pct,
                    "metadata": result_metadata
                })
        t_end_results = time.time()
        results_time = t_end_results - t_start_results
        logger.info(f"Results formatting time: {results_time:.4f}s")
        
        # Total time
        total_time = time.time() - start_time
        logger.info(f"Total search time: {total_time:.4f}s")
        
        timing = {
            "processing_time": proc_time,
            "search_time": search_time,
            "results_time": results_time,
            "total_time": total_time
        }
        
        return results, timing
    
    def search_from_bytes(self, image_bytes, k=5):
        """
        Search for similar frames from raw image bytes
        
        Args:
            image_bytes: Raw image bytes (from uploaded file)
            k: Number of results to return
            
        Returns:
            Search results and timing information
        """
        # Convert bytes to cv2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image data. Make sure it's a valid image file.")
        
        # Search using the image
        return self.search_similar_frame(img, k)
    
    def get_status(self):
        """Get service status including memory usage and data info"""
        return {
            "data_loaded": self._data_loaded,
            "index_size": self.index.ntotal if self._data_loaded else 0,
            "metadata_count": len(self.metadata) if self._data_loaded else 0,
            "current_memory_mb": self.get_memory_usage(),
            "max_memory_mb": self.max_memory
        }
