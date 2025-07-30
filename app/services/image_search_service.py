import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add root directory to path to import searchPhash
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

try:
    from searchPhash import search, load_data, get_data_status
except ImportError as e:
    print(f"Error importing searchPhash: {e}")
    search = None
    load_data = None
    get_data_status = None


class ImageSearchService:
    @staticmethod
    def initialize():
        """Initialize the service by pre-loading FAISS index and metadata."""
        if load_data is None:
            raise Exception("searchPhash module not available")
        
        try:
            load_data()
            print("✅ ImageSearchService initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing ImageSearchService: {e}")
            raise
    
    @staticmethod
    def search_anime_episode(image_path: str) -> Dict[str, Any]:
        """
        Search for anime episode using an image.
        
        Args:
            image_path: Path to the search image
            
        Returns:
            Dictionary with search results
            
        Raises:
            Exception: If there's an error in search or if searchPhash is not available
        """
        if search is None:
            raise Exception("searchPhash module not available")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Use cached data for better performance
            result = search(image_path, use_cached=True)
            return result
        except Exception as e:
            raise Exception(f"Error performing search: {str(e)}")
    
    @staticmethod
    def validate_image_file(file_path: str) -> bool:
        """
        Validate if the file is a valid image.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if it's a valid image, False otherwise
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        file_extension = Path(file_path).suffix.lower()
        
        return file_extension in valid_extensions and os.path.exists(file_path)
    
    @staticmethod
    def get_service_status() -> Dict[str, Any]:
        """Get service status and statistics."""
        if get_data_status is None:
            return {
                "error": "searchPhash module not available",
                "initialized": False,
                "index_loaded": False,
                "metadata_loaded": False,
                "index_size": 0,
                "metadata_entries": 0
            }
        
        try:
            status = get_data_status()
            return {
                "initialized": status["loaded"],
                "index_loaded": status["loaded"],
                "metadata_loaded": status["loaded"],
                "index_size": status["index_size"],
                "metadata_entries": status["metadata_entries"]
            }
        except Exception as e:
            return {
                "error": str(e),
                "initialized": False,
                "index_loaded": False,
                "metadata_loaded": False,
                "index_size": 0,
                "metadata_entries": 0
            }
