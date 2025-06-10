# app/services/__init__.py

# Importações para facilitar o uso (como index.js no Node)
from .frame_extractor import FrameExtractorService
from .frame_search import FrameSearchService

# Exportar tudo que pode ser usado externamente
__all__ = [
    "FrameExtractorService",
    "FrameSearchService"
]