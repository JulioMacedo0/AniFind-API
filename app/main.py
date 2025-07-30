from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import image_search
from app.services.image_search_service import ImageSearchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the search service
    print("🚀 Initializing AniFind API...")
    try:
        ImageSearchService.initialize()
        print("✅ AniFind API initialization completed")
    except Exception as e:
        print(f"❌ Failed to initialize AniFind API: {e}")
        raise
    
    yield
    
    # Shutdown: Clean up resources if needed
    print("🔥 AniFind API shutting down...")


app = FastAPI(
    title="AniFind API",
    description="API to find anime episodes through image search",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(image_search.router, prefix="/api/v1", tags=["image-search"])

@app.get("/")
async def root():
    return {"message": "AniFind API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
