#!/usr/bin/env python3
"""
Script to run the AniFind FastAPI application.

Usage:
    python run_api.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 8080
    python run_api.py --reload  # For development
"""

import uvicorn
import argparse
from pathlib import Path
import sys

# Adicionar o diretório atual ao Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def main():
    parser = argparse.ArgumentParser(description="Run AniFind API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (default: 1)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting AniFind API...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {'Enabled' if args.reload else 'Disabled'}")
    print(f"👥 Workers: {args.workers}")
    print(f"📖 Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 Health check: http://{args.host}:{args.port}/health")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        access_log=True
    )


if __name__ == "__main__":
    main()
