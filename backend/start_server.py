#!/usr/bin/env python3
import sys
import os

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from app.main import app
import uvicorn

if __name__ == "__main__":
    import os

    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))  # Default to 1 worker for development
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    print("🚀 Ouartech ")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"👷 Workers: {workers}")
    print(f"🔄 Reload: {reload}")
    print(f"🔧 Background Processing: {'Enabled' if workers == 1 else 'Multi-worker mode'}")

    # Validate configuration
    if workers > 1:
        print("⚠️ Warning: Multi-worker mode detected. Background processing will only work in the main process.")
        print("💡 For production with multiple workers, consider using Redis or external task queue.")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        access_log=True,
        log_level="info",
        # Increase maximum request size for larger file uploads
        limit_max_requests=52428800  # 50MB
    )