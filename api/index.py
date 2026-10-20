import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.api.main import app

# Normalization middleware for Vercel URL rewrites
@app.middleware("http")
async def vercel_url_middleware(request, call_next):
    # If Vercel passed the original path in headers, use it
    matched_path = request.headers.get("x-matched-path") or request.headers.get("x-forwarded-uri")
    if matched_path and not matched_path.startswith("/api/index"):
        request.scope["path"] = matched_path
    else:
        path = request.scope.get("path", "")
        for prefix in ["/api/index.py", "/api/index", "/api"]:
            if path == prefix:
                request.scope["path"] = "/"
                break
            elif path.startswith(prefix + "/"):
                request.scope["path"] = path[len(prefix):]
                break
    return await call_next(request)
