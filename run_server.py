# Wrapper — does NOT modify vision.py.
# Patches CORS onto the imported app BEFORE uvicorn starts handling requests.
# Must use app.add_middleware() called at module load time (this file is the
# module uvicorn imports, so it runs before any request comes in — that's fine).

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Import the app object from vision.py
from newbackend import app  # noqa: E402

# Add CORS — this must be done before the first request, which it is.
# Starlette middleware is inserted at the front of the chain regardless of
# when add_middleware() is called relative to route definitions.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # any origin (fine for local testing)
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# Serve the test HTML at the root so it's same-origin as the API calls
@app.get("/ui")
def serve_frontend():
    return FileResponse("test_frontend.html")
