import os

import uvicorn

uvicorn.run(
    "mock_backend.app:app",
    host=os.environ.get("MOCK_BACKEND_HOST", "127.0.0.1"),
    port=int(os.environ.get("MOCK_BACKEND_PORT", "8001")),
)
