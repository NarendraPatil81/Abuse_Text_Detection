from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Import routers
from app.routes.document import router as document_router
from app.routes.chat import router as chat_router
from app.routes.auth import router as auth_router
from app.database import init_db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("data", exist_ok=True)

# Initialize database
init_db()

# Create FastAPI app with explicit OpenAPI configuration
app = FastAPI(
    title="RAG API with Authentication",
    description="RAG API with simplified username authentication",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with correct prefixes
# The auth router should be mounted at /auth
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "RAG API is running with authentication",
        "documentation": "/docs",
        "authentication": "Use /auth/login with your username to obtain a JWT token",
        "example": {
            "request": {"username": "admin"},
            "usage": "Add the returned access_token to Authorization header as 'Bearer {token}'"
        },
        "status": "active"
    }

# Additional health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
