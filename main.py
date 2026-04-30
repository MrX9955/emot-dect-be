"""
Emotion Detection API — FastAPI entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import get_settings
from database import connect_db, close_db
from routes.auth_routes import router as auth_router
from routes.emotion_routes import router as emotion_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    await connect_db()
    yield
    await close_db()


app = FastAPI(
    title="Emotion Detection API",
    description="Real-time facial and speech emotion detection with user authentication.",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
origins = settings.cors_origins_list

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api")
app.include_router(emotion_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "message": "Emotion Detection API is running 🚀",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
