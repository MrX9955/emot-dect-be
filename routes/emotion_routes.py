"""
Emotion detection routes:
  POST /emotion/facial  — analyze a base64 image frame
  POST /emotion/speech  — analyze an uploaded audio file
  GET  /emotion/history — get the user's emotion history
  GET  /emotion/stats   — get aggregated emotion statistics
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import List, Optional
from database import get_database
from auth.dependencies import get_current_user
from ml_models.facial_emotion import analyze_facial_emotion
from ml_models.speech_emotion import analyze_speech_emotion


def to_python_types(obj):
    """
    Recursively convert numpy types to native Python types
    so MongoDB (bson) can serialize them without errors.
    """
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    return obj

router = APIRouter(prefix="/emotion", tags=["Emotion Detection"])


class FacialRequest(BaseModel):
    """Request body for facial emotion detection."""
    image: str  # base64-encoded image (with or without data URI prefix)


# ─── Facial Detection ────────────────────────────────────────────────────────

@router.post("/facial")
async def detect_facial_emotion(
    request: FacialRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Analyze a webcam frame for facial emotion.
    Accepts a base64-encoded JPEG/PNG image.
    """
    if not request.image:
        raise HTTPException(status_code=400, detail="Image data is required")

    result = analyze_facial_emotion(request.image)

    # Persist result to database — convert numpy types to plain Python
    db = get_database()
    record = {
        "user_email": current_user["email"],
        "detection_type": "facial",
        "emotion": str(result["emotion"]),
        "confidence": float(result["confidence"]) if result.get("confidence") is not None else None,
        "all_emotions": to_python_types(result.get("all_emotions")),
        "timestamp": datetime.now(timezone.utc),
    }
    await db["emotion_history"].insert_one(record)

    return to_python_types(result)


# ─── Speech Detection ─────────────────────────────────────────────────────────

@router.post("/speech")
async def detect_speech_emotion(
    audio: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Analyze an audio recording for speech emotion.
    Accepts WAV, WebM, MP3, or OGG files (max 10 MB).
    """
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Audio file too large (max 10 MB)")

    # Determine file extension from content type or filename
    content_type = audio.content_type or ""
    filename = audio.filename or ""
    if "wav" in content_type or filename.endswith(".wav"):
        ext = "wav"
    elif "webm" in content_type or filename.endswith(".webm"):
        ext = "webm"
    elif "ogg" in content_type or filename.endswith(".ogg"):
        ext = "ogg"
    elif "mp3" in content_type or "mpeg" in content_type or filename.endswith(".mp3"):
        ext = "mp3"
    else:
        ext = "webm"  # browser default

    result = analyze_speech_emotion(audio_bytes, ext)

    # Persist result to database — convert numpy types to plain Python
    db = get_database()
    record = {
        "user_email": current_user["email"],
        "detection_type": "speech",
        "emotion": str(result["emotion"]),
        "confidence": float(result["confidence"]) if result.get("confidence") is not None else None,
        "all_emotions": to_python_types(result.get("all_emotions")),
        "timestamp": datetime.now(timezone.utc),
    }
    await db["emotion_history"].insert_one(record)

    return to_python_types(result)


# ─── History ──────────────────────────────────────────────────────────────────

@router.get("/history")
async def get_emotion_history(
    limit: int = 50,
    detection_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieve the authenticated user's emotion detection history.
    Optionally filter by detection_type ('facial' or 'speech').
    """
    db = get_database()
    query = {"user_email": current_user["email"]}
    if detection_type in ("facial", "speech"):
        query["detection_type"] = detection_type

    cursor = db["emotion_history"].find(query).sort("timestamp", -1).limit(limit)
    records = []
    async for doc in cursor:
        records.append({
            "id": str(doc["_id"]),
            "detection_type": doc["detection_type"],
            "emotion": doc["emotion"],
            "confidence": doc.get("confidence"),
            "timestamp": doc["timestamp"].isoformat(),
        })

    return {"history": records, "total": len(records)}


# ─── Stats ────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_emotion_stats(
    current_user: dict = Depends(get_current_user),
):
    """
    Return aggregated emotion frequency statistics for the current user.
    Used to power the dashboard charts.
    """
    db = get_database()

    pipeline = [
        {"$match": {"user_email": current_user["email"]}},
        {
            "$group": {
                "_id": {
                    "emotion": "$emotion",
                    "detection_type": "$detection_type",
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"count": -1}},
    ]

    cursor = db["emotion_history"].aggregate(pipeline)
    facial_stats = {}
    speech_stats = {}

    async for doc in cursor:
        emotion = doc["_id"]["emotion"]
        dtype = doc["_id"]["detection_type"]
        count = doc["count"]
        if dtype == "facial":
            facial_stats[emotion] = count
        elif dtype == "speech":
            speech_stats[emotion] = count

    # Total detections
    total = await db["emotion_history"].count_documents(
        {"user_email": current_user["email"]}
    )

    return {
        "total_detections": total,
        "facial": facial_stats,
        "speech": speech_stats,
    }
