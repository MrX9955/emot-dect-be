"""
Facial Emotion Detection using DeepFace.
Accepts base64-encoded image frames from the frontend.
Imports are lazy — ML packages only loaded when first called.
"""
import base64
import logging
from typing import Dict

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def decode_base64_image(base64_string: str):
    """Decode a base64 image string to a numpy array."""
    import cv2
    import numpy as np

    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


def analyze_facial_emotion(base64_image: str) -> Dict:
    """
    Analyze facial emotion from a base64-encoded image.
    Returns mock data if DeepFace/OpenCV are not installed.
    """
    try:
        from deepface import DeepFace

        image = decode_base64_image(base64_image)
        if image is None:
            return _mock_facial_response("Could not decode image")

        results = DeepFace.analyze(
            img_path=image,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )

        result = results[0] if isinstance(results, list) else results
        dominant_emotion = result.get("dominant_emotion", "neutral")
        emotion_scores = result.get("emotion", {})

        total = sum(emotion_scores.values()) or 1
        normalized = {k: round(v / total, 4) for k, v in emotion_scores.items()}
        confidence = normalized.get(dominant_emotion, 0.0)

        return {
            "emotion": dominant_emotion,
            "confidence": round(confidence, 4),
            "all_emotions": normalized,
            "detection_type": "facial",
        }

    except ImportError:
        return _mock_facial_response("DeepFace not installed — showing demo data")
    except Exception as e:
        logger.error(f"Facial emotion analysis error: {e}")
        return _mock_facial_response(str(e))


def _mock_facial_response(note: str = "") -> Dict:
    """Return mock facial emotion data when ML packages are unavailable."""
    import random
    emotion = random.choice(EMOTION_LABELS)
    scores = {label: round(random.uniform(0.01, 0.3), 4) for label in EMOTION_LABELS}
    total = sum(scores.values())
    scores = {k: round(v / total, 4) for k, v in scores.items()}
    scores[emotion] = round(max(scores.values()) + 0.1, 4)

    return {
        "emotion": emotion,
        "confidence": scores[emotion],
        "all_emotions": scores,
        "detection_type": "facial",
        "note": note or "Running in demo mode — install deepface for real detection",
    }
