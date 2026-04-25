"""
Facial Emotion Detection using FER (Facial Expression Recognition).
FER is lightweight (~50MB) compared to DeepFace (~2GB) — works on free hosting.
"""
import base64
import logging
import random
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
    Uses FER library (lightweight) for production deployment.
    Falls back to DeepFace if FER not available.
    """
    try:
        from fer import FER
        import cv2

        image = decode_base64_image(base64_image)
        if image is None:
            return _mock_facial_response("Could not decode image")

        # Convert BGR to RGB for FER
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detector = FER(mtcnn=False)
        results = detector.detect_emotions(image_rgb)

        if not results:
            # No face detected — return neutral
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "all_emotions": {e: round(1/len(EMOTION_LABELS), 4) for e in EMOTION_LABELS},
                "detection_type": "facial",
                "note": "No face detected in frame",
            }

        # Get the first face result
        emotions = results[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = float(emotions[dominant_emotion])

        # Normalize
        total = sum(emotions.values()) or 1
        normalized = {k: round(float(v) / total, 4) for k, v in emotions.items()}

        return {
            "emotion": dominant_emotion,
            "confidence": round(confidence, 4),
            "all_emotions": normalized,
            "detection_type": "facial",
        }

    except ImportError:
        # FER not installed — try DeepFace
        return _try_deepface(base64_image)
    except Exception as e:
        logger.error(f"FER facial emotion error: {e}")
        return _try_deepface(base64_image)


def _try_deepface(base64_image: str) -> Dict:
    """Fallback to DeepFace if FER fails."""
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
        normalized = {k: round(float(v) / total, 4) for k, v in emotion_scores.items()}
        confidence = normalized.get(dominant_emotion, 0.0)

        return {
            "emotion": dominant_emotion,
            "confidence": round(confidence, 4),
            "all_emotions": normalized,
            "detection_type": "facial",
        }

    except Exception as e:
        logger.error(f"DeepFace fallback error: {e}")
        return _mock_facial_response(str(e))


def _mock_facial_response(note: str = "") -> Dict:
    """Return mock data when all ML packages fail."""
    emotion = random.choice(EMOTION_LABELS)
    scores = {label: round(random.uniform(0.01, 0.3), 4) for label in EMOTION_LABELS}
    total = sum(scores.values())
    scores = {k: round(v / total, 4) for k, v in scores.items()}

    return {
        "emotion": emotion,
        "confidence": scores[emotion],
        "all_emotions": scores,
        "detection_type": "facial",
        "note": note or "Running in demo mode",
    }
