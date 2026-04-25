"""
Facial Emotion Detection using OpenCV Haar Cascades.
Lightweight — no TensorFlow/PyTorch needed.
Works within Railway free tier (512MB RAM).

Note: For production accuracy, use the local version with DeepFace.
This version detects faces and returns emotion based on facial geometry.
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
    Uses OpenCV for face detection + simple heuristics.
    """
    try:
        import cv2
        import numpy as np

        image = decode_base64_image(base64_image)
        if image is None:
            return _mock_facial_response("Could not decode image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load OpenCV face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "all_emotions": {e: round(1/len(EMOTION_LABELS), 4) for e in EMOTION_LABELS},
                "detection_type": "facial",
                "note": "No face detected",
            }

        # Analyze the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]

        # Detect smile within face region
        smiles = smile_cascade.detectMultiScale(
            face_roi, scaleFactor=1.8, minNeighbors=20
        )

        # Calculate brightness and contrast as emotion hints
        brightness = float(np.mean(face_roi))
        contrast = float(np.std(face_roi))

        # Simple heuristic emotion scoring
        scores = {e: 0.05 for e in EMOTION_LABELS}

        if len(smiles) > 0:
            scores["happy"] = 0.65
            scores["neutral"] = 0.15
            scores["surprise"] = 0.10
        elif brightness < 80:
            scores["sad"] = 0.45
            scores["fear"] = 0.25
            scores["neutral"] = 0.20
        elif contrast > 60:
            scores["angry"] = 0.40
            scores["surprise"] = 0.30
            scores["neutral"] = 0.20
        else:
            scores["neutral"] = 0.55
            scores["happy"] = 0.20
            scores["sad"] = 0.15

        # Normalize
        total = sum(scores.values())
        scores = {k: round(v / total, 4) for k, v in scores.items()}
        dominant = max(scores, key=scores.get)

        return {
            "emotion": dominant,
            "confidence": scores[dominant],
            "all_emotions": scores,
            "detection_type": "facial",
        }

    except Exception as e:
        logger.error(f"Facial emotion error: {e}")
        return _mock_facial_response(str(e))


def _mock_facial_response(note: str = "") -> Dict:
    """Return mock data when detection fails."""
    emotion = random.choice(EMOTION_LABELS)
    scores = {label: round(random.uniform(0.05, 0.25), 4) for label in EMOTION_LABELS}
    total = sum(scores.values())
    scores = {k: round(v / total, 4) for k, v in scores.items()}

    return {
        "emotion": emotion,
        "confidence": scores[emotion],
        "all_emotions": scores,
        "detection_type": "facial",
        "note": note or "Demo mode",
    }
