"""
Speech Emotion Detection using a trained CNN model.
Imports are lazy — ML packages only loaded when first called.
"""
import os
import logging
import tempfile
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SPEECH_EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

# Use absolute path resolution — works regardless of working directory
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(_BASE_DIR, "trained_models", "speech_model.h5")
ENCODER_PATH = os.path.join(_BASE_DIR, "trained_models", "label_encoder.pkl")

_speech_model = None
_label_encoder = None


def load_speech_model():
    """Load the trained speech emotion model and label encoder (cached)."""
    global _speech_model, _label_encoder
    if _speech_model is not None:
        return _speech_model

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        import tensorflow as tf
        import pickle

        _speech_model = tf.keras.models.load_model(MODEL_PATH)

        # Load label encoder if available
        if os.path.exists(ENCODER_PATH):
            with open(ENCODER_PATH, "rb") as f:
                _label_encoder = pickle.load(f)
            logger.info("✅ Speech model + label encoder loaded.")
        else:
            logger.info("✅ Speech model loaded (no encoder found).")

        return _speech_model
    except Exception as e:
        logger.error(f"Failed to load speech model: {e}")
        return None


def extract_audio_features(audio_path: str, max_pad_len: int = 174) -> Optional[object]:
    """Extract MFCC + Chroma + Mel features from an audio file."""
    try:
        import librosa
        import numpy as np

        # librosa with audioread fallback handles most formats
        # For webm/ogg from browser, try loading directly first
        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=3.0)
        except Exception:
            # Try converting webm to wav using soundfile fallback
            import soundfile as sf
            data, sr = sf.read(audio_path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)  # stereo to mono
            import resampy
            y = resampy.resample(data.astype(np.float32), sr, 22050)
            sr = 22050

        if len(y) == 0:
            logger.error("Audio file is empty")
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

        def pad(feat):
            if feat.shape[1] < max_pad_len:
                return np.pad(feat, ((0, 0), (0, max_pad_len - feat.shape[1])), mode="constant")
            return feat[:, :max_pad_len]

        features = np.vstack([pad(mfcc), pad(chroma), pad(mel)])
        logger.info(f"Features extracted successfully: {features.shape}")
        return features[np.newaxis, ..., np.newaxis].astype(np.float32)

    except ImportError as e:
        logger.error(f"librosa not installed: {e}")
        return None
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None


def analyze_speech_emotion(audio_bytes: bytes, file_extension: str = "wav") -> Dict:
    """Analyze speech emotion from raw audio bytes."""
    with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = load_speech_model()
        features = extract_audio_features(tmp_path)

        if model is None:
            return _mock_speech_response("Speech model not found — run train_speech_model.py")

        if features is None:
            return _mock_speech_response("Audio feature extraction failed — check audio format")

        import numpy as np
        predictions = model.predict(features, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))

        # Use label encoder if available, else fall back to SPEECH_EMOTIONS list
        if _label_encoder is not None:
            dominant_emotion = str(_label_encoder.classes_[predicted_idx])
            emotion_labels = [str(c) for c in _label_encoder.classes_]
        else:
            dominant_emotion = SPEECH_EMOTIONS[predicted_idx]
            emotion_labels = SPEECH_EMOTIONS

        confidence = float(predictions[predicted_idx])

        return {
            "emotion": dominant_emotion,
            "confidence": round(confidence, 4),
            "all_emotions": {
                emotion_labels[i]: round(float(predictions[i]), 4)
                for i in range(len(emotion_labels))
            },
            "detection_type": "speech",
        }

    except Exception as e:
        logger.error(f"Speech emotion analysis error: {e}")
        return _mock_speech_response()

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _mock_speech_response() -> Dict:
    """Return mock speech emotion data when ML packages are unavailable."""
    import random
    emotion = random.choice(SPEECH_EMOTIONS)
    import random as r
    scores = {e: round(r.uniform(0.01, 0.25), 4) for e in SPEECH_EMOTIONS}
    total = sum(scores.values())
    scores = {k: round(v / total, 4) for k, v in scores.items()}

    return {
        "emotion": emotion,
        "confidence": round(max(scores.values()), 4),
        "all_emotions": scores,
        "detection_type": "speech",
        "note": "Running in demo mode — install librosa & tensorflow for real detection",
    }
