"""
Emotion detection result schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


class EmotionResult(BaseModel):
    """Schema for storing emotion detection results."""
    user_email: str
    detection_type: str  # "facial" or "speech"
    emotion: str
    confidence: Optional[float] = None
    additional_data: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EmotionResponse(BaseModel):
    """Schema for emotion detection API response."""
    emotion: str
    confidence: Optional[float] = None
    all_emotions: Optional[Dict[str, float]] = None
    detection_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EmotionHistory(BaseModel):
    """Schema for emotion history response."""
    id: str
    detection_type: str
    emotion: str
    confidence: Optional[float]
    timestamp: datetime
