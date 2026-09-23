from typing import Optional
from pydantic import BaseModel, field_validator, ConfigDict
from settings import (
    STABLE_AUDIO_MODELS,
    STABLE_AUDIO_3_MODELS,
    MIN_DURATION,
    MAX_DURATION,
)


class AudioProcessRequest(BaseModel):
    prompt: Optional[str] = None
    duration: Optional[int] = 10
    model: Optional[str] = "stable-audio-open-1.0"
    bpm: Optional[int] = None
    key: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        allowed = (
            {"stable-audio-open-1.0"}
            | set(STABLE_AUDIO_MODELS.keys())
            | set(STABLE_AUDIO_3_MODELS.keys())
        )
        if v not in allowed:
            raise ValueError(f"model must be one of {allowed}")
        return v

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v, info):
        if info.data.get("action") == "generate" and v is not None:
            if not (MIN_DURATION <= v <= MAX_DURATION):
                raise ValueError(
                    f"duration must be between {MIN_DURATION} and {MAX_DURATION}"
                )
        return v

    @field_validator("prompt", mode="before")
    @classmethod
    def validate_prompt(cls, v, info):
        action = info.data.get("action") if info.data else None
        if action == "generate":
            if v is None or not str(v).strip():
                raise ValueError("prompt is required for generate action")
        return v
