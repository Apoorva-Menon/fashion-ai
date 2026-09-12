from typing import List, Literal, Optional
from pydantic import BaseModel, Field

BodyShapeType = Literal["hourglass", "pear", "inverted_triangle", "rectangle", "apple", "athletic"]
ColorSeasonType = Literal["warm_spring", "cool_summer", "deep_autumn", "bright_winter", "neutral"]

class Proportions(BaseModel):
    shoulder_to_hip_ratio: str = Field(..., description="Description of shoulder vs hip balance (e.g., 'shoulders wider than hips', 'balanced')")
    torso_to_leg_ratio: str = Field(..., description="Proportion description (e.g., 'long torso, shorter legs', 'balanced')")
    vertical_line: str = Field(..., description="Visual height perception (e.g., 'petite', 'moderate', 'elongated')")
    waist_definition: str = Field(..., description="Level of waist indent (e.g., 'sharply defined', 'moderate', 'straight/undefined')")

class ToneAnalysis(BaseModel):
    undertone: Literal["warm", "cool", "neutral", "olive"] = Field(..., description="Underlying skin undertone")
    contrast_level: Literal["low", "medium", "high"] = Field(..., description="Contrast between skin, hair, and eyes")
    dominant_hex_colors: List[str] = Field(default_factory=list, description="Dominant sampled skin/hair hex codes")
    recommended_season: ColorSeasonType = Field(..., description="Best matching color season")

class FacialStructure(BaseModel):
    face_shape: str = Field(..., description="Face shape (e.g., oval, round, square, heart, oblong)")
    jawline: str = Field(..., description="Jawline definition (e.g., soft, angular, sharp)")

class BodyFeatures(BaseModel):
    body_shape: BodyShapeType = Field(..., description="Primary body shape classification")
    color_season: ColorSeasonType = Field(..., description="Primary color season category")
    proportions: Proportions
    tone: ToneAnalysis
    face: FacialStructure
    overall_summary: str = Field(..., description="Summary of body geometry and color profile")
