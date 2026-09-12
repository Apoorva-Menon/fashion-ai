from typing import Optional, List
from pydantic import BaseModel, Field

class PerspectiveImages(BaseModel):
    front_image_path: str = Field(..., description="Path to front-facing full body portrait")
    side_image_path: Optional[str] = Field(None, description="Path to side profile full body portrait")
    angled_image_path: Optional[str] = Field(None, description="Path to 45-degree angled portrait")

class UserMetrics(BaseModel):
    height_cm: float = Field(..., description="User height in centimeters (e.g., 172.5)")
    weight_kg: float = Field(..., description="User weight in kilograms (e.g., 68.0)")
    reported_skin_color: str = Field(..., description="Self-reported skin tone / undertone description")
    gender_expression: Optional[str] = Field("unspecified", description="Styling preference: feminine, masculine, unisex, etc.")

class UserProfile(BaseModel):
    profile_id: str = Field(..., description="Unique profile identifier")
    images: PerspectiveImages
    metrics: UserMetrics
