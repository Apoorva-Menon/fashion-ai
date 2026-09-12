from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class GarmentArtifact(BaseModel):
    artifact_id: str = Field(..., description="Unique ID of the garment or outfit artifact")
    name: str = Field(..., description="Name of garment/outfit (e.g. 'Oversized Hoodie with Baggy Cargo Pants')")
    category: str = Field(..., description="Apparel category (e.g., streetwear, athleisure, formal)")
    image_path: Optional[str] = Field(None, description="Path to garment image artifact")
    color: str = Field(..., description="Color or pattern description")
    fabric: str = Field(..., description="Fabric type (e.g. 'Heavyweight fleece cotton & rigid twill')")
    silhouette: str = Field(..., description="Cut style (e.g. 'Oversized top, relaxed loose bottom')")

class TryOnVerdict(BaseModel):
    artifact_id: str
    garment_name: str
    verdict: Literal["good", "not_good", "conditional"] = Field(..., description="Suitability rating for user")
    score_out_of_10: float = Field(..., description="Compatibility score")
    verdict_headline: str = Field(..., description="Snappy one-sentence verdict")
    detailed_rationale: str = Field(..., description="Why this works or fails based on user's body shape and tone")
    how_to_style_or_fix: List[str] = Field(default_factory=list, description="Adjustments (e.g., half-tuck, belt, color swap)")
    tryon_video_subtitle_script: str = Field(..., description="Video subtitle / narration text for the live try-on review")
    preview_output_path: Optional[str] = Field(None, description="Path to generated visual try-on comparison/render")
